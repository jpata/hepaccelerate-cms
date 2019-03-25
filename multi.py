import uproot, cupy, awkward
import numpy as np
import glob
from collections import OrderedDict
import math
import numba
import time
from concurrent.futures import ThreadPoolExecutor
from dask.distributed import as_completed, wait

from analysis import *

from distributed import Client, LocalCluster

"""
Preloads all data into memory.

Args:
    dataset_files (Dict[str, List[str]]): a dictionary of dataset name to the list of filenames for each dataset

returns:
    Dict[str, List[Dict[str, JaggedArray]]]
    
    For each dataset, a list of awkward-array JaggedArray dictionaries 
"""
def preload_data(dataset_files, arrays_to_load):
    ret = {}
    for dsname, dsfiles in dataset_files.items():
        ret[dsname] = [load_arrays(fn, arrays_to_load) for fn in dsfiles]
    return ret

def load_arrays(fn, arrays_to_load):
    #print("opening file {0}".format(fn))
    fi = uproot.open(fn)
    tt = fi.get("Events")
    #with ThreadPoolExecutor(max_workers=1) as executor:
    #    arrs = tt.arrays(arrays_to_load, executor=executor)
    arrs = tt.arrays(arrays_to_load)
    #print("loaded {0} arrays, shape=({1},)".format(len(arrs), tt.numentries))
    return arrs

def make_objects_gpu(arrs):
    muons = JaggedStruct.from_arraydict(
        {k: v for k, v in arrs.items() if "Muon_" in str(k)},
        "Muon_", cupy
    )
    jets = JaggedStruct.from_arraydict({
        k: v for k, v in arrs.items() if "Jet_" in str(k)
        }, "Jet_", cupy
    )
    return muons, jets


class Histogram:
    def __init__(self, contents, contents_w2, edges):
        self.contents = cupy.asnumpy(contents)
        self.contents_w2 = cupy.asnumpy(contents_w2)
        self.edges = cupy.asnumpy(edges)
        
    @staticmethod
    def from_vector(data, weights, bins):        
        out_w = cupy.zeros(len(bins) - 1, dtype=np.float32)
        out_w2 = cupy.zeros(len(bins) - 1, dtype=np.float32)
        fill_histogram[32, 1024](data, weights, bins, out_w, out_w2)
        return Histogram(out_w, out_w2, bins)
    
    def __add__(self, other):
        assert(np.all(self.edges == other.edges))
        return Histogram(self.contents +  other.contents, self.contents_w2 +  other.contents_w2, self.edges)

    def plot(self):
        line = plt.step(self.edges[:-1], self.contents, where="mid")
        plt.errorbar(midpoints(self.edges), self.contents, np.sqrt(self.contents), lw=0, elinewidth=1, color=line[0].get_color())
    
    
def get_histogram(data, weights, bins):
    return Histogram.from_vector(data, weights, bins)


class Results(dict):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __add__(self, other):
        d0 = self
        d1 = other
        
        d_ret = Results({})
        k0 = set(d0.keys())
        k1 = set(d1.keys())

        for k in k0.intersection(k1):
            d_ret[k] = d0[k] + d1[k]

        for k in k0.difference(k1):
            d_ret[k] = d0[k]

        for k in k1.difference(k0):
            d_ret[k] = d1[k]

        return d_ret

class JaggedStruct(object):
    def __init__(self, offsets, attrs_data, numpy_lib=np):
        self.numpy_lib = numpy_lib
        
        self.offsets = offsets
        self.attrs_data = attrs_data
        
        num_items = None
        for (k, v) in self.attrs_data.items():
            num_items_next = len(v)
            if num_items and num_items != num_items_next:
                raise AttributeError("Mismatched attribute {0}".format(k))
            else:
                num_items = num_items_next
            setattr(self, k, v)
        self.num_items = num_items
    
        self.masks = {}
        self.masks["all"] = self.make_mask()
    
    def make_mask(self):
        return self.numpy_lib.ones(self.num_items, dtype=self.numpy_lib.bool)
    
    def mask(self, name):
        if not name in self.masks.keys():
            self.masks[name] = self.make_mask()
        return self.masks[name]
    
    def size(self):
        size_tot = self.offsets.size
        for k, v in self.attrs_data.items():
            size_tot += v.size
        return size_tot
    
    def __len__(self):
        return len(self.offsets) - 1
    
    @staticmethod
    def from_arraydict(arraydict, prefix, numpy_lib=np):
        ks = [k for k in arraydict.keys() if prefix in str(k, 'ascii')]
        k0 = ks[0]
        return JaggedStruct(
            numpy_lib.array(arraydict[k0].offsets),
            {str(k, 'ascii').replace(prefix, ""): numpy_lib.array(v.content)
             for (k,v) in arraydict.items()},
            numpy_lib=numpy_lib
        )

    def savez(self, path):
        with open(path, "wb") as of:
            self.numpy_lib.savez(of, offsets=self.offsets, **self.attrs_data)
    
    @staticmethod 
    def load(path, numpy_lib):
        with open(path, "rb") as of:
            fi = numpy_lib.load(of)
            ks = [f for f in fi.npz_file.files if f!="offsets"]
            return JaggedStruct(fi["offsets"], {k: fi.npz_file[k] for k in ks})  

def get_selected_muons(muons, mu_pt_cut_leading, mu_pt_cut_subleading, mu_iso_cut):
    
    passes_iso = muons.pfRelIso03_all < mu_iso_cut
    passes_id = muons.mediumId == 1
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    
    #select events with at least 2 muons passing cuts
    muons_passing_id = passes_iso & passes_id & passes_subleading_pt
    
    events_all = cupy.ones(len(muons), dtype=cupy.bool)
    events_passes_muid = sum_in_offsets(muons, muons_passing_id, events_all, muons.masks["all"], cupy.int8) >= 2
    events_passes_leading_pt = sum_in_offsets(muons, muons_passing_id & passes_leading_pt, events_all, muons.masks["all"], cupy.int8) >= 1
    events_passes_subleading_pt = sum_in_offsets(muons, muons_passing_id & passes_subleading_pt, events_all, muons.masks["all"], cupy.int8) >= 2

    base_event_sel = events_passes_muid & events_passes_leading_pt & events_passes_subleading_pt
    
    muons_passing_os = select_muons_opposite_sign(muons, muons_passing_id & passes_subleading_pt)
    events_passes_os = sum_in_offsets(muons, muons_passing_os, events_all, muons.masks["all"], cupy.int8) == 2
    
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    
    return {
         "selected_events": final_event_sel,
         "selected_muons": final_muon_sel,
    }


def get_selected_jets(jets, muons, mask_muons, jet_pt_cut, jet_eta_cut, dr_cut):
    events_all = cupy.ones(len(jets), dtype=cupy.bool)

    jets_pass_dr = mask_deltar_first(jets, jets.masks["all"], muons, mask_muons, dr_cut)
    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = (jets.pt > jet_pt_cut) & (cupy.abs(jets.eta) < jet_eta_cut) & (((jets.jetId & 2)>>1)==1) & jets_pass_dr

    num_jets = sum_in_offsets(jets, selected_jets, events_all, jets.masks["all"], cupy.int8)

    return {
        "selected_jets": selected_jets,
        "num_jets": num_jets
    }

def compute_inv_mass(objects, mask_events, mask_objects):
    pt = objects.pt
    eta = objects.eta
    phi = objects.phi
    mass = objects.mass

    px = pt * cupy.cos(phi)
    py = pt * cupy.sin(phi)
    pz = pt * cupy.sinh(eta)

    px_total = sum_in_offsets(objects, px, mask_events, mask_objects)
    py_total = sum_in_offsets(objects, py, mask_events, mask_objects)
    pz_total = sum_in_offsets(objects, pz, mask_events, mask_objects)
    mass_total = sum_in_offsets(objects, mass, mask_events, mask_objects)

    inv_mass = cupy.sqrt(px_total**2 + py_total**2 + pz_total**2 - mass_total**2)
    return inv_mass

def analyze_data(
    muons, jets,
    mu_pt_cut_leading=26, mu_pt_cut_subleading=10,
    mu_iso_cut=0.3, jet_pt_cut=30,
    jet_eta_cut=4.1, jet_mu_drcut=0.2, doverify=True
    ):

    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(muons, mu_pt_cut_leading, mu_pt_cut_subleading, mu_iso_cut)
    
    if doverify:
        z = sum_in_offsets(muons, ret_mu["selected_muons"], ret_mu["selected_events"], ret_mu["selected_muons"], dtype=cupy.int8)
        assert(cupy.all(z[z!=0] == 2))

    #get the passing jets for events that pass muon selection
    ret_jet = get_selected_jets(jets, muons, ret_mu["selected_muons"], jet_pt_cut, jet_eta_cut, jet_mu_drcut)    
    if doverify:
        z = min_in_offsets(jets, jets.pt, ret_mu["selected_events"], ret_jet["selected_jets"])
        assert(cupy.all(z[z>0] > jet_pt_cut))

        
    inv_mass = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"])
    
    inds = 0*cupy.ones(len(muons), dtype=cupy.int32)
    leading_muon_pt = get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_muon_eta = get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_jet_pt = get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    leading_jet_eta = get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    
    inds[:] = 1
    subleading_muon_pt = get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_muon_eta = get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_jet_pt = get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    subleading_jet_eta = get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    
    if doverify:
        assert(cupy.all(leading_muon_pt[leading_muon_pt>0] > mu_pt_cut_leading))
        assert(cupy.all(subleading_muon_pt[subleading_muon_pt>0] > mu_pt_cut_subleading))

    
    weights = cupy.ones(len(muons), dtype=cupy.float32)
    hist_inv_mass_d = get_histogram(inv_mass[ret_mu["selected_events"]], weights, cupy.linspace(0,500,201))

    #get histograms of leading and subleading muon momenta
    hist_leading_muon_pt_d = get_histogram(leading_muon_pt[ret_mu["selected_events"]], weights, cupy.linspace(0.0, 200.0, 401))
    hist_subleading_muon_pt_d = get_histogram(subleading_muon_pt[ret_mu["selected_events"]], weights, cupy.linspace(0.0, 200.0, 401))

    #get histograms of leading and subleading muon eta
    hist_leading_muon_eta_d = get_histogram(leading_muon_eta[ret_mu["selected_events"]], weights, cupy.linspace(-4.0, 4.0, 401))
    hist_subleading_muon_eta_d = get_histogram(subleading_muon_eta[ret_mu["selected_events"]], weights, cupy.linspace(-4.0, 4.0, 401))

    hist_leading_jet_pt_d = get_histogram(leading_jet_pt[ret_mu["selected_events"] & (ret_jet["num_jets"]>=1)], weights, cupy.linspace(0, 300.0, 401))
    hist_subleading_jet_pt_d = get_histogram(subleading_jet_pt[ret_mu["selected_events"] & (ret_jet["num_jets"]>=2)], weights, cupy.linspace(0, 300.0, 401))

    return {
        "hist_inv_mass_d": hist_inv_mass_d,
        "hist_mu0_pt": hist_leading_muon_pt_d,
        "hist_mu1_pt": hist_subleading_muon_pt_d,
        
        "hist_mu0_eta": hist_leading_muon_eta_d,
        "hist_mu1_eta": hist_subleading_muon_eta_d,
        
#         #"hist_dimuon_mass_gen": hist_inv_mass_gen_d,        
#         "hist_dimuon_mass": hist_inv_mass_d,        
        
        "hist_leading_jet_pt": hist_leading_jet_pt_d,        
        "hist_subleading_jet_pt": hist_subleading_jet_pt_d,        
    }

arrays_ev = [
    "PV_npvsGood", "Flag_METFilters", "Flag_goodVertices", "Generator_weight", "genWeight", "HLT_IsoMu24"
]
arrays_jet = [
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId"
]

arrays_muon = [
    "nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso03_all", "Muon_mediumId", "Muon_charge"
]

arrays_to_load = arrays_ev + arrays_jet + arrays_muon

def load(filename):
    return load_arrays(filename, arrays_to_load) 

def transfer_to_gpu(data):
    return make_objects_gpu(data) 

def analyze(gpu_data):
    ret = analyze_data(gpu_data[0], gpu_data[1])
    del gpu_data
    return ret

def load_and_analyze(data):
    gpu_data = transfer_to_gpu(data)
    ret = analyze(gpu_data)
    return ret

def result(ret):
    return Results(ret)

def sum_result(rets):
    return sum(rets, Results({}))

if __name__ == "__main__":
    cluster = LocalCluster(ip="0.0.0.0", n_workers=1, threads_per_worker=16, processes=False, diagnostics_port=8181, memory_limit="60GB")
    #cluster.start_worker(ncores=16, memory_limit="60GB")
    client = Client(cluster)
    print(client)
    
    filenames = glob.glob("/nvmedata/store/mc/RunIIFall17NanoAOD/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", recursive=True)[:100]

    dsk = {}
    for ifn, fn in enumerate(filenames):
        dsk['load-{0}'.format(ifn)] = (load, fn)
        dsk['transfer-{0}'.format(ifn)] = (transfer_to_gpu, 'load-{0}'.format(ifn)) 
        dsk['analyze-{0}'.format(ifn)] = (analyze, 'transfer-{0}'.format(ifn))
        dsk['result-{0}'.format(ifn)] = (result, 'analyze-{0}'.format(ifn))
     
    dsk['sum'] = (sum_result, [k for k in dsk.keys() if "result-" in k])
    ret = client.get(dsk, 'sum')
    
    #results = []
    #for fn in filenames:
    #    load_fut = client.submit(load, fn)
    #    res = client.submit(load_and_analyze, load_fut, priority=10)
    #    results += [res] 

    #for res in as_completed(results):
    #    print(res) 
