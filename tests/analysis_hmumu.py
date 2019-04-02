import glob
import time
import numpy
import os
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"

import numba

import argparse

import sys
import concurrent.futures
import threading
from threading import Thread
from queue import Queue
import queue
import gc

import uproot
import hepaccelerate
from hepaccelerate.utils import Results
from hepaccelerate.utils import NanoAODDataset
from hepaccelerate.utils import Histogram

import fnal_column_analysis_tools
from fnal_column_analysis_tools.lumi_tools import LumiMask, LumiData

genweight_scalefactor = 1e6
chunksize = 1

from types import FrameType
def print_cycles(objects, outstream=sys.stdout, show_progress=False):
    """
    objects:       A list of objects to find cycles in.  It is often useful
                   to pass in gc.garbage to find the cycles that are
                   preventing some objects from being garbage collected.
    outstream:     The stream for output.
    show_progress: If True, print the number of objects reached as they are
                   found.
    """
    def print_path(path):
        for i, step in enumerate(path):
            # next "wraps around"
            next = path[(i + 1) % len(path)]

            outstream.write("   %s -- " % str(type(step)))
            if isinstance(step, dict):
                for key, val in step.items():
                    if val is next:
                        outstream.write("[%s]" % repr(key))
                        break
                    if key is next:
                        outstream.write("[key] = %s" % repr(val))
                        break
            elif isinstance(step, list):
                outstream.write("[%d]" % step.index(next))
            elif isinstance(step, tuple):
                outstream.write("[%d]" % list(step).index(next))
            else:
                outstream.write(repr(step))
            outstream.write(" ->\n")
        outstream.write("\n")
    
    def recurse(obj, start, all, current_path):
        if show_progress:
            outstream.write("%d\r" % len(all))

        all[id(obj)] = None

        referents = gc.get_referents(obj)
        for referent in referents:
            # If we've found our way back to the start, this is
            # a cycle, so print it out
            if referent is start:
                print_path(current_path)

            # Don't go back through the original list of objects, or
            # through temporary references to the object, since those
            # are just an artifact of the cycle detector itself.
            elif referent is objects or isinstance(referent, FrameType): 
                continue

            # We haven't seen this object before, so recurse
            elif id(referent) not in all:
                recurse(referent, start, all, current_path + [obj])

    for obj in objects:
        outstream.write("Examining: %r\n" % obj)
        recurse(obj, obj, { }, [])


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.to_kill = False
    
    def __call__(self):
        return self.to_kill
    
    def set_tokill(self,tokill):
        with self.lock:
            self.to_kill = tokill

def get_histogram(data, weights, bins):
    return Histogram(*histogram_from_vector(data, weights, bins))

def get_selected_muons(muons, trigobj, mask_events, mu_pt_cut_leading, mu_pt_cut_subleading, mu_aeta_cut, mu_iso_cut):
    """
    Given a list of muons in events, selects the muons that pass quality, momentum and charge criteria.
    Selects events that have at least 2 such muons. Selections are made by producing boolean masks.
 
    muons (list of JaggedArray) - The muon content of a given file
    mask_events (array of bool) - a mask of events that are used for muon processing
    mu_pt_cut_leading (float) - the pt cut on the leading muon
    mu_pt_cut_subleading (float) - the pt cut on all other muons
    mu_iso_cut (float) - cut to choose isolated muons

    """ 
    passes_iso = muons.pfRelIso04_all < mu_iso_cut
    passes_id = muons.mediumId == 1
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(muons.eta) < mu_aeta_cut
    
    trigobj.masks["mu"] = (trigobj.id == 13)
  
    muons_matched_to_trigobj = NUMPY_LIB.invert(mask_deltar_first(muons, muons.masks["all"], trigobj, trigobj.masks["mu"], 0.1))
    
    #select muons that pass these cuts
    muons_passing_id = passes_iso & passes_id & passes_subleading_pt & muons_matched_to_trigobj
 
    #select events that have muons passing cuts 
    events_passes_muid = sum_in_offsets(muons, muons_passing_id, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2
    events_passes_leading_pt = sum_in_offsets(muons, muons_passing_id & passes_leading_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 1
    events_passes_subleading_pt = sum_in_offsets(muons, muons_passing_id & passes_subleading_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2

    base_event_sel = mask_events & events_passes_muid & events_passes_leading_pt & events_passes_subleading_pt
    
    muons_passing_os = select_muons_opposite_sign(muons, muons_passing_id & passes_subleading_pt)
    events_passes_os = sum_in_offsets(muons, muons_passing_os, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 2
    
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    
    return {
         "selected_events": final_event_sel,
         "selected_muons": final_muon_sel,
    }


def get_selected_jets(jets, muons, mask_muons, mask_events, jet_pt_cut, jet_eta_cut, dr_cut):
    """
    Given jets and selected muons in events, choose jets that pass quality criteria and that are not dR-matched
    to muons.
 
    """
    jets_pass_dr = mask_deltar_first(jets, jets.masks["all"], muons, mask_muons, dr_cut)
    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = (jets.pt > jet_pt_cut) & (NUMPY_LIB.abs(jets.eta) < jet_eta_cut) & (((jets.jetId & 2)>>1)==1) & jets_pass_dr

    num_jets = sum_in_offsets(jets, selected_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)

    return {
        "selected_jets": selected_jets,
        "num_jets": num_jets
    }

def compute_inv_mass(objects, mask_events, mask_objects):
    pt = objects.pt
    eta = objects.eta
    phi = objects.phi
    mass = objects.mass

    px = pt * NUMPY_LIB.cos(phi)
    py = pt * NUMPY_LIB.sin(phi)
    pz = pt * NUMPY_LIB.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    px_total = sum_in_offsets(objects, px, mask_events, mask_objects)
    py_total = sum_in_offsets(objects, py, mask_events, mask_objects)
    pz_total = sum_in_offsets(objects, pz, mask_events, mask_objects)
    e_total = sum_in_offsets(objects, e, mask_events, mask_objects)
    inv_mass = NUMPY_LIB.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
    return inv_mass

def fill_with_weights(values, weight_dict, mask, bins):
    ret = {
        wn: get_histogram(values[mask], weight_dict[wn][mask], bins)
        for wn in weight_dict.keys()
    }
    return ret

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0

def fix_large_weights(weights, maxw=10.0):
    weights[weights > maxw] = maxw
    weights[:] = weights[:] / NUMPY_LIB.mean(weights)

def compute_pu_weights(pu_corrections_target, weights, mc_nvtx, reco_nvtx):
    pu_edges, (values_nom, values_up, values_down) = pu_corrections_target

    src_pu_hist = get_histogram(mc_nvtx, weights, pu_edges)
    norm = sum(src_pu_hist.contents)
    src_pu_hist.contents = src_pu_hist.contents/norm
    src_pu_hist.contents_w2 = src_pu_hist.contents_w2/norm

    ratio = values_nom / src_pu_hist.contents
    remove_inf_nan(ratio)
    pu_weights = NUMPY_LIB.zeros_like(weights)
    get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio), pu_weights)
    fix_large_weights(pu_weights) 
     
    ratio_up = values_up / src_pu_hist.contents
    remove_inf_nan(ratio_up)
    pu_weights_up = NUMPY_LIB.zeros_like(weights)
    get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio_up), pu_weights_up)
    fix_large_weights(pu_weights_up) 
    
    ratio_down = values_down / src_pu_hist.contents
    remove_inf_nan(ratio_down)
    pu_weights_down = NUMPY_LIB.zeros_like(weights)
    get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio_down), pu_weights_down)
    fix_large_weights(pu_weights_down) 
    
    return pu_weights, pu_weights_up, pu_weights_down

def select_events_trigger(scalars, mask_events):

    flags = [
        "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter",
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_BadPFMuonFilter",
        "Flag_BadChargedCandidateFilter"
    ]
 
    for flag in flags:
        mask_events = mask_events & scalars[flag]
    mask_events = mask_events & scalars["HLT_IsoMu24"] & scalars["PV_npvsGood"]>0

def get_int_lumi(runs, lumis, mask_events, lumidata):
    print("computing integrated luminosity from {0} lumis".format(len(lumis)))
    processed_runs = NUMPY_LIB.asnumpy(runs[mask_events])
    processed_lumis = NUMPY_LIB.asnumpy(lumis[mask_events])
    runs_lumis = np.zeros((processed_runs.shape[0], 2), dtype=np.int64)
    runs_lumis[:, 0] = processed_runs[:]
    runs_lumis[:, 1] = processed_lumis[:]
    lumi_proc = lumidata.get_lumi(runs_lumis)
    print("intlumi=", lumi_proc)
    return lumi_proc

def get_gen_sumweights(filenames):
    sumw = 0
    for fi in filenames:
        ff = uproot.open(fi)
        bl = ff.get("Runs")
        sumw += bl.array("genEventSumw").sum()/genweight_scalefactor
    return sumw

def analyze_data(
    data,
    is_mc=True,
    pu_corrections_target=None,
    lumimask=None,
    lumidata=None,
    mu_pt_cut_leading=30,
    mu_pt_cut_subleading=20,
    mu_aeta_cut=2.4,
    mu_iso_cut=0.25, jet_pt_cut=30,
    jet_eta_cut=4.7, jet_mu_drcut=0.4,
    doverify=True,
    debug=True
    ):
    
    muons = data["Muon"]
    jets = data["Jet"]
    trigobj = data["TrigObj"]
    scalars = data["eventvars"]
 
    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    select_events_trigger(scalars, mask_events)
    if debug:
        print("{0} events passed trigger".format(NUMPY_LIB.sum(mask_events)))

    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"]/genweight_scalefactor
        pu_weights, pu_weights_up, pu_weights_down = compute_pu_weights(pu_corrections_target, weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        weights["puWeight"] = weights["nominal"] * pu_weights
        weights["puWeight_up"] = weights["nominal"] * pu_weights_up
        weights["puWeight_down"] = weights["nominal"] * pu_weights_down
    
    
    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(muons, trigobj, mask_events, mu_pt_cut_leading, mu_pt_cut_subleading, mu_aeta_cut, mu_iso_cut)
    
    if doverify:
        z = sum_in_offsets(muons, ret_mu["selected_muons"], ret_mu["selected_events"], ret_mu["selected_muons"], dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))
    if debug:
        print("{0} events passed muon".format(NUMPY_LIB.sum(ret_mu["selected_events"])))
    
    #for i in range(100):
    #    if ret_mu["selected_events"][i]:
    #        print("ev", i)
    #        for idxmu in range(muons.offsets[i], muons.offsets[i+1]):
    #            if ret_mu["selected_muons"][idxmu]:
    #                print(muons.charge[idxmu], muons.pt[idxmu], muons.eta[idxmu], muons.mediumId[idxmu], muons.pfRelIso04_all[idxmu])
    
    #get the passing jets for events that pass muon selection
    ret_jet = get_selected_jets(jets, muons, ret_mu["selected_muons"], mask_events, jet_pt_cut, jet_eta_cut, jet_mu_drcut)    
    if doverify:
        z = min_in_offsets(jets, jets.pt, ret_mu["selected_events"], ret_jet["selected_jets"])
        assert(NUMPY_LIB.all(z[z>0] > jet_pt_cut))

        
    inv_mass = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"])
    if not is_mc:
        inv_mass[(inv_mass > 120) & (inv_mass < 130)] = 0
 
    inds = NUMPY_LIB.zeros(muons.numevents(), dtype=NUMPY_LIB.int32)
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
        assert(NUMPY_LIB.all(leading_muon_pt[leading_muon_pt>0] > mu_pt_cut_leading))
        assert(NUMPY_LIB.all(subleading_muon_pt[subleading_muon_pt>0] > mu_pt_cut_subleading))
 
  
    hist_npvs_d = fill_with_weights(scalars["PV_npvsGood"], weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0,100,101))
    hist_inv_mass_d = fill_with_weights(inv_mass, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(60,150,201))

    #get histograms of leading and subleading muon momenta
    hist_leading_muon_pt_d = fill_with_weights(leading_muon_pt, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 401))
    hist_subleading_muon_pt_d = fill_with_weights(subleading_muon_pt, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 401))
    
    #create lots of histograms 
    hist_leading_muon_pt_d_weights = fill_with_weights(leading_muon_pt, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 401))  

    #get histograms of leading and subleading muon eta
    hist_leading_muon_eta_d = fill_with_weights(leading_muon_eta, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(-4.0, 4.0, 401))
    hist_subleading_muon_eta_d = fill_with_weights(subleading_muon_eta, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(-4.0, 4.0, 401))

    hist_leading_jet_pt_d = fill_with_weights(leading_jet_pt, weights, ret_mu["selected_events"] & (ret_jet["num_jets"]>=1), NUMPY_LIB.linspace(0, 300.0, 401))
    hist_subleading_jet_pt_d = fill_with_weights(subleading_jet_pt, weights, ret_mu["selected_events"] & (ret_jet["num_jets"]>=2), NUMPY_LIB.linspace(0, 300.0, 401))
    
    int_lumi = 0 
    if not is_mc and not (lumimask is None):
        mask_lumi = lumimask(scalars["run"], scalars["luminosityBlock"])
        mask_events = mask_events & mask_lumi
        #get integrated luminosity in this file
        #if not (lumidata is None): 
        #    int_lumi = get_int_lumi(scalars["run"], scalars["luminosityBlock"], mask_events, lumidata)
    
    ret = Results({
        "int_lumi": int_lumi,
        "hist_npvs_d": Results(hist_npvs_d),
        "hist_inv_mass_d": Results(hist_inv_mass_d),
        "hist_mu0_pt": Results(hist_leading_muon_pt_d),
        
        "hist_mu0_pt_weights": Results(hist_leading_muon_pt_d_weights),
        "hist_mu1_pt": Results(hist_subleading_muon_pt_d),
        
        "hist_mu0_eta": Results(hist_leading_muon_eta_d),
        "hist_mu1_eta": Results(hist_subleading_muon_eta_d),
        
        "hist_leading_jet_pt": Results(hist_leading_jet_pt_d),
        "hist_subleading_jet_pt": Results(hist_subleading_jet_pt_d),        
    })
    
    if is_mc:   
        hist_puweight = get_histogram(pu_weights, NUMPY_LIB.ones_like(pu_weights), NUMPY_LIB.linspace(0, 10, 100))
        ret["hist_puweight"] = hist_puweight
    return ret
 
def load_puhist_target(filename):
    fi = uproot.open(filename)
    
    h = fi["pileup"]
    edges = np.array(h.edges)
    values_nominal = np.array(h.values)
    values_nominal = values_nominal / np.sum(values_nominal)
    
    h = fi["pileup_plus"]
    values_up = np.array(h.values)
    values_up = values_up / np.sum(values_up)
    
    h = fi["pileup_minus"]
    values_down = np.array(h.values)
    values_down = values_down / np.sum(values_down)
    return edges, (values_nominal, values_up, values_down)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cache_data(filenames, is_mc, nworkers=16):
    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        tot_ev = 0
        tot_mb = 0
        for result in executor.map(cache_data_multiproc_worker, [(fn, is_mc) for fn in filenames]):
            tot_ev += result[0]
            tot_mb += result[1]
        print() 
    return tot_ev, tot_mb

def create_dataset(filenames, is_mc):
    arrays_ev = [
        "PV_npvsGood",
        "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_goodVertices",
        "Flag_globalSuperTightHalo2016Filter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter",
        "HLT_IsoMu24",
        "run", "luminosityBlock", "event"
    ]
    if is_mc:
        arrays_ev += ["Pileup_nTrueInt", "Generator_weight", "genWeight"]
    arrays_jet = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId"
    ]
    
    arrays_muon = [
        "nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso04_all", "Muon_mediumId", "Muon_charge"
    ]
    
    arrays_trigobj = [
        "nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id"
    ]
    
    arrays_to_load = arrays_ev + arrays_jet + arrays_muon + arrays_trigobj
    ds = NanoAODDataset(filenames, arrays_to_load, "Events", ["Muon", "Jet", "TrigObj"], arrays_ev)
    return ds

def cache_data_multiproc_worker(args):
    filename, is_mc = args
    t0 = time.time()
    ds = create_dataset([filename], is_mc)
    ds.numpy_lib = np
    ds.preload()
    ds.make_objects()
    ds.to_cache()
    t1 = time.time()
    dt = t1 - t0
    processed_size_mb = ds.memsize()/1024.0/1024.0
    print("built cache for {0}, {1:.2f} MB, {2:.2E} Hz, {3:.2f} MB/s".format(filename, processed_size_mb, len(ds)/dt, processed_size_mb/dt))
    return len(ds), processed_size_mb

class InputGen:
    def __init__(self, paths, is_mc, nthreads):
        self.paths_chunks = list(chunks(paths, chunksize))
        self.chunk_lock = threading.Lock()
        self.loaded_lock = threading.Lock()
        self.num_chunk = 0
        self.num_loaded = 0
        self.is_mc = is_mc
        self.nthreads = nthreads

    def is_done(self):
        return (self.num_chunk == len(self.paths_chunks)) and (self.num_loaded == len(self.paths_chunks))
 
    def __iter__(self):
        return self.generator()

    #did not make this a generator to simplify handling the thread locks
    def nextone(self):

        self.chunk_lock.acquire()
        if self.num_chunk == len(self.paths_chunks):
            self.chunk_lock.release()
            return None

        ds = create_dataset(self.paths_chunks[self.num_chunk], is_mc)
        self.num_chunk += 1
        ds.numpy_lib = numpy
        self.chunk_lock.release()

        ds.from_cache(nthreads=16, verbose=True)

        with self.loaded_lock:
            self.num_loaded += 1

        return ds

    def __call__(self):
        return self.__iter__()

def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    while not tokill():
        ds = dataset_generator.nextone()
        if not ds:
            break 
        batches_queue.put(ds, block=True)
    #print("Cleaning up threaded_batches_feeder worker", threading.get_ident())
    return

def event_loop(train_batches_queue, use_cuda):
    ds = train_batches_queue.get(block=True)
    #print("event_loop nev={0}, queued={1}".format(len(ds), train_batches_queue.qsize()))

    if use_cuda:
        ds.numpy_lib = cupy
        ds.move_to_device(cupy)

    ret = ds.analyze(analyze_data, verbose=True, is_mc=is_mc, lumimask=lumimask, lumidata=lumidata, pu_corrections_target=pu_corrections_2016, debug=False)
    ret["num_events"] = len(ds)

    train_batches_queue.task_done()
    #print("analyzed {0}".format(len(ds)))
    return ret, len(ds), ds.memsize()/1024.0/1024.0

def parse_args():
    parser = argparse.ArgumentParser(description='Example HiggsMuMu analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--action', '-a', action='append', help='List of actions to do', choices=['cache', 'analyze'])
    parser.add_argument('--nthreads', '-t', action='store', help='Number of CPU threads or workers to use', type=int, default=4, required=False)
    parser.add_argument('--datapath', '-p', action='store', help='Prefix to load NanoAOD data from', default="/nvmedata")
    parser.add_argument('--maxfiles', '-m', action='store', help='Maximum number of files to process', default=-1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    nev_total = 0
    t0 = time.time()
    
    args = parse_args()  
    print(args)

    processed_size_mb = 0
    num_processed = 0

    if args.use_cuda:
        print("Using the GPU CUDA backend")
        import cupy
        dev = cupy.cuda.Device(0)
        NUMPY_LIB = cupy
        from hepaccelerate.backend_cuda import *
        NUMPY_LIB.searchsorted = searchsorted
    else:
        print("Using the numpy CPU backend")
        NUMPY_LIB = numpy
        from hepaccelerate.backend_cpu import *
        NUMPY_LIB.asnumpy = numpy.array

    NanoAODDataset.numpy_lib = NUMPY_LIB
    LumiMask.numpy_lib = NUMPY_LIB

    lumimask = LumiMask("data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt")
    lumidata = LumiData("data/lumi2017.csv")
    pu_corrections_2016 = load_puhist_target("data/RunII_2017_data.root") 

    for datasetname, globpattern, is_mc in [
        ("data_2017", args.datapath + "/store/data/Run2017*/SingleMuon/NANOAOD/Nano14Dec2018-v1/**/*.root", False),
#        ("ggh", args.datapath + "/store/mc/RunIIFall17NanoAODv4/GluGluHToMuMu_M125_*/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
        ("dy", args.datapath + "/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8*/**/*.root", True),
#        ("ttjets_dl", args.datapath + "/store/mc/RunIIFall17NanoAODv4/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8/**/*.root", True),
#        ("ww", args.datapath + "/store/mc/RunIIFall17NanoAODv4/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
#        ("wz", args.datapath + "/store/mc/RunIIFall17NanoAODv4/WZTo3LNu_13TeV-powheg-pythia8/**/*.root", True),
#        ("zz", args.datapath + "/store/mc/RunIIFall17NanoAODv4/ZZTo2L2Nu_13TeV_powheg_pythia8/**/*.root", True),
        ]:
        filenames_all = glob.glob(globpattern, recursive=True)
        filenames_all = [fn for fn in filenames_all if not "Friend" in fn]
        print("dataset {0} has {1} files".format(datasetname, len(filenames_all)))
        if args.maxfiles > 0:
            filenames_all = filenames_all[:args.maxfiles]

        print("processing {0} files".format(len(filenames_all)))

        if "cache" in args.action:
            print("Preparing caches ROOT files")
            nev_total, processed_size_mb = cache_data(filenames_all, is_mc, nworkers=args.nthreads)

        if "analyze" in args.action:        
            training_set_generator = InputGen(list(filenames_all), is_mc, args.nthreads)
            threadk = thread_killer()
            threadk.set_tokill(False)
            train_batches_queue = Queue(maxsize=10)
            cuda_batches_queue = Queue(maxsize=1)
 
            for _ in range(1):
                t = Thread(target=threaded_batches_feeder, args=(threadk, train_batches_queue, training_set_generator))
                t.start()

            rets = []

            #loop over all data
            while num_processed < len(training_set_generator.paths_chunks):
                ret, nev, memsize = event_loop(train_batches_queue, args.use_cuda) 
                rets += [ret]
                processed_size_mb += memsize
                nev_total += nev
                num_processed += 1

                if args.use_cuda:
                    mempool = cupy.get_default_memory_pool()
                    pinned_mempool = cupy.get_default_pinned_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    dev.synchronize()

            #clean up threads
            threadk.set_tokill(True)

            #
            ret = sum(rets, Results({}))
            if is_mc:
                ret["gen_sumweights"] = get_gen_sumweights(filenames_all)
            ret.save_json("out/{0}.json".format(datasetname))

    t1 = time.time()
    dt = t1 - t0
    print("Overall processed {nev:.2E} ({nev}) events in total {size:.2f} GB, {dt:.1f} seconds, {evspeed:.2E} Hz, {sizespeed:.2f} MB/s".format(
        nev=nev_total, dt=dt, size=processed_size_mb/1024.0, evspeed=nev_total/dt, sizespeed=processed_size_mb/dt)
    )
