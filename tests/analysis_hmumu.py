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
import numpy as np

import uproot
import hepaccelerate
import hepaccelerate.utils
from hepaccelerate.utils import Results
from hepaccelerate.utils import NanoAODDataset
from hepaccelerate.utils import Histogram
from hepaccelerate.utils import choose_backend, LumiData, LumiMask
import hepaccelerate.backend_cpu as backend_cpu

import fnal_column_analysis_tools

genweight_scalefactor = 1e6
chunksize = 1

from types import FrameType

ha = None
NUMPY_LIB = None

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
    return Histogram(*ha.histogram_from_vector(data, weights, bins))

def get_selected_muons(muons, trigobj, mask_events, mu_pt_cut_leading, mu_pt_cut_subleading, mu_aeta_cut, mu_iso_cut, muon_id_type, muon_trig_match_dr):
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
    
    if muon_id_type == "medium":
        passes_id = muons.mediumId == 1
    elif muon_id_type == "tight":
        passes_id = muons.tightId == 1
    else:
        raise Exception("unknown muon id: {0}".format(muon_id_type))

    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(muons.eta) < mu_aeta_cut
    muons_passing_id =  passes_iso & passes_id & passes_subleading_pt & passes_aeta

    #Get muons that are high-pt and are matched to trigger

    mask_trigger_objects_mu = (trigobj.id == 13)
    muons_matched_to_trigobj = NUMPY_LIB.invert(ha.mask_deltar_first(
        muons, muons_passing_id & passes_leading_pt, trigobj, mask_trigger_objects_mu, muon_trig_match_dr
    ))

    #At least one muon must be matched to trigger
    events_passes_triggermatch = ha.sum_in_offsets(muons, muons_matched_to_trigobj, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 1
    
    #select events that have muons passing cuts 
    events_passes_muid = ha.sum_in_offsets(muons, muons_passing_id, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2
    events_passes_leading_pt = ha.sum_in_offsets(muons, muons_passing_id & passes_leading_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 1
    events_passes_subleading_pt = ha.sum_in_offsets(muons, muons_passing_id & passes_subleading_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2

    base_event_sel = mask_events & events_passes_triggermatch & events_passes_muid & events_passes_leading_pt & events_passes_subleading_pt
    
    muons_passing_os = ha.select_muons_opposite_sign(muons, muons_passing_id & passes_subleading_pt)
    events_passes_os = ha.sum_in_offsets(muons, muons_passing_os, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 2
    
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    
    return {
         "selected_events": final_event_sel,
         "selected_muons": final_muon_sel,
    }


def get_bit_values(array, bit_index):
    """
    Given an array of N binary values (e.g. jet IDs), return the bit value at bit_index in [0, N-1].
    """
    return (array & 2**(bit_index - 1)) >> 1

def get_selected_jets(jets, muons, mask_muons, mask_events, jet_pt_cut, jet_eta_cut, jet_dr_cut, jet_id, jet_puid):
    """
    Given jets and selected muons in events, choose jets that pass quality criteria and that are not dR-matched
    to muons.
 
    """

    #Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto
    if jet_id == "tight":
        pass_jetid = get_bit_values(jets.jetId, 1)
    elif jet_id == "loose":
        pass_jetid = get_bit_values(jets.jetId, 0)

    #The value is a bit representation of the fulfilled working points: tight (1), medium (2), and loose (4).
    #As tight is also medium and medium is also loose, there are only 4 different settings: 0 (no WP, 0b000), 4 (loose, 0b100), 6 (medium, 0b110), and 7 (tight, 0b111).
    if jet_puid == "loose":
        pass_jet_puid = jets.puId == 4
    elif jet_puid == "medium":
        pass_jet_puid = jets.puId == 6
    elif jet_puid == "tight":
        pass_jet_puid = jets.puId == 7

    selected_jets = (jets.pt > jet_pt_cut) & (NUMPY_LIB.abs(jets.eta) < jet_eta_cut) & pass_jetid & pass_jet_puid
    jets_pass_dr = ha.mask_deltar_first(jets, selected_jets, muons, mask_muons, jet_dr_cut)
    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = selected_jets & jets_pass_dr

    num_jets = ha.sum_in_offsets(jets, selected_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)

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

    px_total = ha.sum_in_offsets(objects, px, mask_events, mask_objects)
    py_total = ha.sum_in_offsets(objects, py, mask_events, mask_objects)
    pz_total = ha.sum_in_offsets(objects, pz, mask_events, mask_objects)
    e_total = ha.sum_in_offsets(objects, e, mask_events, mask_objects)
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
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio), pu_weights)
    fix_large_weights(pu_weights) 
     
    ratio_up = values_up / src_pu_hist.contents
    remove_inf_nan(ratio_up)
    pu_weights_up = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio_up), pu_weights_up)
    fix_large_weights(pu_weights_up) 
    
    ratio_down = values_down / src_pu_hist.contents
    remove_inf_nan(ratio_down)
    pu_weights_down = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio_down), pu_weights_down)
    fix_large_weights(pu_weights_down) 
    
    return pu_weights, pu_weights_up, pu_weights_down

def select_events_trigger(scalars, mask_events, parameters):

    flags = [
        "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter",
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_BadPFMuonFilter",
        "Flag_BadChargedCandidateFilter"
    ]
    for flag in flags:
        mask_events = mask_events & scalars[flag]
    
    pvsel = scalars["PV_npvsGood"] > parameters["nPV"]
    pvsel = pvsel & (scalars["PV_ndof"] > parameters["NdfPV"])
    pvsel = pvsel & (scalars["PV_z"] < parameters["zPV"])

    mask_events = mask_events & scalars["HLT_IsoMu24"] & pvsel

def get_int_lumi(runs, lumis, mask_events, lumidata):
    print("computing integrated luminosity from {0} lumis".format(len(lumis)))
    processed_runs = NUMPY_LIB.asnumpy(runs[mask_events])
    processed_lumis = NUMPY_LIB.asnumpy(lumis[mask_events])
    runs_lumis = np.zeros((processed_runs.shape[0], 2), dtype=np.uint32)
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
    pu_corrections=None,
    lumimask=None,
    lumidata=None,
    parameters={},
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

    # scalars["run"] = NUMPY_LIB.array(scalars["run"], dtype=NUMPY_LIB.uint32)
    # scalars["luminosityBlock"] = NUMPY_LIB.array(scalars["run"], dtype=NUMPY_LIB.uint32)

    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    select_events_trigger(scalars, mask_events, parameters)
    if debug:
        print("{0} events passed trigger".format(NUMPY_LIB.sum(mask_events)))

    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"]/genweight_scalefactor
        pu_weights, pu_weights_up, pu_weights_down = compute_pu_weights(pu_corrections, weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        weights["puWeight"] = weights["nominal"] * pu_weights
        weights["puWeight_up"] = weights["nominal"] * pu_weights_up
        weights["puWeight_down"] = weights["nominal"] * pu_weights_down
    
    
    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(
        muons, trigobj, mask_events,
        parameters["muon_pt_leading"], parameters["muon_pt"],
        parameters["muon_eta"], parameters["muon_iso"],
        parameters["muon_id"], parameters["muon_trigger_match_dr"]
    )
    
    if doverify:
        z = ha.sum_in_offsets(muons, ret_mu["selected_muons"], ret_mu["selected_events"], ret_mu["selected_muons"], dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))
    if debug:
        print("{0} events passed muon".format(NUMPY_LIB.sum(ret_mu["selected_events"])))
    
    #get the passing jets for events that pass muon selection
    ret_jet = get_selected_jets(jets, muons,
        ret_mu["selected_muons"], mask_events, parameters["jet_pt"],
        parameters["jet_eta"], parameters["jet_mu_dr"], parameters["jet_id"], parameters["jet_puid"]
    )    
    if doverify:
        z = ha.min_in_offsets(jets, jets.pt, ret_mu["selected_events"], ret_jet["selected_jets"])
        assert(NUMPY_LIB.all(z[z>0] > jet_pt_cut))

        
    inv_mass = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"])
    if not is_mc:
        inv_mass[(inv_mass > 120) & (inv_mass < 130)] = 0
 
    inds = NUMPY_LIB.zeros(muons.numevents(), dtype=NUMPY_LIB.int32)
    leading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_muon_eta = ha.get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    leading_jet_eta = ha.get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    
    inds[:] = 1
    subleading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_muon_eta = ha.get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    subleading_jet_eta = ha.get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    
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
        runs = NUMPY_LIB.asnumpy(scalars["run"])
        lumis = NUMPY_LIB.asnumpy(scalars["luminosityBlock"])
        mask_lumi = NUMPY_LIB.array(lumimask(runs, lumis))
        mask_events = mask_events & mask_lumi
        #get integrated luminosity in this file
        if not (lumidata is None): 
            int_lumi = get_int_lumi(runs, lumis, mask_events, lumidata)
    
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
        "PV_npvsGood", "PV_ndof", "PV_z",
        "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_goodVertices",
        "Flag_globalSuperTightHalo2016Filter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter",
        "HLT_IsoMu24",
        "run", "luminosityBlock", "event"
    ]
    if is_mc:
        arrays_ev += ["Pileup_nTrueInt", "Generator_weight", "genWeight"]
    arrays_jet = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId", "Jet_puId",
    ]
    
    arrays_muon = [
        "nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso04_all", "Muon_mediumId", "Muon_tightId", "Muon_charge"
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

        ds = create_dataset(self.paths_chunks[self.num_chunk], self.is_mc)
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

def event_loop(train_batches_queue, use_cuda, **kwargs):
    ds = train_batches_queue.get(block=True)
    #print("event_loop nev={0}, queued={1}".format(len(ds), train_batches_queue.qsize()))

    if use_cuda:
        #copy dataset to GPU and make sure future operations are done on it
        import cupy
        ds.numpy_lib = cupy
        ds.move_to_device(cupy)

    ret = ds.analyze(analyze_data, **kwargs)
    ret["num_events"] = len(ds)

    train_batches_queue.task_done()

    #clean up CUDA memory
    if use_cuda:
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    return ret, len(ds), ds.memsize()/1024.0/1024.0

def parse_args():
    parser = argparse.ArgumentParser(description='Example HiggsMuMu analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--action', '-a', action='append', help='List of actions to do', choices=['cache', 'analyze'], required=True)
    parser.add_argument('--nthreads', '-t', action='store', help='Number of CPU threads or workers to use', type=int, default=4, required=False)
    parser.add_argument('--datapath', '-p', action='store', help='Prefix to load NanoAOD data from', default="/nvmedata")
    parser.add_argument('--maxfiles', '-m', action='store', help='Maximum number of files to process', default=-1, type=int)
    args = parser.parse_args()
    return args

def main(parameters):
    global NUMPY_LIB, ha

    nev_total = 0
    t0 = time.time()
    
    args = parse_args()  
    print(args)

    NUMPY_LIB, ha = choose_backend(args.use_cuda)
    NanoAODDataset.numpy_lib = NUMPY_LIB
    
    if args.use_cuda:
        import cupy
    else:
        os.environ["NUMBA_NUM_THREADS"] = str(args.nthreads)

    lumimask = LumiMask("data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt", np, backend_cpu)
    lumidata = LumiData("data/lumi2017.csv")
    pu_corrections_2017 = load_puhist_target("data/RunII_2017_data.root")

    processed_size_mb = 0
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

        print("processing {0} files, {1}".format(len(filenames_all), args.action))

        if "cache" in args.action:
            print("Preparing caches from ROOT files")
            _nev_total, _processed_size_mb = cache_data(filenames_all, is_mc, nworkers=args.nthreads)
            nev_total += _nev_total
            processed_size_mb += _processed_size_mb

        if "analyze" in args.action:        
            print("Starting analysis")
            training_set_generator = InputGen(list(filenames_all), is_mc, args.nthreads)
            threadk = thread_killer()
            threadk.set_tokill(False)
            train_batches_queue = Queue(maxsize=10)
            cuda_batches_queue = Queue(maxsize=1)
 
            for _ in range(1):
                t = Thread(target=threaded_batches_feeder, args=(threadk, train_batches_queue, training_set_generator))
                t.start()

            rets = []

            num_processed = 0
            #loop over all data, call the analyze function
            while num_processed < len(training_set_generator.paths_chunks):
                ret, nev, memsize = event_loop(train_batches_queue, args.use_cuda, debug=False, verbose=True, is_mc=is_mc, lumimask=lumimask, lumidata=lumidata, pu_corrections=pu_corrections_2017, parameters=parameters) 
                rets += [ret]
                processed_size_mb += memsize
                nev_total += nev
                num_processed += 1


            #clean up threads
            threadk.set_tokill(True)

            #save output
            ret = sum(rets, Results({}))
            if is_mc:
                ret["gen_sumweights"] = get_gen_sumweights(filenames_all)
            ret.save_json("out/{0}.json".format(datasetname))

    t1 = time.time()
    dt = t1 - t0
    print("Overall processed {nev:.2E} ({nev}) events in total {size:.2f} GB, {dt:.1f} seconds, {evspeed:.2E} Hz, {sizespeed:.2f} MB/s".format(
        nev=nev_total, dt=dt, size=processed_size_mb/1024.0, evspeed=nev_total/dt, sizespeed=processed_size_mb/dt)
    )

if __name__ == "__main__":
    # import yappi
    # filename = 'callgrind.filename.prof'
    # yappi.set_clock_type('cpu')
    # yappi.start(builtins=True)

    parameters = {
        "NdfPV": 4,
        "zPV": 24,
        "nPV": 0,
        "muon_pt": 20,
        "muon_pt_leading": 30,
        "muon_eta": 2.4,
        "muon_iso": 0.25,
        "muon_id": "medium",
        "muon_trigger_match_dr": 0.1,
        "jet_mu_dr": 0.4,
        "jet_pt": 30,
        "jet_eta": 4.7,
        "jet_id": "tight",
        "jet_puid": "loose",
    }

    main(parameters)

    # stats = yappi.get_func_stats()
    # stats.save(filename, type='callgrind')
