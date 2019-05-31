import time
import os
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import numba
from importlib import reload

import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
import uproot

import numpy as np
import json

use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
NUMPY_LIB, ha = choose_backend(use_cuda=use_cuda)

def load_dataset():
    
    datastructures = {
        "Muon": [
            ("Muon_pt", "float32"), ("Muon_eta", "float32"),
            ("Muon_phi", "float32"), ("Muon_mass", "float32"),
            ("Muon_pdgId", "int32"),
            ("Muon_pfRelIso04_all", "float32"), ("Muon_mediumId", "bool"),
            ("Muon_tightId", "bool"), ("Muon_charge", "int32"),
            ("Muon_isGlobal", "bool"), ("Muon_isTracker", "bool"),
            ("Muon_nTrackerLayers", "int32"),
        ],
        "Electron": [
            ("Electron_pt", "float32"), ("Electron_eta", "float32"),
            ("Electron_phi", "float32"), ("Electron_mass", "float32"),
            ("Electron_mvaFall17V1Iso_WP90", "bool"),
        ],
        "Jet": [
            ("Jet_pt", "float32"), ("Jet_eta", "float32"),
            ("Jet_phi", "float32"), ("Jet_mass", "float32"),
            ("Jet_btagDeepB", "float32"),
            ("Jet_jetId", "int32"), ("Jet_puId", "int32"),
        ],
        "TrigObj": [
            ("TrigObj_pt", "float32"),
            ("TrigObj_eta", "float32"),
            ("TrigObj_phi", "float32"),
            ("TrigObj_id", "int32")
        ],
        "EventVariables": [
            ("PV_npvsGood", "float32"), 
            ("PV_ndof", "float32"),
            ("PV_z", "float32"),
            ("Flag_BadChargedCandidateFilter", "bool"),
            ("Flag_HBHENoiseFilter", "bool"),
            ("Flag_HBHENoiseIsoFilter", "bool"),
            ("Flag_EcalDeadCellTriggerPrimitiveFilter", "bool"),
            ("Flag_goodVertices", "bool"),
            ("Flag_globalSuperTightHalo2016Filter", "bool"),
            ("Flag_BadPFMuonFilter", "bool"),
            ("Flag_BadChargedCandidateFilter", "bool"),
            ("HLT_IsoMu27", "bool"),
            ("run", "uint32"),
            ("luminosityBlock", "uint32"),
            ("event", "uint64")
        ],
    }
    datastructures["EventVariables"] += [
        ("Pileup_nTrueInt", "uint32"),
        ("Generator_weight", "float32"),
        ("genWeight", "float32")
    ]
    datastructures["Muon"] += [
        ("Muon_genPartIdx", "int32"),
    ]
    datastructures["GenPart"] = [
        ("GenPart_pt", "float32"),
    ]
    dataset = Dataset(["/nvme1/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano14Dec2018_new_pmx_102X_mc2017_realistic_v6_ext1-v1/00000/D91A0C3A-B4A9-D748-ACB5-CE4DC654037E.root"], datastructures, cache_location="./mycache/", treename="Events", datapath="/nvme1")
    
    try:
        dataset.from_cache(verbose=False)
    except FileNotFoundError as e:
        dataset.load_root()
        dataset.to_cache()

    return dataset

def time_kernel(dataset, test_kernel):
    test_kernel(dataset)

    t0 = time.time()
    for i in range(5):
        n = test_kernel(dataset)
    t1 = time.time()

    dt = (t1 - t0) / 5.0
    speed = float(n)/dt
    return speed

def test_kernel_sum_in_offsets(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    z = ha.sum_in_offsets(
        muons,
        muons.pt,
        sel_ev,
        sel_mu, dtype=NUMPY_LIB.float32)
    return muons.numevents()

def test_kernel_max_in_offsets(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    z = ha.max_in_offsets(
        muons,
        muons.pt,
        sel_ev,
        sel_mu)
    return muons.numevents()

def test_kernel_simple_cut(dataset):
    muons = dataset.structs["Muon"][0]
    sel_mu = muons.pt > 30.0
    return muons.numevents()

def test_kernel_mask_deltar_first(dataset):
    muons = dataset.structs["Muon"][0]
    trigobj = dataset.structs["TrigObj"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    sel_trigobj = (trigobj.id == 13)
    muons_matched_to_trigobj = ha.mask_deltar_first(
        muons, sel_mu, trigobj,
        sel_trigobj, 0.5
    )
    return muons.numevents()
    
def test_kernel_select_muons_opposite_sign(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    muons_passing_os = ha.select_muons_opposite_sign(
            muons, sel_mu)
    return muons.numevents()

def test_kernel_histogram_from_vector(dataset):
    muons = dataset.structs["Muon"][0]
    weights = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.float32)
    ha.histogram_from_vector(muons.pt, weights, NUMPY_LIB.linspace(0,200,100))
    return muons.numevents()

def run_all_tests():
    global NUMPY_LIB, ha

    ds = load_dataset()

    t0 = time.time()
    for i in range(5):
        ds.move_to_device(NUMPY_LIB)
    t1 = time.time()
    dt = (t1 - t0)/5.0

    ret = {}
    print("Memory transfer speed: {0:.2f} MHz, event size {1:.2f} bytes, data transfer speed {2:.2f} MB/s".format(ds.numevents() / dt / 1000.0 / 1000.0, ds.eventsize(), ds.memsize()/dt/1000/1000))
    ret["memory_transfer"] = ds.numevents() / dt / 1000.0 / 1000.0

    t = time_kernel(ds, test_kernel_sum_in_offsets)
    print("sum_in_offsets {0:.2f} MHz".format(t/1000/1000))
    ret["sum_in_offsets"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_simple_cut)
    print("simple_cut {0:.2f} MHz".format(t/1000/1000))
    ret["simple_cut"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_max_in_offsets)
    print("max_in_offsets {0:.2f} MHz".format(t/1000/1000))
    ret["max_in_offsets"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_mask_deltar_first)
    print("mask_deltar_first {0:.2f} MHz".format(t/1000/1000))
    ret["mask_deltar_first"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_select_muons_opposite_sign)
    print("select_muons_opposite_sign {0:.2f} MHz".format(t/1000/1000))
    ret["select_muons_opposite_sign"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_histogram_from_vector)
    print("histogram_from_vector {0:.2f} MHz".format(t/1000/1000))
    ret["histogram_from_vector"] = t/1000/1000

    ret["use_cuda"] = use_cuda
    ret["num_threads"] = numba.config.NUMBA_NUM_THREADS
    return ret
   
if __name__ == "__main__":
  
    with open("kernel_benchmarks.txt", "a") as of:
        for i in range(5):  
            of.write(json.dumps(run_all_tests()) + '\n')
