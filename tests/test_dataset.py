import os
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import numba

import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
import uproot

import numpy as np

NUMPY_LIB, ha = choose_backend(use_cuda=False)

def test_load_dataset():
    print("test_load_dataset")
    
    fi = uproot.open("data/HZZ.root")
    #print(fi.keys())
    #print(fi.get("events").keys())
    
    datastructures = {
            "Muon": [
                ("Muon_Px", "float32"),
                ("Muon_Py", "float32"),
                ("Muon_Pz", "float32"), 
                ("Muon_E", "float32"),
                ("Muon_Charge", "int32"),
                ("Muon_Iso", "float32")
            ],
            "Jet": [
                ("Jet_Px", "float32"),
                ("Jet_Py", "float32"),
                ("Jet_Pz", "float32"),
                ("Jet_E", "float32"),
                ("Jet_btag", "float32"),
                ("Jet_ID", "bool")
            ],
            "EventVariables": [
                ("NPrimaryVertices", "int32"),
                ("triggerIsoMu24", "bool"),
                ("EventWeight", "float32")
            ]
        }
    dataset = Dataset(["data/HZZ.root"], datastructures, cache_location="./mycache/", treename="events")
    assert(dataset.filenames[0] == "data/HZZ.root")
    assert(len(dataset.structs["Jet"]) == 0)
    assert(len(dataset.eventvars) == 0)

    return dataset

def test_dataset_to_cache():
    print("test_dataset_to_cache")
    dataset = test_load_dataset()

    dataset.load_root()
    assert(len(dataset.data_host) == 1)
    
    assert(len(dataset.structs["Jet"]) == 1)
    assert(len(dataset.eventvars) == 1)

    dataset.to_cache()
    return dataset

def test_dataset_from_cache():
    print("test_dataset_from_cache")
    dataset = test_load_dataset()
    dataset.from_cache()
    
    dataset2 = test_load_dataset()
    dataset2.load_root()

    assert(dataset.num_objects_loaded("Jet") == dataset2.num_objects_loaded("Jet"))
    assert(dataset.num_events_loaded("Jet") == dataset2.num_events_loaded("Jet"))

def map_func(dataset, ifile):
    mu = dataset.structs["Muon"][ifile]
    mu_pt = np.sqrt(mu.Px**2 + mu.Py**2)
    mu_pt_pass = mu_pt > 20
    mask_rows = np.ones(mu.numevents(), dtype=np.bool)
    mask_content = np.ones(mu.numobjects(), dtype=np.bool)
    ret = ha.sum_in_offsets(mu, mu_pt_pass, mask_rows, mask_content, dtype=np.int8) 
    return ret

def test_dataset_map():
    dataset = test_load_dataset()
    dataset.load_root()

    rets = dataset.map(map_func)
    assert(len(rets) == 1)
    assert(len(rets[0]) == dataset.structs["Muon"][0].numevents())
    assert(np.sum(rets[0]) > 0)
    return rets

def test_dataset_compact():
    dataset = test_load_dataset()
    dataset.load_root()

    memsize1 = dataset.memsize()
    rets = dataset.map(map_func)
    dataset.compact(rets)
    memsize2 = dataset.memsize()
    assert(memsize1 > memsize2)
    print(memsize2/memsize1)

if __name__ == "__main__":
    test_load_dataset()
    test_dataset_to_cache()
    test_dataset_from_cache()
    test_dataset_map()
    test_dataset_compact()
