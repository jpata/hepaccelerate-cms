import os
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import numba

import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
import uproot

def test_load_dataset():
    NUMPY_LIB, ha = choose_backend(use_cuda=False)
    
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
    dataset = test_load_dataset()

    dataset.preload()
    assert(len(dataset.data_host) == 1)
    
    dataset.make_objects()
    assert(len(dataset.structs["Jet"]) == 1)
    assert(len(dataset.eventvars) == 1)

    dataset.to_cache()
    return dataset

def test_dataset_from_cache():
    dataset = test_load_dataset()
    dataset.from_cache()
    
    dataset2 = test_load_dataset()
    dataset2.preload()
    dataset2.make_objects()

    assert(dataset.num_objects_loaded("Jet") == dataset2.num_objects_loaded("Jet"))
    assert(dataset.num_events_loaded("Jet") == dataset2.num_events_loaded("Jet"))

if __name__ == "__main__":
    test_load_dataset()
    test_dataset_to_cache()
    test_dataset_from_cache()
