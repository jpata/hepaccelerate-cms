#usr/bin/env python3
#Run as PYTHONPATH=. python3 tests/example.py
import hepaccelerate
from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

#choose whether or not to use the GPU backend
NUMPY_LIB, ha = choose_backend(use_cuda=False)

#define our analysis function
def analyze_data_function(data, parameters):
    ret = Results()

    num_events = data["num_events"]
    muons = data["Muon"]
    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    mask_muons_passing_pt = muons.pt > parameters["muons_ptcut"]
    num_muons_event = ha.sum_in_offsets(muons, mask_muons_passing_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8)
    mask_events_dimuon = num_muons_event == 2

    #get the leading muon pt in events that have exactly two muons
    inds = NUMPY_LIB.zeros(num_events, dtype=NUMPY_LIB.int32)
    leading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, mask_events_dimuon, mask_muons_passing_pt)

    #compute a weighted histogram
    weights = NUMPY_LIB.ones(num_events, dtype=NUMPY_LIB.float32)
    bins = NUMPY_LIB.linspace(0,300,101)
    hist_muons_pt = Histogram(*ha.histogram_from_vector(leading_muon_pt[mask_events_dimuon], weights[mask_events_dimuon], bins))

    #save it to the output
    ret["hist_leading_muon_pt"] = hist_muons_pt
    return ret

#Load this input file
filename = "/Volumes/Samsung_T3/nanoad//store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano14Dec2018_new_pmx_102X_mc2017_realistic_v6_ext1-v1/00000/8AAF0CFA-542F-8947-973E-A61A78293481.root"
#Optionally set try_cache=True to make the the analysis faster the next time around when reading the same branches
try_cache = True

#Predefine which branches to read from the TTree and how they are grouped to objects
#This will be verified against the actual ROOT TTree when it is loaded
datastructures = {
        "Muon": [
            ("Muon_pt", "float32"),("Muon_eta", "float32"),
            ("Muon_phi", "float32"), ("Muon_mass", "float32"),
            ("Muon_pfRelIso04_all", "float32"), ("Muon_mediumId", "bool"),
            ("Muon_tightId", "bool"), ("Muon_charge", "int32")
        ],
        "Jet": [
            ("Jet_pt", "float32"), ("Jet_eta", "float32"),
            ("Jet_phi", "float32"), ("Jet_btagDeepB", "float32"),
            ("Jet_jetId", "int32"), ("Jet_puId", "int32"),
        ],
        "EventVariables": [
            ("HLT_IsoMu24", "bool"),
            ("run", "uint32"),
            ("luminosityBlock", "uint32"),
            ("event", "uint64")
        ]
    }

dataset = NanoAODDataset([filename], datastructures, cache_location="./mycache/")

if try_cache:
    print("Trying to load branch data from cache...")
    try:
        dataset.from_cache(verbose=True, nthreads=4)
        print("Loaded data from cache, did not touch original ROOT files.")
    except FileNotFoundError as e:
        print("Cache not found, creating...")
        dataset.preload(nthreads=4, verbose=True)
        dataset.make_objects()
        dataset.to_cache(verbose=True, nthreads=4)
else:
    print("Loading data directly from ROOT file...")
    dataset.preload(nthreads=4, verbose=True)
    dataset.make_objects()

results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
results.save_json("out.json")
