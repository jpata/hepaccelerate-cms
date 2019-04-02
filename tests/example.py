import hepaccelerate
from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

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

    weights = NUMPY_LIB.ones(num_events, dtype=NUMPY_LIB.float32)
    bins = NUMPY_LIB.linspace(0,300,101)
    hist_muons_pt = Histogram(*ha.histogram_from_vector(leading_muon_pt[mask_events_dimuon], weights, bins))

    ret["hist_leading_muon_pt"] = hist_muons_pt
    return ret

dataset = NanoAODDataset(
    ["/Volumes/Samsung_T3/nanoad//store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano14Dec2018_new_pmx_102X_mc2017_realistic_v6_ext1-v1/00000/8AAF0CFA-542F-8947-973E-A61A78293481.root"],
    ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "HLT_IsoMu24"], "Events", ["Jet", "Muon"], ["HLT_IsoMu24"])
dataset.preload(nthreads=4, verbose=True)
dataset.make_objects()
results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
results.save_json("out.json")
