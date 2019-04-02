# hepaccelerate

Accelerated array analysis on flat ROOT data. Process 1 billion events to histograms in 10 minutes on a single GPU workstation. Weighted histograms, jet-lepton deltaR matching and pileup weighting directly on the GPU! Or, if you don't have a GPU, no big deal, the same code works on the CPU too, just somewhat slower! No batch jobs or servers required!

Requirements:
 - python 3
 - uproot
 - awkward-array
 - numba

Optional for CUDA acceleration:
 - cupy
 - cudatoolkit

We also provide a singularity image with the python libraries preinstalled, see below.

## Sneak peek

~~~
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
    ["8AAF0CFA-542F-8947-973E-A61A78293481.root"],
    ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "HLT_IsoMu24"], "Events", ["Jet", "Muon"], ["HLT_IsoMu24"])
dataset.preload(nthreads=4, verbose=True)
dataset.make_objects()
results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
results.save_json("out.json")
~~~

## Getting started

~~~
git clone git@github.com:jpata/hepaccelerate.git
cd hepaccelerate

#prepare a list of files (currently must be on the local filesystem, not on xrootd) to read
#replace /nvmedata with your local location of ROOT files
find /nvmedata/store/mc/RunIIFall17NanoAODv4/GluGluHToMuMu_M125_*/NANOAODSIM -name "*.root | head -n100 > filelist.txt

#Run the test analysis
PYTHONPATH=.:$PYTHONPATH python3 tests/simple.py --filelist filelist.txt

#output will be stored in this json
cat out.json
~~~

This script loads the ROOT files, prepares local caches from the branches you read and processes the data
~~~
#second time around, you can load the data from the cache, which is much faster
PYTHONPATH=.:$PYTHONPATH python3 tests/simple.py --filelist filelist.txt --from-cache

#use CUDA for array processing on a GPU!
PYTHONPATH=.:$PYTHONPATH python3 tests/simple.py --filelist filelist.txt --from-cache --use-cuda
~~~

## Singularity image
Singularity allows you to try out the package without needing to install the python libraries.

~~~
#Download the uproot+cupy+numba singularity image from CERN
wget https://jpata.web.cern.ch/jpata/singularity/cupy.simg -o singularity/cupy.simg

##or on lxplus
#cp /eos/user/j/jpata/www/singularity/cupy.simg ./singularity/
##or compile yourself if you have ROOT access on your machine
#cd singularity;make

LC_ALL=C PYTHONPATH=.:$PYTHONPATH singularity exec -B /nvmedata --nv singularity/cupy.simg python3 ...
~~~