[![Build Status](https://travis-ci.com/jpata/hepaccelerate.svg?branch=master)](https://travis-ci.com/jpata/hepaccelerate)
[![pipeline status](https://gitlab.cern.ch/jpata/hepaccelerate/badges/libhmm/pipeline.svg)](https://gitlab.cern.ch/jpata/hepaccelerate/commits/libhmm)

# hepaccelerate

Accelerated array analysis on flat ROOT data. Process 1 billion events to histograms in 4 minutes on a single workstation.
Weighted histograms, jet-lepton deltaR matching and pileup weighting! Works on both the CPU and GPU!

Required python libraries:
 - python 3
 - uproot
 - awkward-array
 - numba >0.43

Optional libraries for CUDA acceleration:
 - cupy
 - cudatoolkit

## Sneak peek

This is a minimal example in [tests/example.py](tests/example.py).

```python
#usr/bin/env python3
#Run as PYTHONPATH=. python3 tests/example.py
import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend

#choose whether or not to use the GPU backend
NUMPY_LIB, ha = choose_backend(use_cuda=False)

#define our analysis function
def analyze_data_function(data, parameters):
    ret = Results()

    num_events = data["num_events"]
    muons = data["Muon"]
    mu_pt = NUMPY_LIB.sqrt(muons.Px**2 + muons.Py**2)
    muons.attrs_data["pt"] = mu_pt

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
filename = "data/HZZ.root"

#Predefine which branches to read from the TTree and how they are grouped to objects
#This will be verified against the actual ROOT TTree when it is loaded
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

dataset = Dataset([filename], datastructures, cache_location="./mycache/", treename="events", datapath="")

#load data to memory
try:
    dataset.from_cache(verbose=True)
    print("Loaded data from cache, did not touch original ROOT files.")
except FileNotFoundError as e:
    print("Cache not found, creating...")
    dataset.load_root()
    dataset.to_cache()

#process data
results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
results.save_json("out.json")
```

## Benchmark on a CMS Higgs analysis

The following benchmarks have been carried out with a realistic CMS analysis on NanoAOD. The analysis can be found in [tests/analysis_hmumu.py](tests/analysis_hmumu.py).

This proto-analysis implements the following:

 - [x] muon selection: pT leading and subleading, eta, ID, isolation, opposite charge, matching to trigger objects
 - [x] jet selection: pt, eta, ID & PU ID, remove jets dR-matched to leptons
 - [x] event selection: MET filters, trigger, primary vertex quality cuts, two opposite sign muons
 - [x] high-level variables: dimuon invariant mass
 - [x] PU and gen weight computation
 - [x] on the fly luminosity calculation, golden JSON lumi filtering
 - [x] weighted histograms of muon, jet and event variables

From original CMS C++ code (CPU only, not included in benchmark), via Python CFFI:

 - [x] muon momentum Rochester corrections
 - [x] lepton scale factors
 - [ ] btag scale factors

Not yet implemented

 - [ ] in-situ training and evaluation of Higgs-to-bkg discriminators

The analysis is carried out using a single script on a single machine, `tests/hmm/analyse_hmumu.py -a cache -a analyze --nthreads N (--use-cuda)`. In order to be able to process a billion events in a few minutes, the first time the analysis is run, the branches that we will use are uncompressed and saved to local disk as numpy files. Without this optional caching step, the upper limit on the processing speed will be dominated by the CPU decompression of the ROOT TTree (first line `caching` in the tables). This means that the second time you run the analysis with reading the same branches, the decompression does not need to be done and we can achieve about 3-4x higher speeds.

### 2015 Macbook Pro, 35M events

We can process 35M events on a macbook in about 5 minutes.

- 4-core 2.6GHz i5
- 8GB DDR3
- 250GB SSD PT250B over USB3
- analyzed 37,021,788 events in 10.11 GB of branches

task    | configuration  |time     | speed       | speed
--------|----------------|---------|-------------|-----------
caching | CPU(4)         | 452.6 s | 8.18E+04 Hz | 22.87 MB/s
analyze | CPU(1)         | 122.9 s | 3.01E+05 Hz | 84.21 MB/s
analyze | CPU(4)         | 116.6 s | 3.22E+05 Hz | 89.90 MB/s


### High-end workstation, 1B events

We can demonstrate that we can process 1B events in about 4 minutes (from existing caches) and in about 20 minutes from raw NanoAOD. The theoretical maximum system throughput is limited by the 1.5GB/s read speed of the SSD.

- 16-core 3GHz i7
- 64GB DDR3
- 6.4TB Intel P4608
- 1x Titan X 12GB
- analyzed 1,093,479,508 events in 292.86 GB of branches


task    | configuration           | tot time | avg speed   | avg speed
--------|-------------------------|----------|-------------|-----------
caching | CPU(16)                 | 1611.1 s | 9.70E+05 Hz | 186.13 MB/s
analyze | CPU(1)                  | 2553.3 s | 4.28E+05 Hz | 117.45 MB/s
analyze | CPU(16)                 | 1689.7 s | 6.47E+05 Hz | 177.48 MB/s
analyze | GPU(1) CPU(16)          | 611.5 s  | 1.79E+06 Hz | 490.39 MB/s


## Getting started

```bash
git clone git@github.com:jpata/hepaccelerate.git
cd hepaccelerate

#prepare a list of files (currently must be on the local filesystem, not on xrootd) to read
#replace /nvmedata with your local location of ROOT files
find /nvmedata/store/mc/RunIIFall17NanoAODv4/GluGluHToMuMu_M125_*/NANOAODSIM -name "*.root | head -n100 > filelist.txt

#Run the test analysis
PYTHONPATH=.:$PYTHONPATH python3 tests/simple.py --filelist filelist.txt

#output will be stored in this json
cat out.json
```

This script loads the ROOT files, prepares local caches from the branches you read and processes the data
```bash
#second time around, you can load the data from the cache, which is much faster
PYTHONPATH=.:$PYTHONPATH python3 tests/simple.py --filelist filelist.txt --from-cache

#use CUDA for array processing on a GPU!
PYTHONPATH=.:$PYTHONPATH python3 tests/simple.py --filelist filelist.txt --from-cache --use-cuda
```

## Singularity image

[Singularity image needs to be updated]

## Recommendations on data locality and remote data
In order to make full use of modern CPUs or GPUs, you want to bring the data as close as possible to where the work is done, otherwise you will spend most of the time waiting for the data to arrive rather than actually performing the computations.

With CMS NanoAOD with event sizes of 1-2 kB/event, 1 million events is approximately 1-2 GB on disk. Therefore, you can fit a significant amount of data used in a HEP analysis on a commodity SSD. In order to copy the data to your local disk, use grid tools such as `gfal-copy` or even `rsync` to fetch it from your nearest Tier2. Preserving the filename structure (`/store/...`) will allow you to easily run the same code on multiple sites.

## Frequently asked questions

 - *Why are you doing this array-based analysis business?* Mainly out of curiosity, and I could not find a tool available with which I could do HEP analysis on data on a local disk with MHz rates. It is possible that dask/spark/RDataFrame will soon work well enough for this purpose, but until then, I can justify writing a few functions.
 - *How does this relate to the awkward-array project?* We use the jagged structure provided by the awkward arrays, but implement common HEP functions such as deltaR matching as parallelizable loops or 'kernels' running directly over the array contents, taking into account the event structure. We make these loops fast with Numba, but allow you to debug them by going back to standard python when disabling the compilation.
 - *How does this relate to the coffea/fnal-columnas-analysis-tools project?* It's very similar, you should check out that project! We implement less methods, mostly by explicit loops in Numba, and on GPUs as well as CPUs.
 - *Why don't you use the array operations (`JaggedArray.sum`, `argcross` etc) implemented in awkward-array?* They are great! However, in order to easily use the same code on either the CPU or GPU, we chose to implement the most common operations explicitly, rather than relying on numpy/cupy to do it internally. This also seems to be faster, at the moment.
 - *What if I don't have access to a GPU?* You should still be able to see event processing speeds in the hundreds of kHz to a few MHz for common analysis tasks.
 - *How do I plot my histograms that are saved in the output JSON?* Load the JSON contents and use the `edges` (left bin edges, plus last rightmost edge), `contents` (weighted bin contents) and `contents_w2` (bin contents with squared weights, useful for error calculation) to access the data directly.
 - *I'm a GPU programming expert, and I worry your CUDA kernels are not optimized. Can you comment?* Good question! At the moment, they are indeed not very optimized, as we do a lot of control flow (`if` statements) in them. However, the GPU analysis is still about 2x faster than a pure CPU analysis, as the CPU is more free to work on loading the data, and this gap is expected to increase as the analysis becomes more complicated (more systematics, more templates). At the moment, we see pure GPU processing speeds of about 8-10 MHz for in-memory data, and data loading from cache at about 4-6 MHz. Have a look at the nvidia profiler results [nvprof1](profiling/nvprof1.png), [nvprof2](profiling/nvprof2.png) to see what's going on under the hood. Please give us a hand to make it even better!
 - *What about running this code on multiple machines?* You can do that, currently just using usual batch tools, but we are looking at other ways (dask, joblib, spark) to distribute the analysis across multiple machines. 
 - *What about running this code on data that is remote (XROOTD)?* You can do that thanks to the `uproot` library, but then you gain very little benefit from having a fast CPU or GPU, as you will spend most of your time just waiting for input.
 - *How much RAM is needed?* The amount of RAM determines how much data can be preloaded to memory. You can either process data in memory all at once, which makes rerunning very fast, or set up a batched pipeline. In case of the batched pipeline, no more than a few GB/thread of RAM is needed.
