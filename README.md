# hepaccelerate

Accelerated array analysis on flat ROOT data. Process 1 billion events to histograms in 10 minutes on a single GPU workstation. Weighted histograms, jet-lepton deltaR matching and pileup weighting directly on the GPU! Or, if you don't have a GPU, no big deal, the same code works on the CPU too, just somewhat slower! No batch jobs or servers required!

Requirements:
 - python 3
 - uproot
 - awkward-array
 - numba
 - fnal-column-analysis-tools

Optional for CUDA acceleration:
 - cupy
 - cudatoolkit

We also provide a singularity image with the python libraries preinstalled, see below.

## Sneak peek

```python
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
```

## Benchmark on a CMS Higgs analysis

The following benchmarks have been carried out with a realistic CMS analysis on NanoAOD.

This proto-analysis implements the following:

 (x) muon selection: pT leading and subleading, eta, ID, isolation, opposite charge, matching to trigger objects
 (x) jet selection: pt, eta, ID & PU ID, remove jets dR-matched to leptons
 (x) event selection: MET filters, trigger, primary vertex quality cuts, two opposite sign muons
 (x) high-level variables: dimuon invariant mass
 (x) PU and gen weight computation
 (x) on the fly luminosity calculation, golden JSON lumi filtering
 (x) weighted histograms of muon, jet and event variables

Not yet implemented:

 ( ) muon momentum Rochester corrections
 ( ) lepton scale factors
 ( ) in-situ training and evaluation of Higgs-to-bkg discriminators

The analysis is carried out using a single script, `tests/analyse_hmumu.py -a cache -a analyze --nthreads N (--use-cuda) --async-data`. In order to be able to process a billion events, the first time the analysis is run, the branches that we will use are uncompressed and saved to local disk as numpy files. Without this optional caching step, the upper limit on the processing speed will be dominated by the CPU decompression of the ROOT TTree (first line `caching` in the tables). This means that the second time you run the analysis with reading the same branches, the decompression does not need to be done and we can achieve about 3-4x higher speeds.

### 2015 Macbook Pro, 35M events

We can process 35M events on a macbook in about 5 minutes.

- 4-core 2.6GHz i5
- 8GB DDR3
- 250GB SSD PT250B over USB3
- analyzed 35,041,780 events in 2.36 GB of branches

Overall processed 3.50E+07 (35041780) events in total 2.36 GB, 365.5 seconds, 9.59E+04 Hz, 6.60 MB/s

task    | configuration  |time     | speed       | speed
--------|----------------|---------|-------------|-----------
caching | CPU(4)         | 365.5 s | 9.59E+04 Hz | 6.60 MB/s
analyze | CPU(4)         | 116.6 s | 3.01E+05 Hz | 20.70 MB/s


### High-end workstation, 1B events

We can demonstrate that we can process 1B events in about 4 minutes (from existing caches) and in about 20 minutes from raw NanoAOD. The maximum system throughput is limited by the read speed of the P4608 SSD, which is 1.5GB/s.

- 16-core 3GHz i7
- 64GB DDR3
- 6.4TB Intel P4608
- 1x Titan X 12GB
- analyzed 978,711,446 events in 62.70 GB of branches


task    | configuration           |time      | speed       | speed
--------|-------------------------|----------|-------------|-----------
caching | CPU(16)                 | 1008.8 s | 9.70E+05 Hz | 63.65 MB/s
analyze | CPU(16)                 | 374.7 s  | 2.61E+06 Hz | 171.38 MB/s
analyze | GPU(1) CPU(16) batch(8) | 229.4 s  | 4.27E+06 Hz | 279.87 MB/s


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
Singularity allows you to try out the package without needing to install the python libraries.

```bash
#Download the uproot+cupy+numba singularity image from CERN
wget https://jpata.web.cern.ch/jpata/singularity/cupy.simg -o singularity/cupy.simg

##or on lxplus
#cp /eos/user/j/jpata/www/singularity/cupy.simg ./singularity/
##or compile yourself if you have ROOT access on your machine
#cd singularity;make

LC_ALL=C PYTHONPATH=.:$PYTHONPATH singularity exec -B /nvmedata --nv singularity/cupy.simg python3 ...
```

## Recommendations on data locality
In order to make full use of modern CPUs or GPUs, you want to bring the data as close as possible to where the work is done, otherwise you will spend most of the time waiting for the data to arrive rather than actually performing the computations.
With CMS NanoAOD with event sizes of 1-2 kB/event, 1 million events is approximately 1-2 GB on disk. Therefore, you can fit a significant amount of data used in a HEP analysis on a commodity SSD. In order to copy the data to your local disk, use grid tools such as `gfal-copy` or even `rsync` to fetch it from your nearest Tier2. Preserving the filename structure (`/store/...`) will allow you to easily run the same code on multiple sites.