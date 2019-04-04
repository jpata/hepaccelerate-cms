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

## Benchmarks

The following benchmarks have been carried out with a realistic CMS analysis on NanoAOD. We first perform a caching step, where the branches that we will use are uncompressed and saved to local disk as numpy files. Without this optional caching step, the upper limit on the processing speed will be dominated by the CPU decompression of the ROOT TTree (first line in the tables). Even with the caching, the IO time of reading the cached arrays is significant, such that the analysis receives only a small boost from multiple CPU threads, and a another small boost from using the GPU and thus freeing the CPU to deal with loading the data from disk.
However, if the analysis part becomes moderately complex


### 2015 Macbook Pro 

- 4-core 2.6GHz i5
- 8GB DDR3
- 250GB SSD PT250B over USB3
- analyzed 35,041,780 events in 2.07GB of branches

task    | configuration  |time    | speed       | speed
--------|----------------|--------|-------------|-----------
caching | CPU(4)         | 378.5s | 9.26E+04 Hz | 5.60 MB/s
analyze | CPU(1)         | 170.9s | 2.05E+05 Hz | 12.41 MB/s
analyze | CPU(2)         | 163.5s | 2.14E+05 Hz | 12.98 MB/s
analyze | CPU(4)         | 161.6s | 2.17E+05 Hz | 13.12 MB/s


### High-end workstation

- 16-core 3GHz i7
- 64GB DDR3
- 6.4TB Intel P4608
- 1x Titan X 12GB
- analyzed 214,989,146 events in 11.87 GB of branches

task    | configuration           |time           | speed       | speed
--------|-------------------------|---------------|-------------|-----------
caching | CPU(16)                 | 281.3 seconds | 7.64E+05 Hz | 43.22 MB/s
analyze | CPU(1)                  | 180.7 seconds | 1.19E+06 Hz | 67.27 MB/s
analyze | CPU(2)                  | 141.1 seconds | 1.52E+06 Hz | 86.16 MB/s
analyze | CPU(4)                  | 140.4 seconds | 1.53E+06 Hz | 86.54 MB/s
analyze | CPU(8)                  | 139.0 seconds | 1.55E+06 Hz | 87.44 MB/s
analyze | CPU(16)                 | 127.9 seconds | 1.68E+06 Hz | 95.05 MB/s
analyze | GPU(1) CPU(16)          | 115.8 seconds | 1.86E+06 Hz | 105.00 MB/s
analyze | GPU(1) CPU(16) batch(2) | 93.7 seconds  | 2.29E+06 Hz | 129.72 MB/s
analyze | GPU(1) CPU(16) batch(4) | 85.9 seconds  | 2.50E+06 Hz | 141.54 MB/s
analyze | GPU(1) CPU(16) batch(8) | 85.1 seconds  | 2.53E+06 Hz | 142.83 MB/s


### Workstation, 1B event run

We can demonstrate that we can process 1B events in about 10 minutes (from existing caches) and in about 30 minutes from raw NanoAOD.

- 978,711,446 events in 54.95 GB of branches
- analyze as before
- analyze, but make 100 more histograms

task                     | configuration           |time             | speed       | speed
-------------------------|-------------------------|-----------------|-------------|-----------
caching                  | CPU(16)                 | 1127.0 seconds  | 8.68E+05 Hz | 49.93 MB/s
analyze                  | CPU(16)                 | 594.1 seconds   | 1.65E+06 Hz | 94.70 MB/s
analyze                  | GPU(1) CPU(16) batch(4) | 565.7 seconds   | 1.73E+06 Hz | 99.46 MB/s
analyze (100 histograms) | CPU(16)                 | 1232.4 seconds  | 7.94E+05 Hz | 45.65 MB/s
analyze (100 histograms) | GPU(1) CPU(16) batch(4) | 832.5 seconds   | 1.18E+06 Hz | 67.59 MB/s


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

## Recommendations on data locality
In order to make full use of modern CPUs or GPUs, you want to bring the data as close as possible to where the work is done, otherwise you will spend most of the time waiting for the data to arrive rather than actually performing the computations.
With CMS NanoAOD with event sizes of 1-2 kB/event, 1 million events is approximately 1-2 GB on disk. Therefore, you can fit a significant amount of data used in a HEP analysis on a commodity SSD. In order to copy the data to your local disk, use grid tools such as `gfal-copy` or even `rsync` to fetch it from your nearest Tier2. Preserving the filename structure (`/store/...`) will allow you to easily run the same code on multiple sites.