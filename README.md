# hepaccelerate

Accelerated array analysis on flat ROOT data. Process 1 billion events to histograms in 10 minutes on a single machine. Weighted histograms, jet-lepton deltaR matching and pileup weighting directly on the GPU! No batch jobs or servers!

Requirements:
 - python 3
 - uproot
 - awkward-array
 - numba

Optional for CUDA acceleration:
 - cupy
 - cudatoolkit

We also provide a singularity image with the python libraries preinstalled, see below.

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