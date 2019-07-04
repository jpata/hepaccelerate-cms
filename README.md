[![Build Status](https://travis-ci.com/jpata/hepaccelerate.svg?branch=master)](https://travis-ci.com/jpata/hepaccelerate-cms)
[![pipeline status](https://gitlab.cern.ch/jpata/hepaccelerate/badges/libhmm/pipeline.svg)](https://gitlab.cern.ch/jpata/hepaccelerate-cms/commits/master)

# hepaccelerate-cms

CMS-specific accelerated analysis code based on the [hepaccelerate](https://github.com/jpata/hepaccelerate) library.

~~~
#Installation
pip3 install --user awkward uproot numba
git clone https://github.com/jpata/hepaccelerate-cms.git
cd hepaccelerate-cms
git submodule init
git submodule update

#Compile the C++ helper code (Rochester corrections and lepton sf, ROOT is needed)
cd tests/hmm/
make
cd ../..

#Produce ntuple caches (one time only)
PYTHONPATH=hepaccelerate:coffea:. python3 tests/hmm/analysis_hmumu.pu --datapath /path/to/store/cms/ --cache-location /path/to/fast/ssd --maxfiles 5 --action cache

#Run analysis
PYTHONPATH=hepaccelerate:coffea:. python3 tests/hmm/analysis_hmumu.pu --datapath /path/to/store/cms/ --cache-location /path/to/fast/ssd --maxfiles 5 --action analyze

#Produce plots from out/baseline/*.json
PYTHONPATH=hepaccelerate:coffea:. python3 cmsutils/plotting.py
~~~

Best results can be had if the CMS data is stored locally on a filesystem (few TB needed) and if you have a cache disk on the analysis machine of a few hundred GB.

A prebuilt singularity image with the GPU libraries is also provided: [link](http://login-1.hep.caltech.edu/~jpata/cupy.simg)
