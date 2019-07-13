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



# Misc notes


Luminosity, details on how to set up on this [link](https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html).
~~~
export PATH=$HOME/.local/bin:/cvmfs/cms-bril.cern.ch/brilconda/bin:$PATH
brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
    -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -u /pb --byls --output-style csv -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/ReReco/Final/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt > lumi2016.csv

brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
    -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -u /pb --byls --output-style csv -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/ReReco/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt > lumi2017.csv

brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
    -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -u /pb --byls --output-style csv -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/ReReco/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt > lumi2018.csv


~~~
