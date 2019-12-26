[![Build Status](https://travis-ci.com/jpata/hepaccelerate-cms.svg?branch=master)](https://travis-ci.com/jpata/hepaccelerate-cms)

# hepaccelerate-cms

CMS-specific (optionally) GPU-accelerated analysis code based on the [hepaccelerate](https://github.com/hepaccelerate/hepaccelerate) backend library.

Currently implemented analyses:
- `tests/hmm/analysis_hmumu.py`: CMS-HIG-19-006, [internal](http://cms.cern.ch/iCMS/analysisadmin/cadilines?line=HIG-19-006&tp=an&id=2254&ancode=HIG-19-006)

Variations of this code have been tested at:
- T2_US_Caltech (jpata, nlu)
- Caltech HPC, http://www.hpc.caltech.edu/ (jpata)
- T2_US_Purdue
- T3_CH_PSI

This code relies on NanoAOD files being available on the local filesystem for the best performance. It is possible to use xrootd, but currently, this is not the primary focus in the interest of maximum throughput, and thus is not officially supported. The full NanoAOD for a Run 2 analysis is on the order of 5TB (1.6TB skimmed), which is generally feasible to store on local disk.

## Installation on lxplus

This code can be tested on lxplus, with the input files located on `/eos/cms/store`.
~~~
#Create the python environment
python3 -m venv venv-hepaccelerate
source venv-hepaccelerate/bin/activate
pip3 install awkward uproot numba tqdm lz4 cloudpickle scipy pyyaml cffi six tensorflow psutil xxhash keras

#Get the code
git clone https://github.com/jpata/hepaccelerate-cms.git
cd hepaccelerate-cms
git submodule init
git submodule update

#Compile the C++ helper code (Rochester corrections and lepton sf, ROOT is needed)
cd tests/hmm/
make
cd ../..

#Run the code on a few NanoAOD files from EOS
./tests/hmm/run_lxplus.sh
~~~

## Installation on Caltech T2 or GPU machine

On Caltech, an existing singularity image can be used to get the required python libraries.
~~~
git clone https://github.com/jpata/hepaccelerate-cms.git
cd hepaccelerate-cms
git submodule init
git submodule update

#Compile the C++ helpers
cd tests/hmm
singularity exec /storage/user/jpata/gpuservers/singularity/images/cupy.simg make -j4
cd ../..

#Run the code as a small test (small subset of the data by default, edit the file to change this)
#This should take approximately 10 minutes and processes 1 file from each dataset for each year
./tests/hmm/run.sh
~~~

## Running on full dataset using batch queue
We use the condor batch queue on Caltech T2 to run the analysis. It takes ~20 minutes for all 3 years using just the Total JEC & JER (2-3h using factorized JEC) using about 200 job slots.

~~~
#Submit batch jobs after this step is successful
mkdir /storage/user/$USER/hmm
export SUBMIT_DIR=`pwd`

#Prepare the list of datasets (out/datasets.json) and the jobfiles (out/jobfiles/*.json) 
./tests/hmm/run.sh

cd batch
mkdir logs

#Run the NanoAOD skimming step (cache creation).
#This is quite heavy (~6h total), so do this only
#when adding new samples
./make_cache_jdl.sh
condor_submit cache.jdl
#...wait until done, create resubmit file if needed
python verify_cache.py
du -csh ~/hmm/skim_merged

#Now run the analysis, this can be between 20 minutes and a few hours
./make_submit_jdl.sh
condor_submit analyze.jdl
#...wait until done, create resubmit file if needed
python verify_analyze.py
du -csh ~/hmm/out_*.tgz

#submit merging and plotting, this should be around 30 minutes
condor_submit merge.jdl
du -csh ~/hmm/out

cd ..

#when all was successful, delete partial results
rm -Rf /storage/user/$USER/hmm/out/partial_results
du -csh /storage/user/$USER/hmm/out
~~~

## Making plots, datacards and histograms
From the output results, one can make datacards and plots by executing this command:
~~~
./tests/hmm/plots.sh out
~~~
This creates a directory called `baseline` which has the datacards and plots. This can also be run on the batch using `merge.jdl`.

# Contributing
If you use this code, we are happy to consider issues and merge improvements.
- Please make an issue on the Issues page for any bugs you find.
- To contribute changes, please use the 'Fork and Pull' model: https://reflectoring.io/github-fork-and-pull.
- For non-trivial pull requests, please ask at least one other person with push access to review the changes.

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
