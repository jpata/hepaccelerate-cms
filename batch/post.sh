#!/bin/bash
set -e

#Should match RequestCpus in merge.jdl
export NTHREADS=8

cd /storage/user/$USER/hmm
#if [ -d /storage/user/$USER/hmm/out ]; then
#    echo "Output directory /storage/user/$USER/hmm/out exists, please delete it"
#    exit 1
#fi

#Unpack archives
\ls -1 out_*.tgz | xargs -P $NTHREADS -n 1 tar --skip-old-files -xf
cd $SUBMIT_DIR

export PYTHONPATH=coffea:hepaccelerate:.

#Run merge
python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads $NTHREADS \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --out /storage/user/$USER/hmm/out

#Run plots
python3 tests/hmm/plotting.py --input /storage/user/$USER/hmm/out --nthreads $NTHREADS
