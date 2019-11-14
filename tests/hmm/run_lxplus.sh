#!/bin/bash
#Abort the script if any step fails
set -e

#Use this many threads, should be around 1-4
export NTHREADS=8

#Set to -1 to run on all files, 1 for debugging/testing
export MAXCHUNKS=1

export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=workqueue
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

#This is the location of the input NanoAOD and generally does not need to be changed
export INPUTDATAPATH=/eos/cms/

## Step 2: Prepare the list of files to process and run the physics analysis
python3 tests/hmm/analysis_hmumu.py \
    --action analyze --action merge --maxchunks $MAXCHUNKS \
    --nthreads $NTHREADS \
    --out ./out \
    --datapath $INPUTDATAPATH \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --era 2016 --era 2017 --datasets data --datasets dy
