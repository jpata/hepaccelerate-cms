#!/bin/bash
#Abort the script if any step fails
set -e

#Use this many threads, should be around 1-4
export NTHREADS=4

#Set to -1 to run on all files, 1 for debugging/testing
export MAXCHUNKS=1

#This is where the intermediate analysis files will be saved and loaded from
#As long as one person produces it, other people can run the analysis on this
#Currently, use the cache provided by Joosep
#export CACHE_PATH=/storage/user/jpata/hmm/cache
#export CACHE_PATH=mycache
export CACHE_PATH=/storage/user/nlu/hmm/cache2
export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy.simg
export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=tbb
export NUMBA_ENABLE_AVX=1
export NUMBA_CPU_FEATURES=+sse,+sse2,+avx,+avx2
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

#This is the location of the input NanoAOD and generally does not need to be changed
export INPUTDATAPATH=/storage/user/jpata

## Step 1: cache ROOT data (need to repeat only when list of files or branches changes)
## This can take a few hours currently for the whole run (using maxchunks -1 and --nthreads 24)
#singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
#   --action cache --maxchunks $MAXCHUNKS --chunksize 1 \
#   --nthreads 1 --cache-location $CACHE_PATH \
#   --datapath $INPUTDATAPATH

## Step 2: Run the physics analysis
singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action analyze --action merge --maxchunks $MAXCHUNKS \
    --cache-location $CACHE_PATH \
    --nthreads $NTHREADS \
    --out ./out --do-factorized-jec \
    --datasets ggh_amcPS --eras 2016 \
    --datapath $INPUTDATAPATH
