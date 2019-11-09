#!/bin/bash
#Abort the script if any step fails
set -e

#Use this many threads, should be around 1-4
export NTHREADS=8

#Set to -1 to run on all files, 1 for debugging/testing
export MAXCHUNKS=1

#This is where the intermediate analysis files will be saved and loaded from
#As long as one person produces it, other people can run the analysis on this
#Currently, use the cache provided by Joosep
export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy2.simg
export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=tbb
export NUMBA_ENABLE_AVX=1
export NUMBA_CPU_FEATURES=+sse,+sse2,+avx,+avx2
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

#This is the location of the input NanoAOD and generally does not need to be changed
export INPUTDATAPATH=/storage/group/allcit

## Step 2: Run the physics analysis
singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action analyze --action merge --maxchunks $MAXCHUNKS \
    --nthreads $NTHREADS \
    --out ./out \
    --datapath $INPUTDATAPATH --datasets ttjets_sl \
    --disable-tensorflow
