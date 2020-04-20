#!/bin/bash
#Abort the script if any step fails
set -e

#Use this many threads, should be around 1-4
export NTHREADS=8

#Set to -1 to run on all files, 1 for debugging/testing
export MAXCHUNKS=1

export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy.simg
export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=omp
export NUMBA_ENABLE_AVX=1
export NUMBA_CPU_FEATURES=+sse,+sse2,+avx,+avx2
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_BOUNDSCHECK=1
#This is the location of the input NanoAOD and generally does not need to be changed
export INPUTDATAPATH=/storage/group/allcit

#nanoAODv5
#export CACHEPATH=/storage/user/jpata/hmm/skim_merged
#nanoAODv6 and private nanoAODs
export CACHEPATH=/storage/user/nlu/hmm/skim_merged

## Step 2: Prepare the list of files to process and run the physics analysis
singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action analyze --action merge --maxchunks $MAXCHUNKS \
    --nthreads $NTHREADS \
    --out ./out --jobfiles ./out/jobfiles/dy_m105_160_amc_2j_2018_24.json \
    --datapath $INPUTDATAPATH \
    --cachepath $CACHEPATH \
    --datasets-yaml data/datasets_NanoAODv6_Run2_mixv1.yml
