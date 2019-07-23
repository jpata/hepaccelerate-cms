#!/bin/bash
#Abort the script if any step fails
set -e

#Set NTHREADS=24 to run on whole machine, =1 for debugging
NTHREADS=20

#Set to -1 to run on all files, 1 for debugging
MAXFILES=-1

#Set to 1 to run analysis on GPU, 0 for debugging
export HEPACCELERATE_CUDA=0

#This is where the intermediate analysis files will be saved and loaded from
#As long as one person produces it, other people can run it
CACHE_PATH=/storage/user/$USER/hmm/cache
#CACHE_PATH=/nvme1/jpata/cache

SINGULARITY_IMAGE=/bigdata/shared/Software/singularity/gpuservers/singularity/images/cupy.simg

if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Singularity image is missing, check the script"
    exit 1
fi


if [ ! -d "$CACHE_PATH" ]; then
    echo "Cache path is missing, check the script"
    exit 1
fi

## Step 1: cache ROOT data (need to repeat only when list of files or branches changes)
## This can take a few hours currently (using MAXFILES=-1 and NTHREADS=24)
PYTHONPATH=coffea:hepaccelerate:. NUMBA_THREADING_LAYER=tbb NUMBA_ENABLE_AVX=0 NUMBA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS HEPACCELERATE_CUDA=0 singularity exec --nv -B /storage -B /nvme1 $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py --action cache --maxfiles $MAXFILES --chunksize 1 --nthreads $NTHREADS --cache-location $CACHE_PATH --datapath /storage/user/jpata/ --era 2016 --era 2017 --era 2018

## Step 2: call physics code, reproduce histograms and all analysis outputs with changed cuts, parameters etc 
## This can take between a few minutes to an hour, depending on the number of files processed
#PYTHONPATH=coffea:hepaccelerate:. NUMBA_THREADING_LAYER=tbb NUMBA_ENABLE_AVX=0 NUMBA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS singularity exec --nv -B /nvme1/ -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py --action analyze --maxfiles $MAXFILES --chunksize 5 --cache-location $CACHE_PATH --datapath /storage/user/jpata/ --pinned --async-data --nthreads $NTHREADS --era 2016
#PYTHONPATH=coffea:hepaccelerate:. NUMBA_THREADING_LAYER=tbb NUMBA_ENABLE_AVX=0 NUMBA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS singularity exec --nv -B /nvme1/ -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py --action analyze --maxfiles $MAXFILES --chunksize 5 --cache-location $CACHE_PATH --datapath /storage/user/jpata/ --pinned --async-data --nthreads $NTHREADS --era 2017
#PYTHONPATH=coffea:hepaccelerate:. NUMBA_THREADING_LAYER=tbb NUMBA_ENABLE_AVX=0 NUMBA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS singularity exec --nv -B /nvme1/ -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py --action analyze --maxfiles $MAXFILES --chunksize 5 --cache-location $CACHE_PATH --datapath /storage/user/jpata/ --pinned --async-data --nthreads $NTHREADS --era 2018

## Step3: make pdf plots based on the output from the previous step
#PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 cmsutils/plotting.py
