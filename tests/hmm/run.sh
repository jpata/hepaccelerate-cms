#!/bin/bash
set -e

#Set NTHREADS=24 to run on whole machine, =1 for debugging
NTHREADS=1

#Set to -1 to run on all files, 1 for debugging
MAXFILES=1

#Set to 1 to run analysis on GPU, 0 for debugging
HEPACCELERATE_CUDA=0

#This is where the intermediate analysis files will be saved and loaded from
CACHE_PATH=/storage/user/$USER/hmm/cache2

SINGULARITY_IMAGE=/bigdata/shared/Software/singularity/gpuservers/singularity/images/cupy.simg
##Step 1: cache ROOT data (need to repeat only when list of files or branches changes)
PYTHONPATH=coffea:hepaccelerate:. NUMBA_THREADING_LAYER=tbb NUMBA_ENABLE_AVX=1 NUMBA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS HEPACCELERATE_CUDA=0 singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py --action cache --maxfiles $MAXFILES --chunksize 1 --nthreads $NTHREADS --cache-location $CACHE_PATH --datapath /storage/user/jpata/

##Step 2: call physics code
PYTHONPATH=coffea:hepaccelerate:. NUMBA_THREADING_LAYER=tbb NUMBA_ENABLE_AVX=1 NUMBA_NUM_THREADS=$NTHREADS OMP_NUM_THREADS=$NTHREADS singularity exec --nv -B /nvme1/ -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py --action analyze --maxfiles $MAXFILES --chunksize 10 --cache-location $CACHE_PATH --datapath /storage/user/jpata/ --pinned --async-data --nthreads $NTHREADS

##Step3: make plots
PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 cmsutils/plotting.py
