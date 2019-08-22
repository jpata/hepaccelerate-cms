#!/bin/bash

hostname

set -e

ls /storage

env

workdir=`pwd`

#Set some default arguments
export NTHREADS=24
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS

#This is where the skim files are loaded form
export CACHE_PATH=/storage/user/$USER/hmm/cache

#Local output director in worker node tmp
export OUTDIR=out

#Go to code directory
cd $SUBMIT_DIR

#Run the code
rm -f $CACHE_PATH/datasets.json
python3 tests/hmm/analysis_hmumu.py \
    --action cache \
    --nthreads $NTHREADS \
    --cache-location $CACHE_PATH \
    --datapath /storage/user/jpata/ \
    --maxchunks -1 --chunksize 1 \
    --out $workdir/out

echo "job done"
