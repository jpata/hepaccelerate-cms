#!/bin/bash

set -e

ls /storage

env

workdir=`pwd`
tar xf jobfiles.tgz

for word in "$@"; do
    echo $workdir/jobfiles/$word.json >> args.txt
done

export JOBFILE=$1
export NTHREADS=4
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

export CACHE_PATH=/storage/user/$USER/hmm/cache
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export OUTDIR=out

cd /data/jpata/hmumu/hepaccelerate-cms/

python3 tests/hmm/analysis_hmumu.py \
    --action analyze \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --datapath /storage/user/jpata/ \
    --out $workdir/$OUTDIR \
    --do-factorized-jec \
    --jobfiles-load $workdir/args.txt


cd $workdir

tar -cvzf out.tgz $OUTDIR
cp out.tgz /storage/user/$USER/hmm/out_$CONDORJOBID.tgz
du /storage/user/$USER/hmm/out_$CONDORJOBID.tgz

echo "job done"
