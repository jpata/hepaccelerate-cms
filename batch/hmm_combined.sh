#!/bin/bash

set -e

ls /storage

env

workdir=`pwd`

cp -R /storage/user/jpata/hmm/dev/hepaccelerate-cms ./

#Create argument list
tar xf jobfiles.tgz
for word in "$@"; do
    echo $workdir/jobfiles/$word.json >> args.txt
done

#Set some default arguments
export NTHREADS=4
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export DATAPATH=/mnt/hadoop

#This is where the skim files are loaded form
export CACHE_PATH=$workdir/cache

#Local output director in worker node tmp
export OUTDIR=out

cd hepaccelerate-cms

#Run the code
python3 tests/hmm/analysis_hmumu.py \
    --action cache \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --out $workdir/$OUTDIR \
    --datapath $DATAPATH \
    --jobfiles-load $workdir/args.txt

python3 tests/hmm/analysis_hmumu.py \
    --action analyze \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --out $workdir/$OUTDIR \
    --do-factorized-jec \
    --datapath $DATAPATH \
    --jobfiles-load $workdir/args.txt

#cd $workdir
#
##copy the output as a tar archive
#tar -cvzf out.tgz $OUTDIR
#cp out.tgz /storage/user/$USER/hmm/out_$CONDORJOBID.tgz
#du /storage/user/$USER/hmm/out_$CONDORJOBID.tgz
#
#echo "job done"
