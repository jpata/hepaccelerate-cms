#!/bin/bash

set -e

ls /storage

env

#Local output director in worker node tmp
export OUTDIR=out

#Set some default arguments
export NTHREADS=2
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS

#This is where the skim files are loaded from
export CACHE_PATH=/storage/user/jpata/hmm/skim_merged

workdir=`pwd`

#Create argument list
tar xf jobfiles.tgz
for word in "$@"; do
    echo $workdir/$OUTDIR/jobfiles/$word.json >> args.txt
done

mkdir $OUTDIR
mv datasets.json $OUTDIR/
mv jobfiles $OUTDIR/

#Go to code directory
cd $SUBMIT_DIR

#Run the code
python3 tests/hmm/analysis_hmumu.py \
    --action analyze \
    --nthreads $NTHREADS \
    --do-factorized-jec \
    --datapath /storage/group/allcit \
    --cachepath $CACHE_PATH \
    --out $workdir/$OUTDIR \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --jobfiles-load $workdir/args.txt

cd $workdir

#copy the output as a tar archive
tar -cvzf out.tgz $OUTDIR
cp out.tgz /storage/user/$USER/hmm/out_$CONDORJOBID.tgz
du /storage/user/$USER/hmm/out_$CONDORJOBID.tgz

rm -Rf *.json out out.tgz args.txt
echo "job done"
