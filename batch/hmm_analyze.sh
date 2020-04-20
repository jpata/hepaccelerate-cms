#!/bin/bash

set -e
set -x

ls /storage

env

OUTFILE=$1
INFILE=$2

#Local output director in worker node tmp
export OUTDIR=out

#Set some default arguments
export NTHREADS=2
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_ENABLE_AVX=1
export NUMBA_THREADING_BACKEND=omp
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export NUMBA_BOUNDSCHECK=1
#This is where the skim files are loaded from
export CACHE_PATH=/storage/user/nlu/hmm/skim_merged

WORKDIR=`pwd`

#Create argument list
tar xf jobfiles.tgz 

mkdir $OUTDIR
mv jobfiles/datasets.json $OUTDIR/
mv jobfiles $OUTDIR/

#replace with absolute path in work dir, hack
awk '{print "/srv/out/"$0}' $WORKDIR/$OUTDIR/$INFILE > $WORKDIR/$OUTDIR/$INFILE.tmp
mv $WORKDIR/$OUTDIR/$INFILE.tmp $WORKDIR/$OUTDIR/$INFILE

cat $WORKDIR/$OUTDIR/$INFILE

#Go to code directory
cd $SUBMIT_DIR

#Run the code
python3 tests/hmm/analysis_hmumu.py \
    --action analyze \
    --nthreads $NTHREADS \
    --datapath /storage/group/allcit \
    --do-fsr \
    --do-factorized-jec \
    --out $WORKDIR/$OUTDIR \
    --datasets-yaml data/datasets_NanoAODv6_Run2_mixv1.yml \
    --jobfiles-load $WORKDIR/$OUTDIR/$INFILE

cd $WORKDIR

#copy the output as a tar archive
tar -cvzf out.tgz $OUTDIR
cp out.tgz $OUTFILE
du $OUTFILE

rm -Rf *.json out out.tgz args.txt
echo "job done"
