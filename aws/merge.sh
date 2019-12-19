#!/bin/bash
set -e
set -o xtrace

BUCKET=hepaccelerate-hmm-skim-merged
WORKDIR=/scratch/merge-${AWS_BATCH_JOB_ID}

env
df -h
python --version

mkdir -p $WORKDIR
cd $WORKDIR

#Download sandbox and set up the code
aws s3 cp s3://$BUCKET/sandbox.tgz ./
tar xf sandbox.tgz
cd hepaccelerate-cms
git pull
git checkout aws
git submodule init
git submodule update
cd tests/hmm
make

cd ../..

#Get the input
aws s3 cp --recursive s3://$BUCKET/out ./out

PYTHONPATH=hepaccelerate:coffea:. python3 tests/hmm/merge.py data/datasets_NanoAODv5.yml ./out
PYTHONPATH=coffea:hepaccelerate:. python3 tests/hmm/plotting.py --input out_merged --nthreads 4

tar czf results.tgz out_merged

#Copy the output
aws s3 cp results.tgz s3://$BUCKET/results.tgz

#cleanup
cd /
rm -Rf $WORKDIR
