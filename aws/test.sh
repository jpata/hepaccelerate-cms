#!/bin/bash
set -e
set -o xtrace

BUCKET=hepaccelerate-hmm-skim-merged

env
df -h
python --version

#Download sandbox and set up the code
aws s3 cp s3://$BUCKET/sandbox.tgz ./
tar xf sandbox.tgz
cd hepaccelerate-cms
git checkout aws
git pull
git submodule update
cd tests/hmm
make

cd ../..

#Get the list of inputs
aws s3 cp s3://$BUCKET/jobfiles.tgz ./
tar xf jobfiles.tgz

#Run the code
mkdir out
PYTHONPATH=coffea:hepaccelerate:. python tests/hmm/run_jd.py jobfiles/jobs.txt $AWS_BATCH_JOB_ARRAY_INDEX out 

#Copy the output
for f in ./out/*.pkl; do
    aws s3 cp $f s3://$BUCKET/out/`basename $f`
done