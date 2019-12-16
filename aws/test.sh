#!/bin/bash
set -e
set -o xtrace

env
df -h
python --version
aws s3 cp s3://hepaccelerate-hmm-skim-merged/files.txt ./files.txt
cat files.txt

aws s3 cp s3://hepaccelerate-hmm-skim-merged/sandbox.tgz ./
tar xf sandbox.tgz
cd hepaccelerate-cms
git checkout aws
git pull
git submodule update

cd tests/hmm
make

cd ../..

aws s3 cp s3://hepaccelerate-hmm-skim-merged/jobfiles.tgz ./
tar xf jobfiles.tgz

mkdir out
PYTHONPATH=coffea:hepaccelerate:. python tests/hmm/run_jd.py jobfiles/jobs.txt $AWS_BATCH_JOB_ARRAY_INDEX out 
ls -al out

for f in ./out/*.pkl; do
    aws s3 cp $f s3://hepaccelerate-hmm-skim-merged/out/`basename $f`
done
