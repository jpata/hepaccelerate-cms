#!/bin/bash
set -e
set -o xtrace

BUCKET=hepaccelerate-hmm-skim-merged

env
df -h
python --version

#Get the input
aws s3 cp --recursive s3://$BUCKET/out /tmp/out

PYTHONPATH=hepaccelerate:coffea:. python3 tests/hmm/merge.py data/datasets_NanoAODv5.yml /tmp/out

#Copy the output
for f in ./out_merged/*.pkl; do
    aws s3 cp $f s3://$BUCKET/out_merged/`basename $f`
done
