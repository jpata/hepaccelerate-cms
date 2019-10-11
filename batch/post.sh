#!/bin/bash

set -e

cd /storage/user/$USER/hmm
#if [ -d /storage/user/$USER/hmm/out ]; then
#    echo "Output directory /storage/user/$USER/hmm/out exists, please delete it"
#    exit 1
#fi

#Unpack archives
\ls -1 out_*.tgz | xargs -P 24 -n 1 tar --skip-old-files -xf

cd /storage/user/idutta/Hmm/Vectorized/my_fork/hepaccelerate-cms/

export PYTHONPATH=coffea:hepaccelerate:.

#Run merge
python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads 4 \
    --out /storage/user/$USER/hmm/out

#Run plots
python3 tests/hmm/plotting.py --input /storage/user/$USER/hmm/out --nthreads 4
