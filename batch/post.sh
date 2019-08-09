#!/bin/bash

set -e

cd /storage/user/$USER/hmm
\ls -1 out_*.tgz | parallel -j10 --gnu tar --skip-old-files -xf {} \;

cd $SUBMIT_DIR

export SINGULARITY_IMAGE=/storage/user/jpata/cupy2.simg
export PYTHONPATH=coffea:hepaccelerate:.
export CACHE_PATH=/storage/user/jpata/hmm/cache

singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action merge --cache-location $CACHE_PATH \
    --nthreads 16 \
    --out /storage/user/$USER/hmm/out

./tests/hmm/plots.sh /storage/user/$USER/hmm/out 
