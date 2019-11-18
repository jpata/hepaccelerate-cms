#!/bin/bash

export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=omp
export NUMBA_ENABLE_AVX=0
export NUMBA_CPU_FEATURES=+sse,+sse2,+avx,+avx2
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy2.simg
export CACHEPATH=/storage/user/jpata/hmm/skim_merged

export OUTDIR=out
\ls -1 $OUTDIR/jobfiles/* | sort -R | parallel --gnu -j20 -n5 \
    singularity exec -B /storage $SINGULARITY_IMAGE \
    python3 tests/hmm/analysis_hmumu.py \
    --out $OUTDIR \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --disable-tensorflow --cachepath $CACHEPATH \
    --jobfiles {} --nthreads 1 --action analyze

singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads 24 \
    --out $OUTDIR \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --cachepath $CACHEPATH
