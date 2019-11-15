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

\ls -1 out/jobfiles/* | sort -R | parallel --gnu -j24 -n50 \
    singularity exec -B /storage $SINGULARITY_IMAGE \
    python3 tests/hmm/analysis_hmumu.py \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --disable-tensorflow --cachepath $CACHEPATH \
    --jobfiles {} --nthreads 1 --action analyze

singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads 24 \
    --out ./out \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --cachepath $CACHEPATH
