#!/bin/bash

set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately

export NTHREADS=24
export HEPACCELERATE_CUDA=1
export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=tbb
export NUMBA_ENABLE_AVX=1
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS 
export SINGULARITY_IMAGE=/bigdata/shared/Software/singularity/gpuservers/singularity/images/cupy.simg

function run_code() {
    singularity exec --nv -B /storage -B /nvme1 $SINGULARITY_IMAGE python3 \
        tests/hmm/analysis_hmumu.py --action analyze --maxfiles -1 --chunksize 2 \
        --cache-location /nvme1/jpata/cache --datapath /storage/user/jpata/ \
        --pinned --async-data --nthreads $NTHREADS --era 2018 --out out2 --dataset $1
}

function run_code_smallsamples() {
    singularity exec --nv -B /storage -B /nvme1 $SINGULARITY_IMAGE python3 \
        tests/hmm/analysis_hmumu.py --action analyze --maxfiles -1 --chunksize 2 \
        --cache-location /nvme1/jpata/cache --datapath /storage/user/jpata/ \
        --pinned --async-data --nthreads $NTHREADS --era 2018 --out out2 \
        --dataset ggh --dataset vbf --dataset ttw --dataset ttz \
        --dataset st_t_top --dataset st_t_antitop --dataset st_tw_antitop --dataset st_tw_top \
        --dataset zz --dataset wmh --dataset wph --dataset zh \
        --dataset tth
}

#run_code vbf
#run_code dy_m105_160_vbf_amc

run_code_smallsamples
run_code data
run_code dy
run_code dy_0j
run_code dy_1j
run_code dy_2j
run_code dy_m105_160_amc
run_code dy_m105_160_vbf_amc
run_code ttjets_dl
run_code ttjets_sl
run_code ewk_lljj_mll50_mjj120
run_code ewk_lljj_mll105_160
run_code ww_2l2nu
run_code wz_3lnu
run_code wz_2l2q
