#!/bin/bash
set -e

export SUBMIT_DIR=$(dirname `pwd`) 

rm -Rf skim_merge
mkdir skim_merge
cd skim_merge

#Prepare list of input files
singularity exec -B /storage ~/gpuservers/singularity/images/cupy2.simg \
    python ../../tests/hmm/prepare_merge_argfile.py \
    -i ../../data/datasets_NanoAODv5.yml \
    --datapath /storage/group/allcit \
    --outpath ~/hmm/skim_merged

cd ..

tar -czf skim_merge.tgz skim_merge
NJOBS=`wc -l skim_merge/args_merge.txt`
echo "Skim jobs prepared: $NJOBS"
echo "Please run 'export SUBMIT_DIR=$SUBMIT_DIR'"
echo "Run 'condor_submit cache.jdl' to submit them"
