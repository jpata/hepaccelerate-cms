#!/bin/bash
mkdir skim_merge
cd skim_merge

singularity exec -B /storage ~/gpuservers/singularity/images/cupy2.simg python ../../tests/hmm/prepare_merge_argfile.py -i ../../data/datasets_NanoAODv5.yml --datapath /storage/group/allcit --outpath ~/skim_merged

cd ..

tar -cvzf skim_merge.tgz skim_merge
