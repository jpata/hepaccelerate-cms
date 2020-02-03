#!/bin/bash
set -e

export SINGULARITY_IMAGE=~jpata/gpuservers/singularity/images/cupy.simg

#echo data/nano_2016_data.root > skim.txt
#singularity exec -B /storage $SINGULARITY_IMAGE python2 tests/hmm/skim_and_recompress.py \
#    -i ./skim.txt -o ./data/nano_2016_data_skim.root \
#    -s "(HLT_IsoMu24 || HLT_IsoTkMu24) && nMuon>=2" -t ./
#
#echo data/myNanoProdMc2016_NANO.root > skim.txt
#singularity exec -B /storage $SINGULARITY_IMAGE python2 tests/hmm/skim_and_recompress.py \
#    -i ./skim.txt -o ./data/myNanoProdMc2016_NANO_skim.root \
#    -s "(HLT_IsoMu24 || HLT_IsoTkMu24) && nMuon>=2" -t ./

PYTHONPATH=hepaccelerate:coffea:. singularity exec -B /storage $SINGULARITY_IMAGE \
    python3 tests/hmm/test_hmumu_utils.py --debug

#PYTHONPATH=hepaccelerate:coffea:. singularity exec -B /storage $SINGULARITY_IMAGE \
#    python3 tests/hmm/test_analysis_full.py --debug
