#!/bin/bash
INDIR=$1

SINGULARITY_IMAGE=/storage/user/jpata/cupy2.simg
PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage -B /nvme2 $SINGULARITY_IMAGE python3 \
    tests/hmm/plotting.py --input $INDIR
