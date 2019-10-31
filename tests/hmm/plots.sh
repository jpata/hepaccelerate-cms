#!/bin/bash
INDIR=$1

export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy.simg
PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 \
    tests/hmm/plotting.py --input $INDIR --nthreads 8
