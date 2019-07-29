#!/bin/bash
INDIR=$1

SINGULARITY_IMAGE=/storage/user/jpata/cupy.simg
PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 tests/hmm/plotting.py --input $INDIR
