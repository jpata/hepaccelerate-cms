#!/bin/bash
SINGULARITY_IMAGE=/bigdata/shared/Software/singularity/gpuservers/singularity/images/cupy.simg
PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 cmsutils/plotting.py
