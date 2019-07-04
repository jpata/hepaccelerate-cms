#!/bin/bash
PYTHONPATH=hepaccelerate:coffea:. singularity exec --nv -B /nvme1/ -B /storage /bigdata/shared/Software/singularity/ibanks/edge_r.simg python3 tests/hmm/analysis_hmumu.py --action $1 --datapath /storage/user/jpata/ --cache-location /nvme1/jpata/mycache_all_presel --maxfiles 1
