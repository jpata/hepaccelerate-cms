#!/bin/bash
INDIR=$1

SINGULARITY_IMAGE=/storage/user/jpata/cupy2.simg
PYTHONPATH=coffea:hepaccelerate:. singularity exec --nv -B /storage -B /nvme2 $SINGULARITY_IMAGE python3 \
    tests/hmm/plotting.py --input $INDIR \
    --histname hist__dimuon_invmass_z_peak_cat5__dnn_pred \
    --histname hist__dimuon_invmass_z_peak_cat5__inv_mass \
    --histname hist__dimuon_invmass_z_peak_cat5__pt_balance \
    --histname hist__dimuon_invmass_h_peak_cat5__dnn_pred \
    --histname hist__dimuon_invmass_h_peak_cat5__inv_mass \
    --histname hist__dimuon_invmass_h_peak_cat5__pt_balance \
    --histname hist__dimuon_invmass_h_sideband_cat5__dnn_pred \
    --histname hist__dimuon_invmass_h_sideband_cat5__inv_mass \
    --histname hist__dimuon_invmass_h_sideband_cat5__pt_balance \
