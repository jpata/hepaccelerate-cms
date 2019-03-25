import sys
import uproot, cupy

import awkward

import numpy as np
import glob
import psutil, os
from collections import OrderedDict
from typing import List, Dict
import math
import numba
import pyarrow

os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda-9.2/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda-9.2/nvvm/libdevice/"
from numba import cuda

infile = '/nvmedata/store/mc/RunIIFall17NanoAOD/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/20000/0C2B3A66-B042-E811-8C6D-44A8423DE2C0.root'

fi = uproot.open(infile)
tt = fi.get("Events")

arrs = tt.arrays([b"Muon_pt", b"Muon_eta", b"Muon_phi", b"Muon_tightId", b"Muon_charge"], awkwardlib=awkward)

class JaggedStruct(object):
    def __init__(self, offsets, attrs_data, numpy_lib=np):
        self.numpy_lib = numpy_lib
        
        self.offsets = offsets
        self.attrs_data = attrs_data
        
        num_items = None
        for (k, v) in self.attrs_data.items():
            num_items_next = len(v)
            if num_items and num_items != num_items_next:
                raise AttributeError("Mismatched attribute {0}".format(k))
            else:
                num_items = num_items_next
            setattr(self, k, v)
    
        self.mask = self.numpy_lib.ones(num_items, dtype=self.numpy_lib.int8)
        
    @staticmethod
    def from_arraydict(arraydict, prefix, numpy_lib=np):
        ks = [k for k in arraydict.keys() if prefix in str(k, 'ascii')]
        k0 = ks[0]
        return JaggedStruct(
            numpy_lib.array(arraydict[k0].offsets),
            {str(k, 'ascii').replace(prefix, ""): numpy_lib.array(v.content)
             for (k,v) in arraydict.items()},
            numpy_lib=numpy_lib
        )

@cuda.jit('void(int32[:], int64[:], int8[:])')
def select_opposite_sign_muons_cudakernel(muon_charges_content, muon_charges_offsets, muon_mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, muon_charges_offsets.shape[0]-1, xstride):
        start = muon_charges_offsets[iev]
        end = muon_charges_offsets[iev + 1]
        
        ch1 = muon_charges_content[start]
        
        for imuon in range(start+1, end):
            ch2 = muon_charges_content[imuon]
            if (ch2 != ch1):
                muon_mask_out[start] = 0
                muon_mask_out[imuon] = 0
                break
    return


@cuda.jit('void(int8[:], int64[:], int8[:])')
def sum_in_event(content, offsets, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            out[iev] += content[ielem] 

def select_muons_opposite_sign(muons, mask_name=None):
    if not mask_name:
        select_opposite_sign_muons_cudakernel[32,1024](muons.charge, muons.offsets, muons.mask)
    else:
        mask = muons.numpy_lib.ones_like(muons.mask)
        select_opposite_sign_muons_cudakernel[32,1024](muons.charge, muons.offsets, mask)
        setattr(muons, mask_name, mask)
        
def select_muons_pt(muons, ptcut, mask_name=None):
    if not mask_name:
        muons.mask = muons.mask & (muons.pt > ptcut)
    else:
        setattr(muons, mask_name, muons.pt > ptcut)

muons = JaggedStruct.from_arraydict(arrs, "Muon_", cupy)
select_muons_pt(muons, 30, "mask_subleading")
select_muons_pt(muons, 30, "mask_leading")

sum_out = cupy.zeros_like(muons.offsets, dtype=cupy.int8)
print(muons.mask_subleading)
sum_in_event[1, 1](muons.mask_subleading, muons.offsets, sum_out)
pass_subleading = sum_out==2
print(pass_subleading.shape, muons.mask_subleading.shape)

import awkward.cuda
import awkward.cuda.array
import awkward.cuda.array.jagged

import timeit

mu_pts = awkward.cuda.array.jagged.JaggedArrayCuda.fromoffsets(muons.offsets, muons.pt)
mask = mu_pts > 10
print(mu_pts[mask])

