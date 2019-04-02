import os
import numba
import numpy as np
import math

@numba.jit(fastmath=True)
def searchsorted_devfunc(arr, val):
    ret = -1
    for i in range(len(arr)):
        if val <= arr[i]:
            ret = i
            break
    return ret

#need atomics to add to bin contents
@numba.jit
def fill_histogram(data, weights, bins, out_w, out_w2):
    for i in range(len(data)):
        bin_idx = searchsorted_devfunc(bins, data[i])
        if bin_idx >=0 and bin_idx < len(out_w):
            out_w[bin_idx] += weights[i]
            out_w2[bin_idx] += weights[i]**2

@numba.njit(parallel=True)
def select_opposite_sign_muons_kernel(muon_charges_content, muon_charges_offsets, content_mask_in, content_mask_out):
    
    for iev in numba.prange(muon_charges_offsets.shape[0]-1):
        start = muon_charges_offsets[iev]
        end = muon_charges_offsets[iev + 1]
        
        ch1 = 0
        idx1 = -1
        ch2 = 0
        idx2 = -1
        
        for imuon in range(start, end):
            if not content_mask_in[imuon]:
                continue
                
            if idx1 == -1:
                ch1 = muon_charges_content[imuon]
                idx1 = imuon
                continue
            else:
                ch2 = muon_charges_content[imuon]
                if (ch2 != ch1):
                    idx2 = imuon
                    content_mask_out[idx1] = 1
                    content_mask_out[idx2] = 1
                    break
    return

@numba.njit(parallel=True)
def sum_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):

    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] += content[ielem]
            
@numba.njit(parallel=True)
def max_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):

    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] > accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum

        
@numba.njit(parallel=True)
def min_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):

    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] < accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum
    
@numba.njit(parallel=True)
def get_in_offsets_kernel(content, offsets, indices, mask_rows, mask_content, out):
    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
        start = offsets[iev]
        end = offsets[iev + 1]
        
        index_to_get = 0
        for ielem in range(start, end):
            if mask_content[ielem]:
                if index_to_get == indices[iev]:
                    out[iev] = content[ielem]
                    break
                else:
                    index_to_get += 1
        
@numba.njit(parallel=True)
def min_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):
    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] < accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum
        
def sum_in_offsets(struct, content, mask_rows, mask_content, dtype=None):
    if not dtype:
        dtype = content.dtype
    sum_offsets = np.zeros(len(struct.offsets) - 1, dtype=dtype)
    sum_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, sum_offsets)
    return sum_offsets

def max_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = np.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    max_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, max_offsets)
    return max_offsets

def min_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = np.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    min_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, max_offsets)
    return max_offsets

def select_muons_opposite_sign(muons, in_mask):
    out_mask = np.invert(muons.make_mask())
    select_opposite_sign_muons_kernel(muons.charge, muons.offsets, in_mask, out_mask)
    return out_mask

def get_in_offsets(content, offsets, indices, mask_rows, mask_content):
    out = np.zeros(len(offsets) - 1, dtype=content.dtype)
    get_in_offsets_kernel(content, offsets, indices, mask_rows, mask_content, out)
    return out

"""
For all events (N), mask the objects in the first collection (M1) if they are closer than dr2 to any object in the second collection (M2).

    etas1: etas of the first object, array of (M1, )
    phis1: phis of the first object, array of (M1, )
    mask1: mask (enabled) of the first object, array of (M1, )
    offsets1: offsets of the first object, array of (N, )

    etas2: etas of the second object, array of (M2, )
    phis2: phis of the second object, array of (M2, )
    mask2: mask (enabled) of the second object, array of (M2, )
    offsets2: offsets of the second object, array of (N, )
    
    mask_out: output mask, array of (M1, )

"""
@numba.njit(parallel=True)
def mask_deltar_first_kernel(etas1, phis1, mask1, offsets1, etas2, phis2, mask2, offsets2, dr2, mask_out):
    
    for iev in numba.prange(len(offsets1)-1):
        a1 = offsets1[iev]
        b1 = offsets1[iev+1]
        
        a2 = offsets2[iev]
        b2 = offsets2[iev+1]
        
        for idx1 in range(a1, b1):
            if not mask1[idx1]:
                continue
                
            eta1 = etas1[idx1]
            phi1 = phis1[idx1]
            for idx2 in range(a2, b2):
                if not mask2[idx2]:
                    continue
                eta2 = etas2[idx2]
                phi2 = phis2[idx2]
                
                deta = abs(eta1 - eta2)
                dphi = (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi
                
                #if first object is closer than dr2, mask element will be *disabled*
                passdr = ((deta**2 + dphi**2) < dr2)
                mask_out[idx1] = mask_out[idx1] | passdr
                
def mask_deltar_first(objs1, mask1, objs2, mask2, drcut):
    assert(mask1.shape == objs1.eta.shape)
    assert(mask2.shape == objs2.eta.shape)
    assert(objs1.offsets.shape == objs2.offsets.shape)
    
    mask_out = np.zeros_like(objs1.eta, dtype=np.bool)
    mask_deltar_first_kernel(
        objs1.eta, objs1.phi, mask1, objs1.offsets,
        objs2.eta, objs2.phi, mask2, objs2.offsets,
        drcut**2, mask_out
    )
    mask_out = np.invert(mask_out)
    return mask_out

def histogram_from_vector(data, weights, bins):        
    out_w = np.zeros(len(bins) - 1, dtype=np.float32)
    out_w2 = np.zeros(len(bins) - 1, dtype=np.float32)
    fill_histogram(data, weights, bins, out_w, out_w2)
    return out_w, out_w2, bins
    
@numba.njit(parallel=True)
def get_bin_contents_kernel(values, edges, contents, out):
    for i in numba.prange(len(values)):
        v = values[i]
        ibin = searchsorted_devfunc(edges, v)
        if ibin>=0 and ibin < len(contents):
            out[i] = contents[ibin]

def get_bin_contents(values, edges, contents, out):
    assert(values.shape == out.shape)
    assert(edges.shape[0] == contents.shape[0]+1)
    get_bin_contents_kernel(values, edges, contents, out)
