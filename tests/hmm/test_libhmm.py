import numpy as numpy_lib
import time

from hmumu_lib import LibHMuMu, RochesterCorrections, LeptonEfficiencyCorrections

def setup():
    n = 10000000
    pt = 50*numpy_lib.random.rand(n).astype('f')
    eta = 5.0*numpy_lib.random.rand(n).astype('f') - 2.5
    phi = 3.14*numpy_lib.random.rand(n).astype('f') - 3.14
    pdgids = numpy_lib.array([13] * n, dtype=numpy_lib.int32)
    
    libhmm = LibHMuMu()
    
    return pt, eta, phi, pdgids, libhmm

def test_rochester():
    pt, eta, phi, pdgids, libhmm = setup()

    roc = RochesterCorrections(libhmm, "data/RoccoR2017v1.txt")

    t0 = time.time()
    print(roc.compute_kScaleDT(pt, eta, phi))
    t1 = time.time()
    print(t1 - t0)

def test_lepsf(): 
    pt, eta, phi, pdgids, libhmm = setup()
    
    lepsf = LeptonEfficiencyCorrections(libhmm, "data/leptonSF/RunBCDEF_SF_ISO.root", "NUM_LooseRelIso_DEN_MediumID_pt_abseta")
    
    t0 = time.time()
    print(lepsf.compute(pdgids, pt, eta))
    t1 = time.time()
    print(t1 - t0)


if __name__ == "__main__":
    test_rochester()
    test_lepsf()
