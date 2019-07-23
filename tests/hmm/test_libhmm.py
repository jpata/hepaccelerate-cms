import numpy as numpy_lib
import time

from hmumu_lib import LibHMuMu, RochesterCorrections, LeptonEfficiencyCorrections, GBREvaluator

def setup():
    n = 10000000
    pt = 50*numpy_lib.random.rand(n).astype('f')
    eta = 5.0*numpy_lib.random.rand(n).astype('f') - 2.5
    phi = 3.14*numpy_lib.random.rand(n).astype('f') - 3.14
    pdgids = numpy_lib.array([13] * n, dtype=numpy_lib.int32)
    charges = numpy_lib.array([-1] * n, dtype=numpy_lib.int32)
    
    libhmm = LibHMuMu()
    
    return pt, eta, phi, pdgids, charges, libhmm

def test_rochester():
    pt, eta, phi, pdgids, charges, libhmm = setup()

    roc = RochesterCorrections(libhmm, "data/RoccoR2017v1.txt")

    t0 = time.time()
    print(roc.compute_kScaleDT(pt, eta, phi, charges))
    t1 = time.time()
    print(t1 - t0)

def test_lepsf(): 
    pt, eta, phi, pdgids, charges, libhmm = setup()
    
    lepsf = LeptonEfficiencyCorrections(libhmm, ["data/leptonSF/2016/RunBCDEF_SF_ISO.root", "data/leptonSF/2016/RunGH_SF_ISO.root"],
            ["NUM_LooseRelIso_DEN_MediumID_eta_pt", "NUM_LooseRelIso_DEN_MediumID_eta_pt"], [0.5, 0.5])
    
    t0 = time.time()
    print(lepsf.compute(pdgids, pt, eta))
    t1 = time.time()
    print(t1 - t0)

def test_gbr():

    pt, eta, phi, pdgids, charges, libhmm = setup()
    gbr = GBREvaluator(libhmm, "data/TMVAClassification_BDTG.weights.2jet_bveto.xml")

    nfeat = gbr.get_bdt_nfeatures()
    nev = 100
    X = numpy_lib.zeros((nev, nfeat), dtype=numpy_lib.float32)
    for i in range(nev):
        X[i, :] = i
    out = gbr.compute(X)
    print(out)

if __name__ == "__main__":
    #test_rochester()
    #test_lepsf()
    test_gbr()