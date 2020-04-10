from hmumu_lib import BTagCalibration, LibHMuMu
from coffea.btag_tools import BTagScaleFactor
import time
import uproot
import numpy as np

if __name__ == "__main__":
    sf_file = "data/btagSF/DeepCSV_102XSF_V1.csv"
    fi = uproot.open("data/myNanoProdMc2016_NANO_skim.root")
    tt = fi.get("Events")
    arr_flav = np.array(tt.array("Jet_hadronFlavour").content, np.int32)
    arr_abs_eta = np.array(np.abs(tt.array("Jet_eta").content), np.float32)
    arr_pt = np.array(tt.array("Jet_pt").content, np.float32)
    arr_discr = np.array(tt.array("Jet_btagDeepB").content, np.float32)
    
    systs = ["jes", "lfstats1", "lfstats2", "hfstats1", "hfstats2", "cferr1", "cferr2", "lf", "hf"]
    
    t0 = time.time()
    sf = BTagScaleFactor(sf_file, BTagScaleFactor.RESHAPE, 'iterativefit,iterativefit,iterativefit', keep_df=True)
    for tsys in systs:
        for sdir in ["up", "down"]:
            tsys_name = sdir + '_' + tsys
            sf.eval(tsys_name, arr_flav[:1], arr_abs_eta[:1], arr_pt[:1], arr_discr[:1], True)
    t1 = time.time()
    print("init_py", t1 - t0)
    
    t0 = time.time()
    libhmm = LibHMuMu()
    print("loading BTagCalibration")
    systs_sdir = []
    for sdir in ["up", "down"]:
        for syst in systs:
            systs_sdir += [sdir + "_" + syst]
    b = BTagCalibration(libhmm, "DeepCSV", sf_file, systs_sdir)
    t1 = time.time()
    print("init_C", t1 - t0)
    
    for syst in systs:
        for sdir in ["up", "down"]:
            print(syst, sdir)
    
            t0 = time.time()
            ret = b.eval(sdir + "_" + syst, arr_flav, arr_abs_eta, arr_pt, arr_discr)
            t1 = time.time()
            print("eval_C", t1 - t0)
    
            t0 = time.time()
            tsys_name = sdir + '_' + syst
            ret2 = sf.eval(tsys_name, arr_flav, arr_abs_eta, arr_pt, arr_discr, True)
            t1 = time.time()

            print("eval_py", t1 - t0)
            k = 0
            for ijet in range(len(arr_pt)):
                if ret[ijet] != ret2[ijet]:
                    print("pt={:.4f} eta={:.4f} discr={:.4f} fl={} sf1={} sf2={}".format(
                        arr_pt[ijet], arr_abs_eta[ijet], arr_discr[ijet], arr_flav[ijet], ret[ijet], ret2[ijet])
                    )
                    k += 1
                if k > 100:
                    break
