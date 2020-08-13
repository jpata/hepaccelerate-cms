from cffi import FFI
import numpy as numpy_lib
import os
import uproot

class LibHMuMu:
    def __init__(self, libpath=os.path.dirname(os.path.realpath(__file__)) + "/libhmm.so"):
        self.ffi = FFI()
        self.ffi.cdef("""
            void* new_roccor(char* filename);
            void roccor_kScaleDT(void* rc, float* out, int n_elem, int* charges, float* pt, float* eta, float* phi, int s, int m);
            void roccor_kScaleDTerror(void* rc, float* out, int n_elem, int* charges, float* pt, float* eta, float* phi);
            void roccor_kSpreadMC_or_kSmearMC(void* rc, float* out, int n_elem,
                int* charges, float* pt, float* eta, float* phi,
                float* genpt, int* tracklayers, float* rand, int s, int m);
            void roccor_kSpreadMCerror_or_kSmearMCerror(void* rc, float* out, int n_elem,                                                                                                                                             int* charges, float* pt, float* eta, float* phi,                                                                                                                                                           float* genpt, int* tracklayers, float* rand);

            void* new_LeptonEfficiencyCorrector(int n, const char** file, const char** histo, float* weights);
            void LeptonEfficiencyCorrector_getSF(void* c, float* out, int n, int* pdgid, float* pt, float* eta);
            void LeptonEfficiencyCorrector_getSFErr(void* c, float* out, int n, int* pdgid, float* pt, float* eta);

            const void* new_gbr(const char* weightfile);
            int gbr_get_nvariables(const void* gbr);
            void gbr_eval(const void* gbr, float* out, int nev, int nfeatures, float* inputs_matrix);


            void csangles_eval(float* out_theta, float* out_phi, int nev, float* pt1, float* eta1, float* phi1, float* mass1, float* pt2, float* eta2, float* phi2, float* mass2, int* charges);

            void csanglesPisa_eval(float* out_theta, float* out_phi, int nev, float* pt1, float* eta1, float* phi1, float* mass1, float* pt2, float* eta2, float* phi2, float* mass2, int* charges);

            void ptcorrgeofit_eval(float* out_pt, int nev, float* d0_BS, float* pt_Roch, float* eta, int* charge, int* year);
            void qglJetWeight_eval(float* out_weight, int nev, int* partonFlavour, float* pt, float* eta, float* qgl, int isHerwig);

            void* new_NNLOPSReweighting(const char* path);
            void NNLOPSReweighting_eval(void* c, int igen, float* out_nnlow, int nev, int* genNjets, float* genHiggs_pt);

            void* new_hRelResolution(const char* path);
            void hRelResolution_eval(void* c, float* out_hres, int nev, float* mu1_pt, float* mu1_eta, float* mu2_pt, float* mu2_eta);
            void* new_ZpTReweighting();
            void ZpTReweighting_eval(void* c, float* out_zptw, int nev, float* pt, int itune);
            
            void* new_BTagCalibration(const char* tagger);
            void BTagCalibration_readCSV(void* obj, const char* file_path);
            void* new_BTagCalibrationReader(int op, const char* syst, int num_other_systs, const char** other_systs);
            void BTagCalibrationReader_load(void* obj, void* obj2, int flav, const char* type);
            void BTagCalibrationReader_eval(
                void* calib_b,
                void* calib_c,
                void* calib_l,
                float* out_w, int nev, const char* sys,
                int* flav, float* abs_eta, float* pt, float* discr);

        """)
        self.libhmm = self.ffi.dlopen(libpath)

        self.new_roccor = self.libhmm.new_roccor
        self.roccor_kScaleDT = self.libhmm.roccor_kScaleDT
        self.roccor_kScaleDTerror = self.libhmm.roccor_kScaleDTerror
        self.roccor_kSpreadMC_or_kSmearMC = self.libhmm.roccor_kSpreadMC_or_kSmearMC
        self.roccor_kSpreadMCerror_or_kSmearMCerror = self.libhmm.roccor_kSpreadMCerror_or_kSmearMCerror

        self.new_LeptonEfficiencyCorrector = self.libhmm.new_LeptonEfficiencyCorrector
        self.LeptonEfficiencyCorrector_getSF = self.libhmm.LeptonEfficiencyCorrector_getSF
        self.LeptonEfficiencyCorrector_getSFErr = self.libhmm.LeptonEfficiencyCorrector_getSFErr

        self.new_gbr = self.libhmm.new_gbr
        self.gbr_get_nvariables = self.libhmm.gbr_get_nvariables
        self.gbr_eval = self.libhmm.gbr_eval

        self.csangles_eval = self.libhmm.csangles_eval
        self.csanglesPisa_eval = self.libhmm.csanglesPisa_eval
        self.ptcorrgeofit_eval = self.libhmm.ptcorrgeofit_eval
        self.qglJetWeight_eval = self.libhmm.qglJetWeight_eval
        
        self.new_NNLOPSReweighting = self.libhmm.new_NNLOPSReweighting
        self.NNLOPSReweighting_eval = self.libhmm.NNLOPSReweighting_eval

        self.new_hRelResolution = self.libhmm.new_hRelResolution
        self.hRelResolution_eval = self.libhmm.hRelResolution_eval

        self.new_ZpTReweighting = self.libhmm.new_ZpTReweighting
        self.ZpTReweighting_eval = self.libhmm.ZpTReweighting_eval
        
        self.new_BTagCalibration = self.libhmm.new_BTagCalibration
        self.BTagCalibration_readCSV = self.libhmm.BTagCalibration_readCSV
        self.new_BTagCalibrationReader = self.libhmm.new_BTagCalibrationReader
        self.BTagCalibrationReader_load = self.libhmm.BTagCalibrationReader_load
        self.BTagCalibrationReader_eval = self.libhmm.BTagCalibrationReader_eval
 
    def cast_as(self, dtype_string, arr):
        return self.ffi.cast(dtype_string, arr.ctypes.data)


class RochesterCorrections:
    def __init__(self, libhmm, filename):
        self.libhmm = libhmm
        self.c_class = self.libhmm.new_roccor(filename.encode('ascii'))

    def compute_kScaleDT(self, pts, etas, phis, charges):
        out = numpy_lib.zeros_like(pts)
        nev = len(pts)
        assert(len(etas) == nev)
        assert(len(phis) == nev)
        assert(len(charges) == nev)
        self.libhmm.roccor_kScaleDT(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            self.libhmm.cast_as("int *", charges),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
            self.libhmm.cast_as("float *", phis),
            0, 0
        )
        return out

    def compute_kScaleDTerror(self, pts, etas, phis, charges):
        out = numpy_lib.zeros_like(pts)
        nev = len(pts)
        assert(len(etas) == nev)
        assert(len(phis) == nev)
        assert(len(charges) == nev)
        self.libhmm.roccor_kScaleDTerror(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output                                                                                                                                          
            self.libhmm.cast_as("int *", charges),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
            self.libhmm.cast_as("float *", phis),
        )
        return out

    def compute_kSpreadMC_or_kSmearMC(self, pts, etas, phis, charges, genpts, tracklayers, rnds):
        out = numpy_lib.zeros_like(pts)
        nev = len(pts)
        assert(len(etas) == nev)
        assert(len(phis) == nev)
        assert(len(charges) == nev)
        assert(len(genpts) == nev)
        assert(len(tracklayers) == nev)
        assert(len(rnds) == nev)
        self.libhmm.roccor_kSpreadMC_or_kSmearMC(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            self.libhmm.cast_as("int *", charges),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
            self.libhmm.cast_as("float *", phis),
            self.libhmm.cast_as("float *", genpts),
            self.libhmm.cast_as("int *", tracklayers),
            self.libhmm.cast_as("float *", rnds),
            0, 0
        )
        return out

    def compute_kSpreadMCerror_or_kSmearMCerror(self, pts, etas, phis, charges, genpts, tracklayers, rnds):
        out = numpy_lib.zeros_like(pts)
        nev = len(pts)
        assert(len(etas) == nev)
        assert(len(phis) == nev)
        assert(len(charges) == nev)
        assert(len(genpts) == nev)
        assert(len(tracklayers) == nev)
        assert(len(rnds) == nev)
        self.libhmm.roccor_kSpreadMCerror_or_kSmearMCerror(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output                                                                                                                                          
            self.libhmm.cast_as("int *", charges),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
            self.libhmm.cast_as("float *", phis),
            self.libhmm.cast_as("float *", genpts),
            self.libhmm.cast_as("int *", tracklayers),
            self.libhmm.cast_as("float *", rnds),
        )
        return out

class LeptonEfficiencyCorrections:
    def __init__(self, libhmm, filenames, histonames, weights):
        self.libhmm = libhmm
        for fn, hn in zip(filenames, histonames):
            if not os.path.isfile(fn):
                raise FileNotFoundError("File {0} does not exist".format(fn))
            fi = uproot.open(fn)
            if not hn in fi:
                raise KeyError("Histogram {0} does not exist in file {1}".format(hn, fn))
 
        filenames_C = [libhmm.ffi.new("char[]", fn.encode("ascii")) for fn in filenames]
        histonames_C = [libhmm.ffi.new("char[]", hn.encode("ascii")) for hn in histonames]
        self.c_class = self.libhmm.new_LeptonEfficiencyCorrector(
            len(filenames), filenames_C, histonames_C, weights
        )
            

    def compute(self, pdgids, pts, etas):
        out = numpy_lib.zeros_like(pts)
        self.libhmm.LeptonEfficiencyCorrector_getSF(
            self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            self.libhmm.cast_as("int *", pdgids),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas))
        return out
    
    def compute_error(self, pdgids, pts, etas):
        out = numpy_lib.zeros_like(pts)
        self.libhmm.LeptonEfficiencyCorrector_getSFErr(
            self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            self.libhmm.cast_as("int *", pdgids),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas))
        return out

class GBREvaluator:
    def __init__(self, libhmm, weightfile):
        self.libhmm = libhmm
        self.c_class = self.libhmm.new_gbr(libhmm.ffi.new("char[]", weightfile.encode("ascii")))

    def compute(self, features):
        nev = features.shape[0]
        nfeat = features.shape[1]
        out = numpy_lib.zeros(nev, dtype=numpy_lib.float32)

        self.libhmm.gbr_eval(
            self.c_class,
            self.libhmm.cast_as("float *", out),
            nev, nfeat,
            self.libhmm.cast_as("float *", features.ravel(order='C'))
        )
        return out

    def get_bdt_nfeatures(self):
        return self.libhmm.gbr_get_nvariables(self.c_class)

class NNLOPSReweighting:
    def __init__(self, libhmm, path):
        self.libhmm = libhmm
        if not os.path.isfile(path):
            raise FileNotFoundError("File {0} does not exist".format(path))
        fi = uproot.open(path)
      
        file_C = self.libhmm.ffi.new("char[]", path.encode("ascii")) 
        self.c_class = self.libhmm.new_NNLOPSReweighting(
            file_C
        )

    def compute(self, genNjets, genHiggs_pt, igen):
        out_nnlow = numpy_lib.ones_like(genNjets, dtype=numpy_lib.float32)
        nev = len(genNjets)
        assert(len(genHiggs_pt) == nev)
        self.libhmm.NNLOPSReweighting_eval(
            self.c_class,
            igen,
            self.libhmm.cast_as("float *", out_nnlow), 
            len(out_nnlow),
            self.libhmm.cast_as("int *", genNjets), 
            self.libhmm.cast_as("float *", genHiggs_pt),  
        )
        return out_nnlow

class hRelResolution:
    def __init__(self, libhmm, path):
        self.libhmm = libhmm
        if not os.path.isfile(path):
            raise FileNotFoundError("File {0} does not exist".format(path))
        fi = uproot.open(path)
        print(path)
        file_C = self.libhmm.ffi.new("char[]", path.encode("ascii"))
        self.c_class = self.libhmm.new_hRelResolution(
            file_C
        )

    def compute(self, mu1_pt, mu1_eta, mu2_pt, mu2_eta):
        out_hres = numpy_lib.ones_like(mu1_pt, dtype=numpy_lib.float32)
        nev = len(mu1_pt)
        assert(len(mu1_eta) == nev)
        assert(len(mu2_pt) == nev)
        assert(len(mu2_eta) == nev)
        self.libhmm.hRelResolution_eval(
            self.c_class,
            self.libhmm.cast_as("float *", out_hres),
            len(out_hres),
            self.libhmm.cast_as("float *", mu1_pt),
            self.libhmm.cast_as("float *", mu1_eta),
            self.libhmm.cast_as("float *", mu2_pt),
            self.libhmm.cast_as("float *", mu2_eta),
        )
        return out_hres

class ZpTReweighting:
    def __init__(self, libhmm):
        self.libhmm = libhmm
        self.c_class = self.libhmm.new_ZpTReweighting()

    def compute(self, pt, itune):
        out_zptw = numpy_lib.ones_like(pt, dtype=numpy_lib.float32)
        self.libhmm.ZpTReweighting_eval(
            self.c_class,
            self.libhmm.cast_as("float *", out_zptw),
            len(out_zptw),
            self.libhmm.cast_as("float *", pt),
            itune,
        )
        return out_zptw

class MiscVariables:
    def __init__(self, libhmm):
        self.libhmm = libhmm

    def csangles(self, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, charges):
        nev = len(pt1)
        assert(len(eta1) == nev)
        assert(len(phi1) == nev)
        assert(len(mass1) == nev)
        assert(len(pt2) == nev)
        assert(len(eta2) == nev)
        assert(len(phi2) == nev)
        assert(len(mass2) == nev)
        assert(len(charges) == nev)
        out_theta = numpy_lib.zeros(nev, dtype=numpy_lib.float32)
        out_phi = numpy_lib.zeros(nev, dtype=numpy_lib.float32)

        self.libhmm.csangles_eval(
            self.libhmm.cast_as("float *", out_theta),
            self.libhmm.cast_as("float *", out_phi),
            nev,
            self.libhmm.cast_as("float *", pt1),
            self.libhmm.cast_as("float *", eta1),
            self.libhmm.cast_as("float *", phi1),
            self.libhmm.cast_as("float *", mass1),
            self.libhmm.cast_as("float *", pt2),
            self.libhmm.cast_as("float *", eta2),
            self.libhmm.cast_as("float *", phi2),
            self.libhmm.cast_as("float *", mass2),
            self.libhmm.cast_as("int *", charges),
        )
        return out_theta, out_phi

    def csanglesPisa(self, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, charges):
        nev = len(pt1)
        assert(len(eta1) == nev)
        assert(len(phi1) == nev)
        assert(len(mass1) == nev)
        assert(len(pt2) == nev)
        assert(len(eta2) == nev)
        assert(len(phi2) == nev)
        assert(len(mass2) == nev)
        assert(len(charges) == nev)
        out_theta = numpy_lib.zeros(nev, dtype=numpy_lib.float32)
        out_phi = numpy_lib.zeros(nev, dtype=numpy_lib.float32)

        self.libhmm.csanglesPisa_eval(
            self.libhmm.cast_as("float *", out_theta),
            self.libhmm.cast_as("float *", out_phi),
            nev,
            self.libhmm.cast_as("float *", pt1),
            self.libhmm.cast_as("float *", eta1),
            self.libhmm.cast_as("float *", phi1),
            self.libhmm.cast_as("float *", mass1),
            self.libhmm.cast_as("float *", pt2),
            self.libhmm.cast_as("float *", eta2),
            self.libhmm.cast_as("float *", phi2),
            self.libhmm.cast_as("float *", mass2),
            self.libhmm.cast_as("int *", charges),
        )
        return out_theta, out_phi
    
    def ptcorrgeofit(self, d0_BS, pt_Roch, eta, charge, year):
        nev = len(pt_Roch)
        out_pt = numpy_lib.zeros(nev, dtype=numpy_lib.float32)
        assert(len(d0_BS) == nev)
        assert(len(eta) == nev)
        assert(len(charge) == nev)
        assert(len(year) == nev)
        self.libhmm.ptcorrgeofit_eval(
            self.libhmm.cast_as("float *", out_pt),
            nev,
            self.libhmm.cast_as("float *", d0_BS),
            self.libhmm.cast_as("float *", pt_Roch),
            self.libhmm.cast_as("float *", eta),
            self.libhmm.cast_as("int *", charge),
            self.libhmm.cast_as("int *", year),
        )
        return out_pt
    
    def qglJetWeight(self, partonFlavour, pt, eta, qgl, isHerwig):
        nev = len(partonFlavour)
        out_weight = numpy_lib.zeros(nev, dtype=numpy_lib.float32)
        assert(len(partonFlavour) == nev)
        assert(len(eta) == nev)
        assert(len(qgl) == nev)
        self.libhmm.qglJetWeight_eval(
            self.libhmm.cast_as("float *", out_weight),
            nev,
            self.libhmm.cast_as("int *", partonFlavour),
            self.libhmm.cast_as("float *", pt),
            self.libhmm.cast_as("float *", eta),
            self.libhmm.cast_as("float *", qgl),
            isHerwig,
        )
        return out_weight

class BTagCalibration:
    def __init__(self, libhmm, tagger, csv_file, systs=[]):
        self.libhmm = libhmm
        tagger_C = self.libhmm.ffi.new("char[]", tagger.encode("ascii"))
        self.c_class = self.libhmm.new_BTagCalibration(tagger_C)
        file_C = self.libhmm.ffi.new("char[]", csv_file.encode("ascii"))
        self.libhmm.BTagCalibration_readCSV(self.c_class, file_C)

        self.calib_b = BTagCalibrationReader(self.libhmm, 3, "central", systs)
        self.calib_c = BTagCalibrationReader(self.libhmm, 3, "central", systs)
        self.calib_l = BTagCalibrationReader(self.libhmm, 3, "central", systs)
        self.calib_b.load(self, 0)
        self.calib_c.load(self, 1)
        self.calib_l.load(self, 2)

    def eval(self, sys_name, arr_flav, arr_abs_eta, arr_pt, arr_discr):
        sys_C = self.libhmm.ffi.new("char[]", sys_name.encode("ascii"))
        out = numpy_lib.zeros_like(arr_abs_eta)

        nev = len(arr_flav)
        assert(arr_flav.dtype == numpy_lib.int32)
        assert(arr_abs_eta.dtype == numpy_lib.float32)
        assert(arr_pt.dtype == numpy_lib.float32)
        assert(arr_discr.dtype == numpy_lib.float32)
        assert(len(arr_abs_eta) == nev)
        assert(len(arr_pt) == nev)
        assert(len(arr_discr) == nev)

        self.libhmm.BTagCalibrationReader_eval(
            self.calib_b.c_class,
            self.calib_c.c_class,
            self.calib_l.c_class,
            self.libhmm.cast_as("float *", out),
            nev, sys_C, 
            self.libhmm.cast_as("int *", arr_flav),
            self.libhmm.cast_as("float *", arr_abs_eta),
            self.libhmm.cast_as("float *", arr_pt),
            self.libhmm.cast_as("float *", arr_discr)
        )
        return out

class BTagCalibrationReader:
    def __init__(self, libhmm, op, syst, other_systs):
        self.libhmm = libhmm

        syst_C = self.libhmm.ffi.new("char[]", syst.encode("ascii"))
        other_systs_C = [self.libhmm.ffi.new("char[]", s.encode("ascii")) for s in other_systs]
        self.c_class = self.libhmm.new_BTagCalibrationReader(op, syst_C, len(other_systs_C), other_systs_C)

    def load(self, calib, flav, typ="iterativefit"):
        typ_C = self.libhmm.ffi.new("char[]", typ.encode("ascii"))
        self.libhmm.BTagCalibrationReader_load(self.c_class, calib.c_class, flav, typ_C)
