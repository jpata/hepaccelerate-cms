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
            void roccor_kSpreadMC_or_kSmearMC(void* rc, float* out, int n_elem,
                int* charges, float* pt, float* eta, float* phi,
                float* genpt, int* tracklayers, float* rand, int s, int m);

            void* new_LeptonEfficiencyCorrector(int n, const char** file, const char** histo, float* weights);
            void LeptonEfficiencyCorrector_getSF(void* c, float* out, int n, int* pdgid, float* pt, float* eta);
            void LeptonEfficiencyCorrector_getSFErr(void* c, float* out, int n, int* pdgid, float* pt, float* eta);

            const void* new_gbr(const char* weightfile);
            int gbr_get_nvariables(const void* gbr);
            void gbr_eval(const void* gbr, float* out, int nev, int nfeatures, float* inputs_matrix);


            void csangles_eval(float* out_theta, float* out_phi, int nev, float* pt1, float* eta1, float* phi1, float* mass1, float* pt2, float* eta2, float* phi2, float* mass2, int* charges);

            void csanglesPisa_eval(float* out_theta, float* out_phi, int nev, float* pt1, float* eta1, float* phi1, float* mass1, float* pt2, float* eta2, float* phi2, float* mass2, int* charges);

            void* new_NNLOPSReweighting(const char* path);
            void NNLOPSReweighting_eval(void* c, int igen, float* out_nnlow, int nev, int* genNjets, float* genHiggs_pt);

            void* new_hRelResolution(const char* path);
            void hRelResolution_eval(void* c, float* out_hres, int nev, float* mu1_pt, float* mu1_eta, float* mu2_pt, float* mu2_eta);
            void* new_ZpTReweighting();
            void ZpTReweighting_eval(void* c, float* out_zptw, int nev, float* pt, int itune);
        """)
        self.libhmm = self.ffi.dlopen(libpath)

        self.new_roccor = self.libhmm.new_roccor
        self.roccor_kScaleDT = self.libhmm.roccor_kScaleDT
        self.roccor_kSpreadMC_or_kSmearMC = self.libhmm.roccor_kSpreadMC_or_kSmearMC

        self.new_LeptonEfficiencyCorrector = self.libhmm.new_LeptonEfficiencyCorrector
        self.LeptonEfficiencyCorrector_getSF = self.libhmm.LeptonEfficiencyCorrector_getSF
        self.LeptonEfficiencyCorrector_getSFErr = self.libhmm.LeptonEfficiencyCorrector_getSFErr

        self.new_gbr = self.libhmm.new_gbr
        self.gbr_get_nvariables = self.libhmm.gbr_get_nvariables
        self.gbr_eval = self.libhmm.gbr_eval

        self.csangles_eval = self.libhmm.csangles_eval
        self.csanglesPisa_eval = self.libhmm.csanglesPisa_eval
        
        self.new_NNLOPSReweighting = self.libhmm.new_NNLOPSReweighting
        self.NNLOPSReweighting_eval = self.libhmm.NNLOPSReweighting_eval

        self.new_hRelResolution = self.libhmm.new_hRelResolution
        self.hRelResolution_eval = self.libhmm.hRelResolution_eval

        self.new_ZpTReweighting = self.libhmm.new_ZpTReweighting
        self.ZpTReweighting_eval = self.libhmm.ZpTReweighting_eval

    def cast_as(self, dtype_string, arr):
        return self.ffi.cast(dtype_string, arr.ctypes.data)


class RochesterCorrections:
    def __init__(self, libhmm, filename):
        self.libhmm = libhmm
        self.c_class = self.libhmm.new_roccor(filename.encode('ascii'))

    def compute_kScaleDT(self, pts, etas, phis, charges):
        out = numpy_lib.zeros_like(pts)
        self.libhmm.roccor_kScaleDT(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            self.libhmm.cast_as("int *", charges),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
            self.libhmm.cast_as("float *", phis),
            0, 0
        )
        return out

    def compute_kSpreadMC_or_kSmearMC(self, pts, etas, phis, charges, genpts, tracklayers, rnds):
        out = numpy_lib.zeros_like(pts)
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
      
        file_C = libhmm.ffi.new("char[]", path.encode("ascii")) 
        self.c_class = self.libhmm.new_NNLOPSReweighting(
            file_C
        )

    def compute(self, genNjets, genHiggs_pt, igen):
        out_nnlow = numpy_lib.ones_like(genNjets, dtype=numpy_lib.float32)
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
        file_C = libhmm.ffi.new("char[]", path.encode("ascii"))
        self.c_class = self.libhmm.new_hRelResolution(
            file_C
        )

    def compute(self, mu1_pt, mu1_eta, mu2_pt, mu2_eta):
        out_hres = numpy_lib.ones_like(mu1_pt, dtype=numpy_lib.float32)
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
