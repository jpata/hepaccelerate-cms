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

        """)
        self.libhmm = self.ffi.dlopen(libpath)

        self.new_roccor = self.libhmm.new_roccor
        self.roccor_kScaleDT = self.libhmm.roccor_kScaleDT
        self.roccor_kSpreadMC_or_kSmearMC = self.libhmm.roccor_kSpreadMC_or_kSmearMC

        self.new_LeptonEfficiencyCorrector = self.libhmm.new_LeptonEfficiencyCorrector
        self.LeptonEfficiencyCorrector_getSF = self.libhmm.LeptonEfficiencyCorrector_getSF

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
