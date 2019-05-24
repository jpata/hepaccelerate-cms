from cffi import FFI
import numpy as numpy_lib
import os

class LibHMuMu:
    def __init__(self, libpath=os.path.dirname(os.path.realpath(__file__)) + "/libhmm.so"):
        self.ffi = FFI()
        self.ffi.cdef("""
            void* new_roccor(char* filename);
            void roccor_kScaleDT(void* rc, float* out, int n_elem, int Q, float* pt, float* eta, float* phi, int s, int m);
            void roccor_kSpreadMC_or_kSmearMC(void* rc, float* out, int n_elem,
                int Q, float* pt, float* eta, float* phi,
                float* genpt, int* tracklayers, float* rand, int s, int m);

            void* new_LeptonEfficiencyCorrector(const char* file, const char* histo);
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

    def compute_kScaleDT(self, pts, etas, phis):
        out = numpy_lib.zeros_like(pts)
        self.libhmm.roccor_kScaleDT(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            1, #Q
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
            self.libhmm.cast_as("float *", phis),
            0, 0
        )
        return out

    def compute_kSpreadMC_or_kSmearMC(self, pts, etas, phis, genpts, tracklayers, rnds):
        out = numpy_lib.zeros_like(pts)
        self.libhmm.roccor_kSpreadMC_or_kSmearMC(self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            1, #Q
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
    def __init__(self, libhmm, filename, histoname):
        self.libhmm = libhmm
        self.c_class = self.libhmm.new_LeptonEfficiencyCorrector(
            filename.encode("ascii"), histoname.encode("ascii")
        )
            

    def compute(self, pdgids, pts, etas):
        out = numpy_lib.zeros_like(pts)
        self.libhmm.LeptonEfficiencyCorrector_getSF(
            self.c_class,
            self.libhmm.cast_as("float *", out), len(out), #output
            self.libhmm.cast_as("int *", pdgids),
            self.libhmm.cast_as("float *", pts),
            self.libhmm.cast_as("float *", etas),
        )
        return out
