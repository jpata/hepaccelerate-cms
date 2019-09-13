#ifndef hRelResolution_h
#define hRelResolution_h

#include <iostream>
#include <TMath.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2F.h>

class hRelResolution {
 public:

  hRelResolution() {}
  ~hRelResolution() {}
  void init(const char* path);
  float hres(float pt1, float eta1, float pt2, float eta2) const;

 private:
  TH2F* hmuon;

};

extern "C" {
    hRelResolution* new_hRelResolution(const char* path){
        auto* ret = new hRelResolution();
        ret->init(path);
        return ret;
    }
    void hRelResolution_eval(
        hRelResolution* c,
        float* out_hres, int nev,
        float* mu1_pt, float* mu1_eta, float* mu2_pt, float* mu2_eta) {
          #pragma omp parallel for default(none) shared(c, out_hres, nev, mu1_pt, mu1_eta, mu2_pt, mu2_eta) schedule(dynamic, 1000)
          for (int iev=0; iev<nev; iev++) {
            const auto ret = c->hres(mu1_pt[iev], mu1_eta[iev], mu2_pt[iev], mu2_eta[iev]);
            out_hres[iev] = (float)ret;
        }
    }
}

#endif
