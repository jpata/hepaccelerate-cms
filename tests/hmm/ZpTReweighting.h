#ifndef ZpTReweighting_h
#define ZpTReweighting_h

#include <iostream>
#include <TMath.h>

class ZpTReweighting {
 public:

  ZpTReweighting() {}
  ~ZpTReweighting() {}
  double zptw(double pT, int itune) const;
};

extern "C" {
    ZpTReweighting* new_ZpTReweighting(){
        auto* ret = new ZpTReweighting();
        return ret;
    }
    void ZpTReweighting_eval(
        ZpTReweighting* c,
        float* out_zptw, int nev,
        float* pt, int itune) {
          #pragma omp parallel for default(none) shared(c, out_zptw, nev, pt, itune) schedule(dynamic, 1000)
          for (int iev=0; iev<nev; iev++) {
            const auto ret = c->zptw((double)pt[iev],itune);
            out_zptw[iev] = (float)ret;
        }
    }
}

#endif
