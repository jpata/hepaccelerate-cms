#ifndef NNLOPSReweighting_h
#define NNLOPSReweighting_h

#include <TLorentzVector.h>
#include <TString.h>
#include <TMath.h>
#include <TGraphErrors.h>
#include <TFile.h>

class NNLOPSReweighting {
 public:

  NNLOPSReweighting() {}
  ~NNLOPSReweighting() {}
  void init(const char* path);
  double nnlopsw(int gen_njets, double gen_Higgs_pT, int igen) const;

 private:
  TGraphErrors* gr_NNLOPSratio_pt_mcatnlo_0jet;
  TGraphErrors* gr_NNLOPSratio_pt_mcatnlo_1jet;
  TGraphErrors* gr_NNLOPSratio_pt_mcatnlo_2jet;
  TGraphErrors* gr_NNLOPSratio_pt_mcatnlo_3jet;
  
  TGraphErrors* gr_NNLOPSratio_pt_powheg_0jet;
  TGraphErrors* gr_NNLOPSratio_pt_powheg_1jet;
  TGraphErrors* gr_NNLOPSratio_pt_powheg_2jet;
  TGraphErrors* gr_NNLOPSratio_pt_powheg_3jet;
};

extern "C" {
    NNLOPSReweighting* new_NNLOPSReweighting(const char* path){
        auto* ret = new NNLOPSReweighting();
        ret->init(path);
        return ret;
    }
    void NNLOPSReweighting_eval(
        NNLOPSReweighting* c,
        int igen,
        float* out_nnlow, int nev,
        int* genNjets, float* genHiggs_pt) {
          #pragma omp parallel for default(none) shared(c, igen, out_nnlow, nev, genNjets, genHiggs_pt) schedule(dynamic, 1000)
          for (int iev=0; iev<nev; iev++) {
            const auto ret = c->nnlopsw(genNjets[iev], (double)genHiggs_pt[iev], igen);
            out_nnlow[iev] = (float)ret;
        }
    }
}

#endif
