#ifndef PhysicsTools_NanoAODTools_LeptonEfficiencyCorrector_h
#define PhysicsTools_NanoAODTools_LeptonEfficiencyCorrector_h

#include <iostream>
#include <string>
#include <vector>
#include <TH2.h>
#include <TFile.h>

#include "WeightCalculatorFromHistogram.h"

class LeptonEfficiencyCorrector {
 public:

  LeptonEfficiencyCorrector() {effmaps_.clear();}
  //LeptonEfficiencyCorrector(std::vector<std::string> files, std::vector<std::string> histos);
  ~LeptonEfficiencyCorrector() {}

  void init(std::vector<std::string> files, std::vector<std::string> histos);
  void setLeptons(int nLep, int *lepPdgId, float *lepPt, float *lepEta);

  float getSF(int pdgid, float pt, float eta);
  float getSFErr(int pdgid, float pt, float eta);
  const std::vector<float> & run();

private:
  std::vector<TH2F*> effmaps_;
  std::vector<float> ret_;
  int nLep_;
  float *Lep_eta_, *Lep_pt_;
  int *Lep_pdgId_;
};

extern "C" {
    LeptonEfficiencyCorrector* new_LeptonEfficiencyCorrector(const char* file, const char* histo) {
        std::vector<std::string> v_files;
        v_files.push_back(file);
        std::vector<std::string> v_histos;
        v_histos.push_back(histo);

        auto* ret = new LeptonEfficiencyCorrector();
        ret->init(v_files, v_histos);
        return ret;
    }

    void LeptonEfficiencyCorrector_getSF(LeptonEfficiencyCorrector* c, float* out, int n, int* pdgid, float* pt, float* eta) {
        for (int i=0; i<n; i++) {
            out[i] = c->getSF(pdgid[i], pt[i], eta[i]);
        }
    }
}
#endif
