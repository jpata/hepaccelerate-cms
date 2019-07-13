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

  void init(std::vector<std::string> files, std::vector<std::string> histos, std::vector<float> weights);
  void setLeptons(int nLep, int *lepPdgId, float *lepPt, float *lepEta);

  float getSF(int pdgid, float pt, float eta) const;
  float getSFErr(int pdgid, float pt, float eta) const;
  const std::vector<float> & run();

private:
  std::vector<TH2F*> effmaps_;
  std::vector<float> weights_;
  std::vector<float> ret_;
  int nLep_;
  float *Lep_eta_, *Lep_pt_;
  int *Lep_pdgId_;
};

extern "C" {
    LeptonEfficiencyCorrector* new_LeptonEfficiencyCorrector(int n, const char** file, const char** histo, float* weights) {
        std::vector<std::string> v_files;
        for (int i=0; i < n; i++) {
          v_files.push_back(file[i]);
        }
        std::vector<std::string> v_histos;
        for (int i=0; i < n; i++) {
          v_histos.push_back(histo[i]);
        }

        std::vector<float> v_weights;
        for (int i=0; i < n; i++) {
          v_weights.push_back(weights[i]);
        }

        auto* ret = new LeptonEfficiencyCorrector();
        ret->init(v_files, v_histos, v_weights);
        return ret;
    }

    void LeptonEfficiencyCorrector_getSF(LeptonEfficiencyCorrector* c, float* out, int n, int* pdgid, float* pt, float* eta) {
        #pragma omp parallel for
        for (int i=0; i<n; i++) {
            out[i] = c->getSF(pdgid[i], pt[i], eta[i]);
        }
    }
}
#endif
