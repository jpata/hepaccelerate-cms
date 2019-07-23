#include "LeptonEfficiencyCorrector.h"

void LeptonEfficiencyCorrector::init(std::vector<std::string> files, std::vector<std::string> histos, std::vector<float> weights) {
  effmaps_.clear();
  assert(files.size()==histos.size());
  assert(weights.size() == files.size());

  for(int i=0; i<(int)files.size();++i) {
    TFile *f = TFile::Open(files[i].c_str(),"read");
    if(!f) {
      std::cout << "WARNING! File " << files[i] << " cannot be opened. Skipping this scale factor " << std::endl;
      continue;
    }
    TH2F *hist = (TH2F*)(f->Get(histos[i].c_str()))->Clone(("eff_"+histos[i]).c_str());
    hist->SetDirectory(0);
    if(!hist) {
      std::cout << "ERROR! Histogram " << histos[i] << " not in file " << files[i] << ". Not considering this SF. " << std::endl;
      continue;
    } else {
      std::cout << "Loading histogram " << histos[i] << " from file " << files[i] << "... " << std::endl;
    }
    effmaps_.push_back(hist);
    WeightCalculatorFromHistogram wc(hist);
    weightcalc_.push_back(wc);
    weights_.push_back(weights[i]);
    f->Close();
  }
}

void LeptonEfficiencyCorrector::setLeptons(int nLep, int *lepPdgId, float *lepPt, float *lepEta) {
  nLep_ = nLep; Lep_pdgId_ = lepPdgId; Lep_pt_ = lepPt; Lep_eta_ = lepEta;
}

float LeptonEfficiencyCorrector::getSF(int pdgid, float pt, float eta) const {
  float out=0.0;
  const float x = abs(pdgid)==13 ? pt : eta;
  const float y = abs(pdgid)==13 ? fabs(eta) : pt;
  int i = 0;
  for(const auto& wc : weightcalc_) {
    out += weights_[i] * wc.getWeight(x,y);
    i++;
  }
  return out;
}

float LeptonEfficiencyCorrector::getSFErr(int pdgid, float pt, float eta) const {
  float out=0.0;
  const float x = abs(pdgid)==13 ? pt : eta;
  const float y = abs(pdgid)==13 ? fabs(eta) : pt;
  int i = 0;
  for(const auto& wc : weightcalc_) {
    //std::cout << x << " " << y << " " << weights_[i] << " " << wc.getWeightErr(x,y) << std::endl; 
    out += weights_[i] * wc.getWeightErr(x,y);
    i++;
  }
  return out;
}

const std::vector<float> & LeptonEfficiencyCorrector::run() {
  ret_.clear();
  for (int iL = 0, nL = nLep_; iL < nL; ++iL) {
    ret_.push_back(getSF((Lep_pdgId_)[iL], (Lep_pt_)[iL], (Lep_eta_)[iL]));
  }
  return ret_;
}
