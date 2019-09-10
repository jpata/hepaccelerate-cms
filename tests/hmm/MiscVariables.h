#include <TLorentzVector.h>
#include <TMath.h>

std::pair<double,double> CSAngles(TLorentzVector& v1, TLorentzVector& v2, int charge);

double mllErr(TLorentzVector& lep1, TLorentzVector& lep2, double ptErr1, double ptErr2);

extern "C" {
  void csangles_eval(
		     float* out_theta, float* out_phi, int nev,
		     float* pt1, float* eta1, float* phi1, float* mass1,
		     float* pt2, float* eta2, float* phi2, float* mass2,
		     int* charges) {
#pragma omp parallel for default(none) shared(out_theta, out_phi, nev, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, charges) schedule(dynamic, 1000)
    for (int iev=0; iev<nev; iev++) {
                TLorentzVector v1, v2;
                v1.SetPtEtaPhiM(pt1[iev], eta1[iev], phi1[iev], mass1[iev]);
                v2.SetPtEtaPhiM(pt2[iev], eta2[iev], phi2[iev], mass2[iev]);
                const auto ret = CSAngles(v1, v2, charges[iev]);
                out_theta[iev] = (float)(ret.first);
                out_phi[iev] = (float)(ret.second);
    }
  }
  void mllErr_eval(
		     float* out_Err, int nev,
		     float* pt1, float* eta1, float* phi1, float* mass1,
		     float* pt2, float* eta2, float* phi2, float* mass2,
		     float* pt1Err, float* pt2Err) {
#pragma omp parallel for default(none) shared(out_Err, nev, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, pt1Err, pt2Err) schedule(dynamic, 1000)
    for (int iev=0; iev<nev; iev++) {
                TLorentzVector v1, v2;
                v1.SetPtEtaPhiM(pt1[iev], eta1[iev], phi1[iev], mass1[iev]);
                v2.SetPtEtaPhiM(pt2[iev], eta2[iev], phi2[iev], mass2[iev]);
                const auto ret = mllErr(v1, v2, pt1Err[iev], pt2Err[iev]);   
                out_Err[iev] = (float)(ret);
    }
  }
}

//double mllErr(TLorentzVector& lep1, TLorentzVector& lep2, double ptErr1, double ptErr2);
/* double mllErr(double muPt1,  double muEta1, double muPhi1, double muMass1, */
/* 	      double muPt2,  double muEta2, double muPhi2, double muMass2, */
/* 	      double ptErr1, double ptErr2); */
