#include "MiscVariables.h"

//https://github.com/alisw/AliPhysics/blob/master/PWGDQ/dielectron/core/AliDielectronPair.cxx
std::pair<double,double> CSAngles(TLorentzVector& v1, TLorentzVector& v2, int charge) {
    const float fBeamEnergy = 6500.0;
    const float proMass = 0.938272;
    TLorentzVector pro1(0.,0.,-fBeamEnergy,TMath::Sqrt(fBeamEnergy*fBeamEnergy+proMass*proMass));
    TLorentzVector pro2(0.,0., fBeamEnergy,TMath::Sqrt(fBeamEnergy*fBeamEnergy+proMass*proMass));
    
    TLorentzVector H = v1 + v2;
    v1.Boost( -( H.BoostVector() ) ); // go to Higgs RFR
    v2.Boost( -( H.BoostVector() ) );
    pro1.Boost( -( H.BoostVector() ) );
    pro2.Boost( -( H.BoostVector() ) );
    
    TVector3 yAxis = ((pro1.Vect()).Cross(pro2.Vect())).Unit();
    TVector3 zAxisCS = ((pro1.Vect()).Unit()-(pro2.Vect()).Unit()).Unit();
    TVector3 xAxisCS = (yAxis.Cross(zAxisCS)).Unit();
    
    TVector3 reco_M1 = v1.Vect().Unit();
    TVector3 reco_M2 = v2.Vect().Unit();
    double thetaCS = zAxisCS.Dot((v1.Vect()).Unit());
    double phiCS   = TMath::ATan2((v1.Vect()).Dot(yAxis), (v1.Vect()).Dot(xAxisCS));
    if(charge>0){
        thetaCS = zAxisCS.Dot((v2.Vect()).Unit());
        phiCS   = TMath::ATan2((v2.Vect()).Dot(yAxis), (v2.Vect()).Dot(xAxisCS));
    }
    return std::pair<double,double>(-thetaCS,-phiCS);
}

//simple uncertainty propagation to get di-muon invariant mass uncertainty, given ptErr on the two muons
double mllErr(TLorentzVector& lep1, TLorentzVector& lep2, double ptErr1, double ptErr2){
// double mllErr(double muPt1,  double muEta1, double muPhi1, double muMass1,
// 	      double muPt2,  double muEta2, double muPhi2, double muMass2,
// 	      double ptErr1, double ptErr2){

  // TLorentzVector lep1; 
  // TLorentzVector lep2;

  // lep1.SetPtEtaPhiM(muPt1, muEta1, muPhi1, muMass1);
  // lep2.SetPtEtaPhiM(muPt2, muEta2, muPhi2, muMass2);

  // just for simplicity
  double pt1 = lep1.Pt();
  double pt2 = lep2.Pt();

  // event is OK if both muons have pt > 0
  if (pt1 > 0. && pt2 > 0.){

    double mass = (lep1+lep2).M();
    double dpt1 = (ptErr1 * mass) / (2 * pt1);
    double dpt2 = (ptErr2 * mass) / (2 * pt2);
    double err  = sqrt(dpt1*dpt1 + dpt2*dpt2);

    return err;
  }
  else {
    return -9999.0;
  }

}
