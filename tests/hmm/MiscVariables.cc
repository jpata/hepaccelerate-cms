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
