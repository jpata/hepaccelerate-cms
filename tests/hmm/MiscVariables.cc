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
    double thetaCS = 0.;
    double phiCS   = 0.;
    if(charge>0){
        thetaCS = zAxisCS.Dot((v2.Vect()).Unit());
        phiCS   = TMath::ATan2((v2.Vect()).Dot(yAxis), (v2.Vect()).Dot(xAxisCS));
    }
    else{
        thetaCS = zAxisCS.Dot((v1.Vect()).Unit());
        phiCS   = TMath::ATan2((v1.Vect()).Dot(yAxis), (v1.Vect()).Dot(xAxisCS));
    }
    return std::pair<double,double>(-thetaCS,-phiCS);
}

//https://github.com/arizzi/PisaHmm/blob/master/boost_to_CS.h
std::pair<double,double> CSAnglesPisa(TLorentzVector& mu_Plus, TLorentzVector& mu_Minus, int mucharge) {
        TLorentzVector muplus  = TLorentzVector( mu_Plus.Px(),  mu_Plus.Py(),  mu_Plus.Pz(),  mu_Plus.E());
        TLorentzVector muminus = TLorentzVector( mu_Minus.Px(), mu_Minus.Py(), mu_Minus.Pz(), mu_Minus.E());

        
        TLorentzVector Wv= muplus+muminus;// this is the Z boson 4vector

        float multiplier=mucharge; //Wv.Eta()*mucharge;//same as W charge

        TVector3 b = Wv.BoostVector();
        muplus.Boost(-b);
        muminus.Boost(-b);

        TLorentzVector PF = TLorentzVector(0,0,-6500,6500); 
        TLorentzVector PW = TLorentzVector(0,0,6500,6500); 

        PF.Boost(-b);
        PW.Boost(-b);
        bool PFMinus= true;
        // choose what to call proton and what anti-proto
        if(Wv.Angle(PF.Vect())<Wv.Angle(PW.Vect()))
        {
                PW= -multiplier*PW;
                PF= multiplier*PF;
        }
        else
        {
                PF= -multiplier*PF;
                PW= multiplier*PW;
                PFMinus=false;
        }
        PF=PF*(1.0/PF.Vect().Mag());
        PW=PW*(1.0/PW.Vect().Mag());

        // Bisector is the new Z axis
        TLorentzVector PBiSec =PW+PF;
        TVector3 PhiSecZ =  PBiSec.Vect().Unit();

        TVector3 PhiSecY = (PhiSecZ.Cross(Wv.Vect().Unit())).Unit();

        TVector3 muminusVec = muminus.Vect();
        TRotation roataeMe;

        // build matrix for transformation into CS frame
        roataeMe.RotateAxes(PhiSecY.Cross(PhiSecZ),PhiSecY,PhiSecZ);
        roataeMe.Invert();
        // tranfor into CS alos the "debugging" vectors
        muminusVec.Transform(roataeMe);

        float theta_cs = muminusVec.Theta();
        float cos_theta_cs = TMath::Cos(theta_cs);
        float phi_cs = muminusVec.Phi();
        std::pair <float,float> CS_pair (cos_theta_cs, phi_cs);
	return CS_pair;
}

float PtCorrGeoFit(float d0_BS_charge, float pt_Roch, float eta, int year) {
    float pt_cor = 0.0;
    if(fabs(d0_BS_charge)<999999.){
      if (year == 2016) {
        if      (fabs(eta) < 0.9) pt_cor = 411.34 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
        else if (fabs(eta) < 1.7) pt_cor = 673.40 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
        else                     pt_cor = 1099.0 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      }
      else if (year == 2017) {
        if      (fabs(eta) < 0.9) pt_cor = 582.32 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
        else if (fabs(eta) < 1.7) pt_cor = 974.05 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
        else                     pt_cor = 1263.4 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      }
      else if (year == 2018) {
        if      (fabs(eta) < 0.9) pt_cor = 650.84 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
        else if (fabs(eta) < 1.7) pt_cor = 988.37 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
        else                     pt_cor = 1484.6 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      }
    }
    return (pt_Roch - pt_cor);
} // end of float PtCorrGeoFit(float d0_BS_charge, float pt_Roch, float eta, int year) 

//References: https://twiki.cern.ch/twiki/bin/viewauth/CMS/QuarkGluonLikelihood and https://github.com/arizzi/PisaHmm
float qglJetWeight(int partonFlavour, float pt, float eta, float qgl, int isHerwig) {

//     std::cout << partonFlavour  << " \t " << eta  << " \t " << qgl << std::endl;
    if (partonFlavour!=0 && pt>0 && fabs(eta)<2 && qgl>0) {
      //pythia
        if(isHerwig<1){
           if (abs(partonFlavour) < 4) return -0.666978*qgl*qgl*qgl + 0.929524*qgl*qgl -0.255505*qgl + 0.981581; 
           if (partonFlavour == 21)    return -55.7067*qgl*qgl*qgl*qgl*qgl*qgl*qgl + 113.218*qgl*qgl*qgl*qgl*qgl*qgl -21.1421*qgl*qgl*qgl*qgl*qgl -99.927*qgl*qgl*qgl*qgl + 92.8668*qgl*qgl*qgl -34.3663*qgl*qgl + 6.27*qgl + 0.612992; 
        }
        //powheg
        else {
            if (abs(partonFlavour) < 4) return 1.16636*qgl*qgl*qgl - 2.45101*qgl*qgl + 1.86096*qgl + 0.596896; 
            if (partonFlavour == 21)    return -63.2397*qgl*qgl*qgl*qgl*qgl*qgl*qgl + 111.455*qgl*qgl*qgl*qgl*qgl*qgl -16.7487*qgl*qgl*qgl*qgl*qgl -72.8429*qgl*qgl*qgl*qgl + 56.7714*qgl*qgl*qgl - 19.2979*qgl*qgl + 3.41825*qgl + 0.919838; 
        }
    }

    return 1.;
}


