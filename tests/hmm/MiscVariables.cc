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
