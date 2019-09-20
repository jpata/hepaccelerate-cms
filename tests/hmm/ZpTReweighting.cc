#include "ZpTReweighting.h"

double ZpTReweighting::zptw(double pt, int itune)const{

    double ZpT_weight = 1.0;
    if(itune==1){
        if(pt<0) ZpT_weight = 1.0;
        else if(pt<10.) ZpT_weight = 0.910385;
        else if(pt<20.) ZpT_weight = 1.13543;
        else if(pt<30.) ZpT_weight = 1.10441;
        else if(pt<40.) ZpT_weight = 1.01315;
        else if(pt<50.) ZpT_weight = 0.982598;
        else if(pt<60.) ZpT_weight = 0.980697;
        else if(pt<70.) ZpT_weight = 0.972673;
        else if(pt<80.) ZpT_weight = 0.972325;
        else if(pt<100.) ZpT_weight = 0.966127;
        else if(pt<150.) ZpT_weight = 0.953262;
        else if(pt<200.) ZpT_weight = 0.933403;
        else if(pt<1000.) ZpT_weight = 0.904518;
    }        
    // 2016 samples with Tune CUETP8M1 have different Z pt reweighting
    else if(itune==2){
        if(pt<0) ZpT_weight = 1.0;
        else if(pt<10.) ZpT_weight = 1.05817;
        else if(pt<20.) ZpT_weight = 0.994488;
        else if(pt<30.) ZpT_weight = 0.930056;
        else if(pt<40.) ZpT_weight = 0.925206;
        else if(pt<50.) ZpT_weight = 0.946403;
        else if(pt<60.) ZpT_weight = 0.962136;
        else if(pt<70.) ZpT_weight = 0.965316;
        else if(pt<80.) ZpT_weight = 0.978209;
        else if(pt<100.) ZpT_weight = 0.988761;
        else if(pt<150.) ZpT_weight = 0.982497;
        else if(pt<200.) ZpT_weight = 0.971749;
        else if(pt<1000.) ZpT_weight = 0.914429;
    }
    return ZpT_weight;
}
