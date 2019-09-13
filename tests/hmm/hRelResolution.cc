#include "hRelResolution.h"

void hRelResolution::init(const char* path){
    std::unique_ptr<TFile> f_param(TFile::Open(path));
    hmuon = (TH2F*)(f_param->Get("PtErrParametrization"))->Clone("copy_PtErrParametrization");
    hmuon->SetDirectory(0);
    f_param->Close();
}

float hRelResolution::hres(float pt1, float eta1, float pt2, float eta2)const{
    float MuonPtErr1=hmuon->GetBinContent(hmuon->FindBin( (log(pt1)),fabs(eta1) ));
    float MuonPtErr2=hmuon->GetBinContent(hmuon->FindBin( (log(pt2)),fabs(eta2) ));
    return sqrt(0.5*(pow(MuonPtErr1,2)+pow(MuonPtErr2,2)));
}
