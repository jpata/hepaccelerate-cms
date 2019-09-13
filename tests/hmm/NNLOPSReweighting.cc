#include "NNLOPSReweighting.h"

//(1) https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWG/SignalModelingTools (2) https://indico.cern.ch/event/628663/contributions/2643971/attachments/1484533/2304733/NNLOPS_reweighting.pdf 
void NNLOPSReweighting::init(const char* path){
    std::unique_ptr<TFile> nnlopsFile(TFile::Open(path));
    gr_NNLOPSratio_pt_mcatnlo_0jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_mcatnlo_0jet");
    gr_NNLOPSratio_pt_mcatnlo_1jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_mcatnlo_1jet");
    gr_NNLOPSratio_pt_mcatnlo_2jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_mcatnlo_2jet");
    gr_NNLOPSratio_pt_mcatnlo_3jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_mcatnlo_3jet");
    gr_NNLOPSratio_pt_powheg_0jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_powheg_0jet");
    gr_NNLOPSratio_pt_powheg_1jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_powheg_1jet");
    gr_NNLOPSratio_pt_powheg_2jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_powheg_2jet");
    gr_NNLOPSratio_pt_powheg_3jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_powheg_3jet");
    nnlopsFile->Close();
}

double NNLOPSReweighting::nnlopsw(int gen_njets, double gen_Higgs_pT, int igen)const{

    double NNLOPS_weight = 1.0;
    if(igen==1){
    if (gen_njets==0) NNLOPS_weight = gr_NNLOPSratio_pt_mcatnlo_0jet->Eval(std::min(gen_Higgs_pT,125.0));
        else if (gen_njets==1) NNLOPS_weight = gr_NNLOPSratio_pt_mcatnlo_1jet->Eval(std::min(gen_Higgs_pT,625.0));
        else if (gen_njets==2) NNLOPS_weight = gr_NNLOPSratio_pt_mcatnlo_2jet->Eval(std::min(gen_Higgs_pT,800.0));
        else if (gen_njets>=3) NNLOPS_weight = gr_NNLOPSratio_pt_mcatnlo_3jet->Eval(std::min(gen_Higgs_pT,925.0));
    }
    else if(igen==2){
        if (gen_njets==0) NNLOPS_weight = gr_NNLOPSratio_pt_powheg_0jet->Eval(std::min(gen_Higgs_pT,125.0));
        else if (gen_njets==1) NNLOPS_weight = gr_NNLOPSratio_pt_powheg_1jet->Eval(std::min(gen_Higgs_pT,625.0));
        else if (gen_njets==2) NNLOPS_weight = gr_NNLOPSratio_pt_powheg_2jet->Eval(std::min(gen_Higgs_pT,800.0));
        else if (gen_njets>=3) NNLOPS_weight = gr_NNLOPSratio_pt_powheg_3jet->Eval(std::min(gen_Higgs_pT,925.0));
    }
    return NNLOPS_weight;
}
