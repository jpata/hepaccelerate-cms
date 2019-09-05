#include "NNLOPSReweighting.h"

//(1) https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWG/SignalModelingTools (2) https://indico.cern.ch/event/628663/contributions/2643971/attachments/1484533/2304733/NNLOPS_reweighting.pdf 
double NNLOPSW(int gen_njets, double gen_Higgs_pT, TString generator) {
    const char* nnlopsfile_path = "data/nnlops/NNLOPS_reweight.root";
    TString ggH_generator = "mcatnlo";
    if(str_dataset.find("powheg")!=std::string::npos){
        ggH_generator = "powheg";
    }
    //std::cout <<"ggH generator is "<<ggH_generator<<endl;
    TFile* nnlopsFile = TFile::Open(nnlopsfile_path);
    gr_NNLOPSratio_pt_0jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_"+ggH_generator+"_0jet");
    gr_NNLOPSratio_pt_1jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_"+ggH_generator+"_1jet");
    gr_NNLOPSratio_pt_2jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_"+ggH_generator+"_2jet");
    gr_NNLOPSratio_pt_3jet = (TGraphErrors*)nnlopsFile->Get("gr_NNLOPSratio_pt_"+ggH_generator+"_3jet");

    if (gen_njets==0) NNLOPS_weight = gr_NNLOPSratio_pt_0jet->Eval(min(gen_Higgs_pT,125.0));
    else if (gen_njets==1) NNLOPS_weight = gr_NNLOPSratio_pt_1jet->Eval(min(gen_Higgs_pT,625.0));
    else if (gen_njets==2) NNLOPS_weight = gr_NNLOPSratio_pt_2jet->Eval(min(gen_Higgs_pT,800.0));
    else if (gen_njets>=3) NNLOPS_weight = gr_NNLOPSratio_pt_3jet->Eval(min(gen_Higgs_pT,925.0));
    return NNLOPS_weight;
}
