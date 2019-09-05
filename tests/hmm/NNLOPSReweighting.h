#include <TLorentzVector.h>
#include <TString.h>
#include <TMath.h>

double NNLOPSW(int gen_njets, double gen_Higgs_pT, TString generator);

extern "C" {
    void NNLOPSW_eval(
        float* out_nnlow, TString gen, int nev,
        float* GenPart_pdgId, float* t_GenPart_status, float* t_GenPart_pt) {
          for (int iev=0; iev<nev; iev++) {
            //calculate Higgs PT
            float gen_Higgs_pT = 0.;
            for(int k=0; k<t_GenPart_pdgId->size(); k++){
               if((*t_GenPart_pdgId)[k]==25 && (*t_GenPart_status)[k]==62){
                  gen_Higgs_pT = (*t_GenPart_pt)[k];
                  break;
               }
            }
            //calculate njets with pT>30 GeV
            int gen_njets = 0;
            for(int k=0; k<t_GenJet_pt->size(); k++){ 
               if((*t_GenJet_pt)[k]>30.) gen_njets++;
            }
            const auto ret = NNLOPSW(gen_njets, gen_Higgs_pT, gen);
            out_nnlow[iev] = ret.first;
        }
    }
}
