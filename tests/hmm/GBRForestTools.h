//--------------------------------------------------------------------------------------------------
//
// GRBForestTools
//
// Utility to parse an XML weights files specifying an ensemble of decision trees into a GRBForest.
//
// Author: Jonas Rembser
//--------------------------------------------------------------------------------------------------

#include "GBRForest.h"
// #include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>

GBRForest* createGBRForest(const std::string &weightsFile);

extern "C" {
    const void* new_gbr(const char* weightfile) {  
      const GBRForest* gbr= createGBRForest(weightfile);
      return gbr;
    }

    int gbr_get_nvariables(const void* gbr) {
      const GBRForest* _gbr = (const GBRForest*)gbr;

      return _gbr->GetNVariables();
    }
    
    void gbr_eval(const void* gbr, float* out, int nev, int nfeatures, float* inputs_matrix) {
      const GBRForest* _gbr = (const GBRForest*)gbr;

      #pragma omp parallel for
      for (int iev=0; iev<nev; iev++) {
          float feats[nfeatures];
          for (int ivar = 0; ivar < nfeatures; ivar++) {
            feats[ivar] = inputs_matrix[iev*nfeatures + ivar];
          } 
          out[iev] = (float)(_gbr->GetResponse(feats));
      }
    }
}
