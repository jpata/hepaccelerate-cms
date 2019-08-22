
#ifndef EGAMMAOBJECTS_GBRForest
#define EGAMMAOBJECTS_GBRForest

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GBRForest                                                            //
//                                                                      //
// A fast minimal implementation of Gradient-Boosted Regression Trees   //
// which has been especially optimized for size on disk and in memory.  //
//                                                                      //
// Designed to be built from TMVA-trained trees, but could also be      //
// generalized to otherwise-trained trees, classification,              //
//  or other boosting methods in the future                             //
//                                                                      //
//  Josh Bendavid - MIT                                                 //
//////////////////////////////////////////////////////////////////////////

#include "GBRTree.h"

#include <vector>
#include <iostream>
#include <cmath>

class GBRForest {
public:
  GBRForest(int _nvariables, double _fInitialResponse) : nvariables(_nvariables), fInitialResponse(_fInitialResponse) {};

  double GetResponse(float* vector) const;
  double GetGradBoostClassifier(float* vector) const;
  double GetAdaBoostClassifier(float* vector) const { return GetResponse(vector); }

  //for backwards-compatibility
  double GetClassifier(float* vector) const { return GetGradBoostClassifier(vector); }
  void normalize(float* vector) const;

  std::vector<GBRTree>& Trees() { return fTrees; }
  const std::vector<GBRTree>& Trees() const { return fTrees; }

  void print() const {
    std::cout << "GBRForest(fTrees=" << fTrees.size() << ")" << std::endl;
    for (const auto& t : fTrees) {
      t.print();
    }
  }

  int GetNVariables() const { return nvariables; }
  double fInitialResponse;
  int nvariables;
  std::vector<GBRTree> fTrees;
  bool doNormalize;
  std::vector<std::pair<double, double>> minmax;
};

inline void GBRForest::normalize(float* vector) const {
  for (int i=0; i<nvariables; i++) {
    const auto val = vector[i];
    const auto min = minmax[i].first; 
    const auto max = minmax[i].second;
    const auto scale = 1.0/(max-min);
    const auto offset = min; 
    vector[i] = (val-offset)*scale * 2.0 - 1.0;
  }
}

//_______________________________________________________________________
inline double GBRForest::GetResponse(float* vector) const {
  double response = fInitialResponse;
  if (doNormalize) {
    normalize(vector);
  }
  for (const auto& it : fTrees) {
    auto r = it.GetResponse(vector);
    response += r;
  }
  return response;
}

//_______________________________________________________________________
inline double GBRForest::GetGradBoostClassifier(float* vector) const {
  double response = GetResponse(vector);
  return 2.0 / (1.0 + exp(-2.0 * response)) - 1;  //MVA output between -1 and 1
}

#endif
