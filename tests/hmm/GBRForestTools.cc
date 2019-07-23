#include <cstdio>
#include <cstdlib>
#include <RVersion.h>
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <map>
#include <iostream>

#include "GBRTree.h"
#include "GBRForest.h"
#include "GBRForestTools.h"
#include "tinyxml2.h"

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


  size_t readVariables(tinyxml2::XMLElement* root, const char* key, std::vector<std::string>& names) {
    size_t n = 0;
    names.clear();

    if (root != nullptr) {
      for (tinyxml2::XMLElement* e = root->FirstChildElement(key); e != nullptr; e = e->NextSiblingElement(key)) {
        names.push_back(e->Attribute("Expression"));
        ++n;
      }
    }

    return n;
  }

  bool isTerminal(tinyxml2::XMLElement* node) {
    bool is = true;
    for (tinyxml2::XMLElement* e = node->FirstChildElement("Node"); e != nullptr; e = e->NextSiblingElement("Node")) {
      is = false;
    }
    return is;
  }

  unsigned int countIntermediateNodes(tinyxml2::XMLElement* node) {
    unsigned int count = 0;
    for (tinyxml2::XMLElement* e = node->FirstChildElement("Node"); e != nullptr; e = e->NextSiblingElement("Node")) {
      count += countIntermediateNodes(e);
    }
    return count > 0 ? count + 1 : 0;
  }

  unsigned int countTerminalNodes(tinyxml2::XMLElement* node) {
    unsigned int count = 0;
    for (tinyxml2::XMLElement* e = node->FirstChildElement("Node"); e != nullptr; e = e->NextSiblingElement("Node")) {
      count += countTerminalNodes(e);
    }
    return count > 0 ? count : 1;
  }

  void addNode(GBRTree& tree,
               tinyxml2::XMLElement* node,
               double scale,
               bool isRegression,
               bool useYesNoLeaf,
               bool adjustboundary,
               bool isAdaClassifier) {
    bool nodeIsTerminal = isTerminal(node);
    if (nodeIsTerminal) {
      double response = 0.;
      if (isRegression) {
        node->QueryDoubleAttribute("res", &response);
      } else {
        if (useYesNoLeaf) {
          node->QueryDoubleAttribute("nType", &response);
        } else {
          if (isAdaClassifier) {
            node->QueryDoubleAttribute("purity", &response);
          } else {
            node->QueryDoubleAttribute("res", &response);
          }
        }
      }
      response *= scale;
      tree.Responses().push_back(response);
    } else {
      int thisidx = tree.CutIndices().size();

      int selector;
      float cutval;
      bool ctype;

      node->QueryIntAttribute("IVar", &selector);
      node->QueryFloatAttribute("Cut", &cutval);
      node->QueryBoolAttribute("cType", &ctype);

      tree.CutIndices().push_back(static_cast<unsigned char>(selector));

      //newer tmva versions use >= instead of > in decision tree splits, so adjust cut value
      //to reproduce the correct behaviour
      if (adjustboundary) {
        cutval = std::nextafter(cutval, std::numeric_limits<float>::lowest());
      }
      tree.CutVals().push_back(cutval);
      tree.LeftIndices().push_back(0);
      tree.RightIndices().push_back(0);

      tinyxml2::XMLElement* left = nullptr;
      tinyxml2::XMLElement* right = nullptr;
      for (tinyxml2::XMLElement* e = node->FirstChildElement("Node"); e != nullptr; e = e->NextSiblingElement("Node")) {
        if (*(e->Attribute("pos")) == 'l')
          left = e;
        else if (*(e->Attribute("pos")) == 'r')
          right = e;
      }
      if (!ctype) {
        std::swap(left, right);
      }

      tree.LeftIndices()[thisidx] = isTerminal(left) ? -tree.Responses().size() : tree.CutIndices().size();
      addNode(tree, left, scale, isRegression, useYesNoLeaf, adjustboundary, isAdaClassifier);

      tree.RightIndices()[thisidx] = isTerminal(right) ? -tree.Responses().size() : tree.CutIndices().size();
      addNode(tree, right, scale, isRegression, useYesNoLeaf, adjustboundary, isAdaClassifier);
    }
  }

  GBRForest* init(const std::string& weightsFileFullPath, std::vector<std::string>& varNames) {
    //
    // Load weights file, for gzipped or raw xml file
    //
    tinyxml2::XMLDocument xmlDoc;

    xmlDoc.LoadFile(weightsFileFullPath.c_str());

    tinyxml2::XMLElement* root = xmlDoc.FirstChildElement("MethodSetup");
    readVariables(root->FirstChildElement("Variables"), "Variable", varNames);

    std::cout << "BDT from file " << weightsFileFullPath << " inititalized with " << varNames.size() << " variables" << std::endl;
    for (auto v : varNames) {
      std::cout << "BDT var=" << v << std::endl;
    }
    // Read in the TMVA general info
    std::map<std::string, std::string> info;
    tinyxml2::XMLElement* infoElem = xmlDoc.FirstChildElement("MethodSetup")->FirstChildElement("GeneralInfo");
    if (infoElem == nullptr) {
      std::cerr << "No GeneralInfo found in " << weightsFileFullPath << " !!\n";
      throw std::exception();
    }
    for (tinyxml2::XMLElement* e = infoElem->FirstChildElement("Info"); e != nullptr;
         e = e->NextSiblingElement("Info")) {
      const char* name;
      const char* value;
      e->QueryStringAttribute("name", &name);
      e->QueryStringAttribute("value", &value);
      info[name] = value;
    }

    // Read in the TMVA options
    std::map<std::string, std::string> options;
    tinyxml2::XMLElement* optionsElem = xmlDoc.FirstChildElement("MethodSetup")->FirstChildElement("Options");
    if (optionsElem == nullptr) {
      std::cerr << "No Options found in " << weightsFileFullPath << " !!\n";
      throw std::exception();
    }
    for (tinyxml2::XMLElement* e = optionsElem->FirstChildElement("Option"); e != nullptr;
         e = e->NextSiblingElement("Option")) {
      const char* name;
      e->QueryStringAttribute("name", &name);
      options[name] = e->GetText();
    }

    // Get root version number if available
    int rootTrainingVersion(0);
    if (info.find("ROOT Release") != info.end()) {
      std::string s = info["ROOT Release"];
      rootTrainingVersion = std::stoi(s.substr(s.find("[") + 1, s.find("]") - s.find("[") - 1));
    }

    // Get the boosting weights
    std::vector<double> boostWeights;
    tinyxml2::XMLElement* weightsElem = xmlDoc.FirstChildElement("MethodSetup")->FirstChildElement("Weights");
    if (weightsElem == nullptr) {
      std::cerr << "No Weights found in " << weightsFileFullPath << " !!\n";
      throw std::exception() ;
    }
    bool hasTrees = false;
    for (tinyxml2::XMLElement* e = weightsElem->FirstChildElement("BinaryTree"); e != nullptr;
         e = e->NextSiblingElement("BinaryTree")) {
      hasTrees = true;
      double w;
      e->QueryDoubleAttribute("boostWeight", &w);
      boostWeights.push_back(w);
    }
    if (!hasTrees) {
      std::cerr << "No BinaryTrees found in " << weightsFileFullPath << " !!\n";
      throw std::exception();
    }

    bool isRegression = info["AnalysisType"] == "Regression";

    //special handling for non-gradient-boosted (ie ADABoost) classifiers, where tree responses
    //need to be renormalized after the training for evaluation purposes
    bool isAdaClassifier = !isRegression && options["BoostType"] != "Grad";
    bool useYesNoLeaf = isAdaClassifier && options["UseYesNoLeaf"] == "True";

    //newer tmva versions use >= instead of > in decision tree splits, so adjust cut value
    //to reproduce the correct behaviour
    bool adjustBoundaries =
        (rootTrainingVersion >= ROOT_VERSION(5, 34, 20) && rootTrainingVersion < ROOT_VERSION(6, 0, 0)) ||
        rootTrainingVersion >= ROOT_VERSION(6, 2, 0);

    auto* forest = new GBRForest(varNames.size(), isRegression ? boostWeights[0] : 0.);

    double norm = 0;
    if (isAdaClassifier) {
      for (double w : boostWeights) {
        norm += w;
      }
    }

    forest->Trees().reserve(boostWeights.size());
    size_t itree = 0;
    // Loop over tree estimators
    for (tinyxml2::XMLElement* e = weightsElem->FirstChildElement("BinaryTree"); e != nullptr;
         e = e->NextSiblingElement("BinaryTree")) {
      double scale = isAdaClassifier ? boostWeights[itree] / norm : 1.0;

      tinyxml2::XMLElement* root = e->FirstChildElement("Node");
      forest->Trees().push_back(GBRTree(countIntermediateNodes(root), countTerminalNodes(root)));
      auto& tree = forest->Trees().back();

      addNode(tree, root, scale, isRegression, useYesNoLeaf, adjustBoundaries, isAdaClassifier);

      //special case, root node is terminal, create fake intermediate node at root
      if (tree.CutIndices().empty()) {
        tree.CutIndices().push_back(0);
        tree.CutVals().push_back(0);
        tree.LeftIndices().push_back(0);
        tree.RightIndices().push_back(0);
      }
      ++itree;
    }

    return forest;
  }

// Create a GBRForest from an XML weight file
GBRForest* createGBRForest(const std::string& weightsFile) {
  std::vector<std::string> varNames;
  auto gbrForest(init(weightsFile, varNames));
  return std::move(gbrForest);
}
