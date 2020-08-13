#ifndef BTagEntry_H
#define BTagEntry_H

/**
 *
 * BTagEntry
 *
 * Represents one pt- or discriminator-dependent calibration function.
 *
 * measurement_type:    e.g. comb, ttbar, di-mu, boosted, ...
 * sys_type:            e.g. central, plus, minus, plus_JEC, plus_JER, ...
 *
 * Everything is converted into a function, as it is easiest to store it in a
 * txt or json file.
 *
 ************************************************************/

#include <string>
#include <TF1.h>
#include <TH1.h>


class BTagEntry
{
public:
  enum OperatingPoint {
    OP_LOOSE=0,
    OP_MEDIUM=1,
    OP_TIGHT=2,
    OP_RESHAPING=3,
  };
  enum JetFlavor {
    FLAV_B=0,
    FLAV_C=1,
    FLAV_UDSG=2,
  };
  struct Parameters {
    OperatingPoint operatingPoint;
    std::string measurementType;
    std::string sysType;
    JetFlavor jetFlavor;
    float etaMin;
    float etaMax;
    float ptMin;
    float ptMax;
    float discrMin;
    float discrMax;

    // default constructor
    Parameters(
      OperatingPoint op=OP_TIGHT,
      std::string measurement_type="comb",
      std::string sys_type="central",
      JetFlavor jf=FLAV_B,
      float eta_min=-2.4,//-99999.,
      float eta_max=2.4,//99999.,
      float pt_min=0.,
      float pt_max=1000.,//99999.,
      float discr_min=0.,
      float discr_max=99999.
    );

  };

  BTagEntry() {}
  BTagEntry(const std::string &csvLine);
  BTagEntry(const std::string &func, Parameters p);
  BTagEntry(const TF1* func, Parameters p);
  BTagEntry(const TH1* histo, Parameters p);
  ~BTagEntry() {}
  static std::string makeCSVHeader();
  std::string makeCSVLine() const;
  static std::string trimStr(std::string str);

  // public, no getters needed
  std::string formula;
  Parameters params;

};

#endif  // BTagEntry_H


#ifndef BTagCalibration_H
#define BTagCalibration_H

/**
 * BTagCalibration
 *
 * The 'hierarchy' of stored information is this:
 * - by tagger (BTagCalibration)
 *   - by operating point or reshape bin
 *     - by jet parton flavor
 *       - by type of measurement
 *         - by systematic
 *           - by eta bin
 *             - as 1D-function dependent of pt or discriminant
 *
 ************************************************************/

#include <map>
#include <vector>
#include <string>
#include <istream>
#include <ostream>


class BTagCalibration
{
public:
  BTagCalibration() {}
  BTagCalibration(const std::string &tagger);
  BTagCalibration(const std::string &tagger, const std::string &filename);
  ~BTagCalibration() {}

  std::string tagger() const {return tagger_;}

  void addEntry(const BTagEntry &entry);
  const std::vector<BTagEntry>& getEntries(const BTagEntry::Parameters &par) const;

  void readCSV(std::istream &s);
  void readCSV(const std::string &s);
  void makeCSV(std::ostream &s) const;
  std::string makeCSV() const;

protected:
  static std::string token(const BTagEntry::Parameters &par);

  std::string tagger_;
  std::map<std::string, std::vector<BTagEntry> > data_;

};

#endif  // BTagCalibration_H


#ifndef BTagCalibrationReader_H
#define BTagCalibrationReader_H

/**
 * BTagCalibrationReader
 *
 * Helper class to pull out a specific set of BTagEntry's out of a
 * BTagCalibration. TF1 functions are set up at initialization time.
 *
 ************************************************************/

#include <memory>
#include <string>



class BTagCalibrationReader
{
public:
  class BTagCalibrationReaderImpl;

  BTagCalibrationReader() {}
  BTagCalibrationReader(BTagEntry::OperatingPoint op,
                        const std::string & sysType="central",
                        const std::vector<std::string> & otherSysTypes={});

  void load(const BTagCalibration & c,
            BTagEntry::JetFlavor jf,
            const std::string & measurementType="comb");

  double eval(BTagEntry::JetFlavor jf,
              float eta,
              float pt,
              float discr=0.) const;

  double eval_auto_bounds(const std::string & sys,
                          BTagEntry::JetFlavor jf,
                          float eta,
                          float pt,
                          float discr=0.) const;

  std::pair<float, float> min_max_pt(BTagEntry::JetFlavor jf,
                                     float eta,
                                     float discr=0.) const;
protected:
  std::shared_ptr<BTagCalibrationReaderImpl> pimpl;
};


#endif  // BTagCalibrationReader_H

#include <fstream>
extern "C" {

    BTagCalibration* new_BTagCalibration(const char* tagger){
        auto* ret = new BTagCalibration(tagger);
        return ret;
    }

    void BTagCalibration_readCSV(BTagCalibration* obj, const char* file_path) {
        std::cout << "readCSV " << file_path << std::endl;
        std::ifstream ifile;
        ifile.open(file_path);
        obj->readCSV(ifile);
        ifile.close(); 
    }

    BTagCalibrationReader* new_BTagCalibrationReader(BTagEntry::OperatingPoint op, const char* syst, int num_other_systs, const char** other_systs) {
        std::vector<std::string> _other_systs;
        for (int i=0; i<num_other_systs; i++) {
            _other_systs.push_back(other_systs[i]);
        }
        return new BTagCalibrationReader(op, syst, _other_systs);
    }
    
    void BTagCalibrationReader_load(BTagCalibrationReader* obj, BTagCalibration* obj2, BTagEntry::JetFlavor flav, const char* type) {
        obj->load(*obj2, flav, type);
    }

    void BTagCalibrationReader_eval(
        BTagCalibrationReader* calib_b, BTagCalibrationReader* calib_c, BTagCalibrationReader* calib_l,
        float* out_w, int nev, const char* sys,
        int* flav, float* abs_eta, float* pt, float* discr) {
          #pragma omp parallel for default(none) shared(c, out_w, nev, sys, flav, abs_eta, pt, discr) schedule(dynamic, 1000)
          for (int iev=0; iev<nev; iev++) {

            auto jet_flav = BTagEntry::FLAV_UDSG;
            BTagCalibrationReader* c = calib_l; 
            if (abs(flav[iev]) == 5) { 
                jet_flav = BTagEntry::FLAV_B;
                c = calib_b;
            }
            else if (abs(flav[iev]) == 4) { 
                jet_flav = BTagEntry::FLAV_C;
                c = calib_c;
            }
            const auto ret = c->eval_auto_bounds(sys, jet_flav, abs_eta[iev], pt[iev], discr[iev]);
            out_w[iev] = ret;
        }
    }
}
