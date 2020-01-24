#include <TLorentzVector.h>
#include <TMath.h>

std::pair<double,double> CSAngles(TLorentzVector& v1, TLorentzVector& v2, int charge);
std::pair<double,double> CSAnglesPisa(TLorentzVector& v1, TLorentzVector& v2, int charge);

extern "C" {
    void csangles_eval(
        float* out_theta, float* out_phi, int nev,
        float* pt1, float* eta1, float* phi1, float* mass1,
        float* pt2, float* eta2, float* phi2, float* mass2,
        int* charges) {
            #pragma omp parallel for default(none) shared(out_theta, out_phi, nev, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, charges) schedule(dynamic, 1000)
            for (int iev=0; iev<nev; iev++) {
                TLorentzVector v1, v2;
                v1.SetPtEtaPhiM(pt1[iev], eta1[iev], phi1[iev], mass1[iev]);
                v2.SetPtEtaPhiM(pt2[iev], eta2[iev], phi2[iev], mass2[iev]);
                const auto ret = CSAngles(v1, v2, charges[iev]);
                out_theta[iev] = (float)(ret.first);
                out_phi[iev] = (float)(ret.second);
            }
    }
    void csanglesPisa_eval(
        float* out_theta, float* out_phi, int nev,
        float* pt1, float* eta1, float* phi1, float* mass1,
        float* pt2, float* eta2, float* phi2, float* mass2,
        int* charges) {
            #pragma omp parallel for default(none) shared(out_theta, out_phi, nev, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, charges) schedule(dynamic, 1000)
            for (int iev=0; iev<nev; iev++) {
                TLorentzVector v1, v2;
                v1.SetPtEtaPhiM(pt1[iev], eta1[iev], phi1[iev], mass1[iev]);
                v2.SetPtEtaPhiM(pt2[iev], eta2[iev], phi2[iev], mass2[iev]);
                const auto ret = CSAnglesPisa(v1, v2, charges[iev]);
                out_theta[iev] = (float)(ret.first);
                out_phi[iev] = (float)(ret.second);
            }
    }
}
