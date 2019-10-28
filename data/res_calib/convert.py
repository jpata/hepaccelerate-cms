import os, sys

import ROOT
import json
import numpy as np
from array import array


def convert_to_th3d(path, in_file, era):    
    modes = ["Data", "MC"]
    pt_bin_names = ["pt_0", "pt_1", "pt_2", "pt_3"]
    pt_bins = [0., 45., 52., 62., 9999.]
    eta_bins = [0., 0.9, 1.8, 2.4]

    f = ROOT.TFile.Open(path+in_file)

    for mode in modes:
        name = "res_calib_"+mode+"_"+era
        hist_3d = ROOT.TH3D(name, name, 4, array('d', pt_bins), 3, array('d', eta_bins), 3, array('d', eta_bins))
        for iptb, ptb in enumerate(pt_bin_names):
            hname = "correction{0}_{1}".format(mode, ptb)
            hist = f.Get(hname)
            for ietab1 in range(1,hist.GetNbinsX() + 1):
                for ietab2 in range(1,hist.GetNbinsY() + 1):
                    factor = hist.GetBinContent(ietab1, ietab2)
                    if iptb is 0: # in first pT bin currently there is only one bin by eta1 -> split it in 3 for consistency
                        hist_3d.SetBinContent(1, 1, ietab2, hist.GetBinContent(1, ietab2))
                        hist_3d.SetBinContent(1, 2, ietab2, hist.GetBinContent(1, ietab2))
                        hist_3d.SetBinContent(1, 3, ietab2, hist.GetBinContent(1, ietab2))
                    else:
                        hist_3d.SetBinContent(iptb+1, ietab1, ietab2, hist.GetBinContent(ietab1, ietab2))

        f_out = ROOT.TFile.Open(path+name+".root", "RECREATE")
        hist_3d.Write()
        f_out.Close()


path = "data/ebeCalib/"

calib_files = {
    "2016": "ebe_mass_uncertaintiy_calibration_2016.root",
    "2017": "ebe_mass_uncertaintiy_calibration_2017_v2.root",
    "2018": "ebe_mass_uncertaintiy_calibration_2018.root",
}

for era, cfile in calib_files.iteritems():
    convert_to_th3d(path, cfile, era)
