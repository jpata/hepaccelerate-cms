import numpy as np

categories = {
    "dimuon": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "wmh_125",
            "wph_125",
            "zh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120_herwig",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy",
            "www","wwz","wzz","zzz",
        ],
    },
    "z_peak": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "wmh_125",
            "wph_125",
            "zh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120_herwig",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_0j", "dy_1j", "dy_2j",
            "www","wwz","wzz","zzz",
        ],
    },
    "h_sideband": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "wmh_125",
            "wph_125",
            "zh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll105_160_ptJ_herwig",
            "ewk_lljj_mll105_160_dipole_pythia",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_m105_160_amc_01j", "dy_m105_160_vbf_amc_01j",
            "dy_m105_160_amc_2j", "dy_m105_160_vbf_amc_2j",
            "www","wwz","wzz","zzz",
        ],
    },
    "h_peak": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "wmh_125",
            "wph_125",
            "zh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll105_160_ptJ_herwig",
            "ewk_lljj_mll105_160_dipole_pythia",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_m105_160_amc_01j", "dy_m105_160_vbf_amc_01j",
            "dy_m105_160_amc_2j", "dy_m105_160_vbf_amc_2j",
            "www","wwz","wzz","zzz",
        ],
    }
}
proc_grps = [
        ("dy_m105_160_01j", ["dy_m105_160_amc_01j", "dy_m105_160_vbf_amc_01j"]),
        ("dy_m105_160_2j", ["dy_m105_160_amc_2j", "dy_m105_160_vbf_amc_2j"]),
        ("vh_125",["wmh_125", "wph_125", "zh_125"]),
        ("vv", ["wz_3lnu", "ww_2l2nu", "wz_2l2q", "zz", "www","wwz","wzz","zzz"]),
        ("top", ["ttjets_sl", "ttjets_dl", "st_tw_top", "st_tw_antitop"]),
    ]
combined_signal_samples= ["ggh_amcPS_pythia_125", "vbf_powheg_pythia_dipole_125", "vh_125", "tth_125"]
combined_categories = {
    "dimuon": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "vh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "vv", 
            "ewk_lljj_mll50_mjj120_herwig",
            #"st_top",
            #"st_t_antitop",
            "top",
            "dy",
            #"vvv",
        ],
    },
    "z_peak": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "vh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "vv",
            "ewk_lljj_mll50_mjj120_herwig",
            #"st_top",
            #"st_t_antitop",
            "top",
            "dy_0j", "dy_1j", "dy_2j",
            #"vvv",
        ],
    },
    "h_sideband": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "vh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "vv", 
            "ewk_lljj_mll105_160_ptJ_herwig",
            "ewk_lljj_mll105_160_dipole_pythia",
            #"st_top",
            #"st_t_antitop",
            "top",
            "dy_m105_160_01j",
            "dy_m105_160_2j", 
            #"vvv",
        ],
    },
    "h_peak": {
        "datacard_processes" : [
            "ggh_amcPS_pythia_125",
            "vbf_powheg_pythia_dipole_125",
            "vbf_powheg_herwig_125",
            "vbf_powheg_pythia_dipole_125_ref",
            "vh_125",
            "tth_125",
            #"wz_1l1nu2q",
            "vv", 
            "ewk_lljj_mll105_160_ptJ_herwig",
            "ewk_lljj_mll105_160_dipole_pythia",
            #"st_top",
            #"st_t_antitop",
            "top",
            "dy_m105_160_01j", 
            "dy_m105_160_2j", 
            #"vvv",
        ],
    }
}

colors = {
    "dy": (254, 254, 83),
    "ewk": (109, 253, 245),
    "top": (67, 150, 42),
    "vvv": (247, 206, 205),
    "vv": (100, 105, 98),
    "higgs": (0, 0, 0),
}

remove_proc = ["vbf_powheg_pythia_dipole_125_ref","vbf_powheg_herwig_125","ewk_lljj_mll105_160_dipole_pythia"]

process_groups = [
    ("higgs", ["ggh_amcPS_pythia_125", "vbf_powheg_pythia_dipole_125", "wmh_125", "wph_125", "zh_125", "tth_125"]),
    ("vv", ["wz_3lnu", "ww_2l2nu", "wz_2l2q", "zz"]),
    ("vvv", ["www","wwz","wzz","zzz"]),
    ("ewk", ["ewk_lljj_mll50_mjj120_herwig", "ewk_lljj_mll105_160_ptJ_herwig"]),
    ("top", ["ttjets_sl", "ttjets_dl", "st_tw_top", "st_tw_antitop"]),
    ("dy", ["dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc_01j", "dy_m105_160_vbf_amc_01j", "dy_m105_160_amc_2j", "dy_m105_160_vbf_amc_2j", "dy"]),
]

extra_plot_kwargs = {
    "hist__dimuon__num_jets": {
        "do_log": True,
        "ylim": (10, 1e10),
    },
    "hist__dnn_presel__num_jets": {
        "do_log": True,
        "ylim": (10, 1e9),
    },
    "hist__dimuon_invmass_z_peak_cat5__subleading_jet_pt": {
        "do_log": True,
        "xlim": (25, 300)
    },
    "hist__dimuon_invmass_h_peak_cat5__subleading_jet_pt": {
        "do_log": True,
        "xlim": (25, 300)
    },
    "hist__dimuon_invmass_h_sideband_cat5__subleading_jet_pt": {
        "do_log": True,
        "xlim": (25, 300)
    },

    "hist__dimuon_invmass_z_peak_cat5__leading_jet_pt": {
        "do_log": True,
        "xlim": (35, 300)
    },
    "hist__dimuon_invmass_h_peak_cat5__leading_jet_pt": {
        "do_log": True,
        "xlim": (35, 300)
    },
    "hist__dimuon_invmass_h_sideband_cat5__leading_jet_pt": {
        "do_log": True,
        "xlim": (35, 300)
    },

    "hist__dimuon_invmass_z_peak_cat5__num_jets": {
        "do_log": True,
        "xlim": (2, 8)
    },
    "hist__dimuon_invmass_h_peak_cat5__num_jets": {
        "do_log": True,
        "xlim": (2, 8)
    },
    "hist__dimuon_invmass_h_sideband_cat5__num_jets": {
        "do_log": True,
        "xlim": (2, 8)
    },

    "hist__dimuon_invmass_z_peak_cat5__num_soft_jets": {
        "do_log": True,
        "xlim": (0, 8)
    },
    "hist__dimuon_invmass_h_peak_cat5__num_soft_jets": {
        "do_log": True,
        "xlim": (0, 8)
    },
    "hist__dimuon_invmass_h_sideband_cat5__num_soft_jets": {
        "do_log": True,
        "xlim": (0, 8)
    },


    "hist__dimuon_invmass_z_peak_cat5__dnn_pred2": {
        "xbins": "uniform",
        "do_log": True
    },
    "hist__dimuon_invmass_h_peak_cat5__dnn_pred2": {
        "xbins": "uniform",
        "xlim": (1, 9),
        "ylim": (0, 50),
        "mask_data_from_bin": 2,
    },
    "hist__dimuon_invmass_h_sideband_cat5__dnn_pred2": {
        "xbins": "uniform",
        "do_log": True,
    },
    "hist__dimuon_invmass_z_peak_cat5__bdt_ucsd": {
        "do_log": True,
    },
    "hist__dimuon_invmass_h_peak_cat5__bdt_ucsd": {
        "do_log": False,
        "mask_data_from_bin": 5,
    },
    "hist__dimuon_invmass_h_sideband_cat5__bdt_ucsd": {
        "do_log": True,
    },
}

controlplots_shape = [
    "inv_mass",
    "dnn_pred2",
    "dnnPisa_pred_atanh"
]

cross_sections = {
    "dy": 2026.96*3, #https://indico.cern.ch/event/841566/contributions/3565385/attachments/1914850/3165328/Drell-Yan_jets_crosssection_September2019.pdf 
    "dy_0j": 4620.52, #https://indico.cern.ch/event/673253/contributions/2756806/attachments/1541203/2416962/20171016_VJetsXsecsUpdate_PH-GEN.pdf
    "dy_1j": 859.59,
    "dy_2j": 338.26,
    "dy_m105_160_mg_01j": 46.9479, #Pisa 47.17
    "dy_m105_160_amc_01j": 46.9479, # https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "dy_m105_160_mg_2j": 46.9479, #Pisa 47.17
    "dy_m105_160_amc_2j": 46.9479, # https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "dy_m105_160_vbf_mg_01j": {"2016": 1.77, "2017": 2.04, "2018": 2.03}, #Using Pisa for sync, caltech group xs 46.9479*0.0425242
    "dy_m105_160_vbf_amc_01j": {"2016": 1.77, "2017": 2.04, "2018": 2.03}, 
    "dy_m105_160_vbf_mg_2j": {"2016": 1.77, "2017": 2.04, "2018": 2.03}, #Using Pisa for sync, caltech group xs 46.9479*0.0425242
    "dy_m105_160_vbf_amc_2j": {"2016": 1.77, "2017": 2.04, "2018": 2.03}, 
    "ggh_powheg_pythia_125": 0.010571, #48.61 * 0.0002176; https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNHLHE2019
    "ggh_amcPS_pythia_125": 0.010571,
    "ggh_powhegPS_pythia_125": 0.010571,
    "ggh_amcPS_125": 0.010571,
    "ggh_amcPS_TuneCP5down_125": 0.010571,
    "ggh_amcPS_TuneCP5up_125": 0.010571,
    "ggh_amc_pythia_125": 0.010571,

    "ggh_amcPS_pythia_120": 1.265E-02, #5.222E+01 x 2.423E-04 (xs(ggH,H=120 GeV) X br(H->mumu,H=120 GeV))
    "ggh_powhegPS_pythia_120": 1.265E-02,

    "ggh_amcPS_pythia_130": 8.505E-03, #4.531E+01 x 1.877E-04 (xs(ggH, H=130 GeV) X br(H->mumu, H=130 GeV))
    "ggh_powhegPS_pythia_130": 8.505E-03,

    "vbf_125": 0.000823,
    "vbf_sync": 0.000823,
    "vbf_powheg_herwig_125": 0.000823,
    "vbf_powheg_pythia_125": 0.000823,
    "vbf_powhegPS_pythia_125": 0.000823,
    "vbf_powheg_pythia_dipole_125": 0.000823,
    "vbf_powheg_pythia_dipole_125_ref": 0.000823,
    "vbf_amc_herwig_125": 0.000823,
    "vbf_amcPS_TuneCP5down_125": 0.000823,
    "vbf_amcPS_TuneCP5up_125": 0.000823,
    "vbf_amcPS_pythia_125": 0.000823,
    "vbf_amc_pythia_125": 0.000823,
    "vbf_amcPS_pythia_125": 0.000823,

    "vbf_powheg_herwig_120": 9.535E-04, #3.935E+00 x 2.423E-04
    "vbf_amcPS_120": 9.535E-04,
    "vbf_powheg_pythia_120": 9.535E-04,
    "vbf_powhegPS_pythia_120": 9.535E-04,
    "vbf_amc_herwig_120": 9.535E-04,
    "vbf_amcPS_pythia_120": 9.535E-04,
    "vbf_powheg_pythia_dipole_120": 9.535E-04,
    "vbf_powheg_pythia_dipole_120_ref": 9.535E-04,
    
    "vbf_powheg_herwig_130": 6.827E-04, #3.637E+00 x 1.877E-04
    "vbf_amcPS_130": 6.827E-04,
    "vbf_powheg_pythia_130": 6.827E-04,
    "vbf_powhegPS_pythia_130": 6.827E-04,
    "vbf_amc_herwig_130": 6.827E-04,
    "vbf_amcPS_pythia_130": 6.827E-04,
    "vbf_powheg_pythia_dipole_130": 6.827E-04,
    "vbf_powheg_pythia_dipole_130_ref": 6.827E-04,
    
    "wmh_125": 0.000116,
    "wmh_120": 1.476E-04,
    "wmh_130": 8.777E-05,

    "wph_125": 0.000183,
    "wph_120": 2.316E-04,
    "wph_130": 1.392E-04,

    "zh_125": 0.000192,
    "zh_120": 2.408E-04,
    "zh_130": 1.483E-04,

    "tth_125": 0.000110,
    "tth_120": 1.380E-04,
    "tth_130": 8.520E-05,

    "bbh_125": 0.000106,
    "ttjets_dl": 85.656,
    "ttjets_sl": 687.0,
    "ww_2l2nu": 5.595,
    "wz_3lnu":  4.42965,
    "wz_2l2q": 5.595,
    "wz_1l1nu2q": 11.61,
    "zz": 16.523,
    "st_top": 136.02,
    "st_t_antitop": 80.95,
    "st_tw_top": 35.85,
    "st_tw_antitop": 35.85,
    "ewk_lljj_mll105_160_dipole_pythia": {"2016": 0.07486, "2017": 0.0789, "2018": 0.0789},
    "ewk_lljj_mll105_160_herwig": 0.0508896,
    "ewk_lljj_mll105_160_ptJ_herwig": {"2016": 0.07486, "2017": 0.0789, "2018": 0.0789}, #from Pisa Group https://github.com/arizzi/PisaHmm/blob/59bbce76ab1532c59b44b41a9371591204f24df6/samples2016.py#L30

    # Note via Nan L.: the 2016 sample has a different tune, for which Stephane C.
    # computed a new cross-section from MINIAOD using
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToGenXSecAnalyzer
    "ewk_lljj_mll50_mjj120_herwig": {"2016": 1.611, "2017": 1.700, "2018": 1.700},
    "ewk_lljj_mll50_mjj120_ptJ_herwig": 2.734,
    "ewk_lljj_mll50_mjj120_pythia": {"2016": 1.611, "2017": 1.700, "2018": 1.700},

    "ttw": 0.2001,
    "ttz": 0.2529,
    "st_t_top": 3.36,
    "www": 0.2086,
    "wwz": 0.1651,
    "wzz": 0.05565,
    "zzz": 0.01398
}

signal_samples = ["ggh_amcPS_pythia_125", "vbf_powhegPS_pythia_125", "wmh_125", "wph_125", "zh_125", "tth_125"]

#has to include 125
mass_point = [125]

#mass scan
#mass_point = [120, 120.5, 121, 121.5, 122, 122.5, 123, 123.5, 124, 124.5, 125, 125.5, 126, 126.5, 127, 127.5, 128, 128.5, 129, 129.5, 130]

#jec_unc = [
#    'AbsoluteMPFBias', 'AbsoluteScale', 'AbsoluteStat',
#    'FlavorQCD', 'TimePtEta', 'Fragmentation', 'PileUpDataMC',
#    'PileUpPtBB', 'PileUpPtEC1', 'PileUpPtEC2',
#    'PileUpPtHF', 'PileUpPtRef', 'RelativeBal', 'RelativeFSR', 'RelativeJEREC1',
#    'RelativeJEREC2', 'RelativeJERHF', 'RelativePtBB', 'RelativePtEC1', 'RelativePtEC2',
#    'RelativePtHF', 'RelativeSample', 'RelativeStatEC', 'RelativeStatFSR', 'RelativeStatHF',
#    'SinglePionECAL', 'SinglePionHCAL']

#These subtotals can be used for cross-checks
#, 'SubTotalAbsolute', 'SubTotalMC', 'SubTotalPileUp',
#    'SubTotalPt', 'SubTotalRelative', 'SubTotalScale', 'Total', 'TotalNoFlavor',
#    'TotalNoFlavorNoTime', 'TotalNoTime']

#Reduced JEC
jec_unc = ['Absolute', 'Absolute2018', 'BBEC1', 'BBEC12018', 'EC2', 'EC22018', 'FlavorQCD', 'HF', 'HF2018', 'RelativeBal', 'RelativeSample2018', 'Absolute2017', 'BBEC12017', 'EC22017', 'HF2017', 'RelativeSample2017', 'Absolute2016', 'BBEC12016', 'EC22016', 'HF2016', 'RelativeSample2016']

jer_unc = ["jerB1","jerB2","jerEC1","jerEC2","jerF1","jerF2"]

#Uncomment to use just the total JEC for quick tests
#jec_unc = ["Total"]

#qcd uncertainties for VBF
VBF_STXS_unc = ["THU_VBF_Yield", "THU_VBF_Mjj60", "THU_VBF_Mjj120", "THU_VBF_Mjj350", "THU_VBF_PTH200", "THU_VBF_PTH25", "THU_VBF_JET01", "THU_VBF_Mjj1000", "THU_VBF_Mjj700", "THU_VBF_Mjj1500"]

#"btag_weight_hfstats1","btag_weight_hfstats2" removed to due very small uncertainty <=2% for DY 2j and one sided.
btag_unc = ["btag_weight_jes","btag_weight_lf","btag_weight_cferr1","btag_weight_cferr2","btag_weight_hf","btag_weight_lfstats1","btag_weight_lfstats2"]

#"jet_puid"
shape_systematics = jec_unc + jer_unc + VBF_STXS_unc + btag_unc + ["trigger", "id", "iso", "qgl_weight", "puWeight", "L1PreFiringWeight","DYLHERenZ","EWZLHERenZ","DYLHEFacZ","EWZLHEFacZ","DYLHEFac","EWZLHEFac","DYLHERen","EWZLHERen","LHEPdfWeight","EWZ105160PS", "VBFHPS","DYshape_DNN"] 

jec_unc_rename = ['CMS_scale_j_absolute', 'CMS_scale_j_absolute_2018', 'CMS_scale_j_bbec1', 'CMS_scale_j_bbec1_2018', 'CMS_scale_j_ec2', 'CMS_scale_j_ec2_2018', 'CMS_scale_j_flavorQCD', 'CMS_scale_j_hf', 'CMS_scale_j_hf_2018', 'CMS_scale_j_relativeBal', 'CMS_scale_j_relativeSample_2018', 'CMS_scale_j_absolute_2017', 'CMS_scale_j_bbec1_2017', 'CMS_scale_j_ec2_2017', 'CMS_scale_j_hf_2017', 'CMS_scale_j_relativeSample_2017', 'CMS_scale_j_absolute_2016', 'CMS_scale_j_bbec1_2016', 'CMS_scale_j_ec2_2016', 'CMS_scale_j_hf_2016', 'CMS_scale_j_relativeSample_2016']

jer_unc_rename = ["CMS_res_j_barrel", "CMS_res_j_endcap1", "CMS_res_j_endcap2lowpt", "CMS_res_j_endcap2highpt", "CMS_res_j_forwardlowpt", "CMS_res_j_forwardhighpt"]

#"CMS_btag_Hfstats1","CMS_btag_Hfstats2"
btag_unc_rename = ["CMS_btag_jes","CMS_btag_Lf","CMS_btag_Cferr1","CMS_btag_Cferr2","CMS_btag_Hf","CMS_btag_Lfstats1","CMS_btag_Lfstats2"]

other_syst_rename = ["CMS_pileup_vbf", "CMS_prefiring"]
rename_syst= jec_unc_rename + jer_unc_rename+ btag_unc_rename + other_syst_rename
org_syst = jec_unc + jer_unc+ btag_unc + ["puWeight", "L1PreFiringWeight"]
decorrelate_syst = ["trigger", "id", "iso", "EWKLHEScaleWeight_norm", "EWZ105160PS","Topxsec","VVxsec"] 

py_samp = ["ggh_amcPS_pythia_125", "dy_m105_160_01j", "dy_m105_160_2j","top", "vv", "vh_125","tth_125"]
herwig_samp = ["ewk_lljj_mll105_160_ptJ_herwig"]
pydipole_samp = ["vbf_powheg_pythia_dipole_125"]

LHEpdf_norm_samp = [ ("ggh_amcPS_pythia_125","ggh"),
                ("dy_m105_160_2j","dy_2j"),
                ("vh_125","vh"),
                ("tth_125","tth"),
                ("ewk_lljj_mll105_160_ptJ_herwig","ewk"),
                ("vbf_powheg_pythia_dipole_125","vbf")]

LHEpdf_samp = [ ("ggh_amcPS_pythia_125","ggh"),
                ("dy_m105_160_2j","dy_2j"),
                ("vh_125","vh"),
                ("tth_125","tth"),
                ("dy_m105_160_01j","dy_01j"),
                ("top","top"),
                ("vv","vv"),
                ("ewk_lljj_mll105_160_ptJ_herwig","ewk"),
                ("vbf_powheg_pythia_dipole_125","vbf")]

common_scale_uncertainties = {
    "lumi_13TeV_2016": {
        "2016":1.022,
        "2017":1.0,
        "2018":1.0,
    },
    "lumi_13TeV_2017": {
        "2016":1.0,
        "2017":1.02,
        "2018":1.0,
    },
    "lumi_13Tev_2018": {
        "2016":1.0,
        "2017":1.0,
        "2018":1.015,
    },
    "lumi_13Tev_XY":{
        "2016":1.009,
        "2017":1.008,
        "2018":1.02,
    },
    "lumi_13Tev_LS":{
        "2016":1.0,
        "2017":1.003,
        "2018":1.002,
    },
    "lumi_13Tev_BBD":{
        "2016":1.004,
        "2017":1.004,
        "2018":1.0,
    },
    "lumi_13Tev_DB":{
        "2016":1.005,
        "2017":1.005,
        "2018":1.0,
    },
    "lumi_13Tev_BCC":{
        "2016":1.0,
        "2017":1.003,
        "2018":1.002,
    },
    "lumi_13Tev_GS":{
        "2016":1.004,
        "2017":1.001,
        "2018":1.0,
    },
}

#http://twiki.ihep.ac.cn/twiki/view/CMS/CombineTutorial
#"zh_125": {"THU_zh_pdfas": 1.016},
#"zh_125": {"THU_zh_qcdscale": 1.038/0.969},
scale_uncertainties = {
    "ggh_amcPS_pythia_125": {"THU_ggh_pdfas": 1.032, "THU_ggh_qcdscale": "1.046/0.933", "THU_hmm_br": 1.0123, "ggHxsec":1.25},
    "vbf_powheg_pythia_dipole_125": {"THU_vbf_pdfas": 1.021, "THU_hmm_br": 1.0123},
    "vh_125": {"THU_vh_pdfas": 1.019, "THU_vh_qcdscale": "1.005/0.993", "THU_hmm_br": 1.0123},
    "tth_125": {"THU_tth_pdfas": 1.036, "THU_tth_qcdscale": "1.058/0.908", "THU_hmm_br": 1.0123},
    "ww_2l2nu": {"VVxsec": 1.10},
    "wz_3lnu": {"VVxsec": 1.10},
    "wz_2l2q": {"VVxsec": 1.10},
    "wz_2l2q": {"VVxsec": 1.10},
    "zz": {"VVxsec": 1.10},
    "wjets": {"WJetsxsec": 1.10},
    "vv" :{"VVxsec": 1.075},
    "top": {"Topxsec": 1.13},
    #"dy_m105_160_amc": {"DYxsec": 1.10},
    #"dy_m105_160__vbf_amc": {"DYxsec": 1.10},
    #"ewk_lljj_mll105_160_ptJ": {"EWZxsec": 1.20},
    #"ewk_lljj_mll50_mjj120": {"EWZxsecZ": 1.20},
    #"ttjets_sl": {"TTxsec": 1.05},
    #"ttjets_dl": {"TTxsec": 1.05},
    #"st_t_top": {"STxsec": 1.05},
    #"st_t_antitop": {"STxsec": 1.05},
    #"st_tw_top": {"STxsec": 1.05},
    #"st_tw_antitop": {"STxsec": 1.05},
}

HSTXS_rel = {
    "THU_VBF_Yield": [0.003799,0.003801,0.003799,0.003789,0.003812,0.003797,0.003788,0.0038,0.0038,0.003798,0.003798,0.003805,0.003796,0.0038,0.003796,0.003792,0.003799,0.003797,0.003806,0.003807,0.003836,0.003747,0.003777,0.003764,0.003815],
    "THU_VBF_PTH200": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.000313,0.0003129,0.0003128,0.0003129,0.0003132,0.0003129,0.000313,0.0003128,-0.002573,-0.002574,-0.002573,-0.002574,-0.002573,-0.002574,-0.002574,-0.002574],
    "THU_VBF_Mjj60": [0.0,0.0,0.0,-0.1777,0.004936,0.004942,-0.1777,0.004939,0.00494,0.004941,0.00494,0.004937,0.00494,0.004936,0.004942,0.004932,0.004938,0.004938,0.004952,0.004922,0.004931,0.004921,0.00496,0.004943,0.004948],
    "THU_VBF_Mjj120": [0.0,0.0,0.0,0.0,-0.07242,0.003604,0.0,-0.07242,0.0003605,0.003607,0.003606,0.003611,0.003607,0.003609,0.003604,0.003611,0.003607,0.003603,0.003604,0.003585,0.003615,0.003623,0.003603,0.003628,0.003613],
    "THU_VBF_Mjj350": [0.0,0.0,0.0,0.0,0.0,-0.01548,0.0,0.0,-0.01548,0.00515,0.005152,0.005149,0.005153,0.005145,0.00515,0.005143,0.005152,0.005138,0.00514,0.005124,0.005153,0.005178,0.005149,0.005168,0.00514],
    "THU_VBF_Mjj700": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.006846,-0.006847,0.004344,0.004345,0.004346,0.004345,0.004347,0.004345,0.004345,0.004338,0.004346,0.004342,0.00435,0.004349,0.004332,0.004342],
    "THU_VBF_Mjj1000": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.01006,-0.01006,0.004833,0.004831,0.004832,0.004832,0.004835,0.004828,0.004834,0.004837,0.00483,0.004835,0.004823,0.004831],
    "THU_VBF_Mjj1500": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.004264,-0.004264,0.003657,0.003658,0.003659,0.003659,0.003659,0.003658,0.003658,0.003656,0.003661,0.003658],
    "THU_VBF_PTH25": [0.0,0.0,0.0,-0.03153,-0.03153,-0.03153,0.06041,0.04633,0.01925,-0.03154,0.01201,-0.03154,0.01036,-0.03153,0.009281,-0.03155,0.006915,-0.03157,0.02988,-0.03148,0.0216,-0.03155,0.0193,-0.03148,0.01418],
    "THU_VBF_JET01": [0.0,-0.01025,-0.01025,0.008923,0.008921,0.008906,0.008932,0.008915,0.008909,0.008913,0.008907,0.008899,0.008909,0.008912,0.008913,0.008922,0.008911,0.00888,0.008922,0.008977,0.00894,0.008853,0.008947,0.008907,0.008885],
}

dymodel_DNN_reshape = {
    "2016":{"h_peak": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "h_sideband":(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        },
    "2017":{
        "h_peak": (1.008679284,
                   1.008443075,
                   0.9533933208,
                   1.000047493,
                   0.9497453485,
                   0.9087377694,
                   0.8660737525,
                   1.003580687,
                   0.9749485311,
                   0.6627367356,
                   0.7011913518,
                   0.7759534665,
                   0.522),
        "h_sideband": (1.037397278,
                       1.052804602,
                       1.007356839,
                       0.9940118619,
                       0.9981218343,
                       0.9114297911,
                       0.9382380765,
                       0.9274733396,
                       0.7535121328,
                       0.8683631121,
                       0.7424364422,
                       0.7416515127,
                       0.7673167661),
    },
    "2018":{
        "h_peak": (0.9951012233,
                   1.016147681,
                   1.009796371,
                   1.017391335,
                   0.8998916965,
                   1.003134189,
                   0.9382245025,
                   0.9398089067,
                   0.9238255108,
                   1.354468107,
                   1.302041425,
                   1.383144437,
                   1.94),
        "h_sideband":(
            1.028935383,
            1.056019234,
            0.9842308641,
            0.9530115489,
            0.9284349217,
            0.9654664347,
            0.9136553811,
            0.8263961795,
            0.9406409542,
            0.7948182252,
            0.9696701881,
            0.9762267725,
            1.578297804),
    }
}
lhe_pdf_variations ={
    "2016":103,
    "2017":33,
    "2018":33
}
data_runs = {
    "2017": [
        (294927, 297019, "RunA"),
        (297020, 299329, "RunB"),
        (299337, 302029, "RunC"),
        (302030, 303434, "RunD"),
        (303435, 304826, "RunE"),
        (304911, 306462, "RunF")
    ],

    "2016": [
        (272007, 275376, "RunB"),  
        (275657, 276283, "RunC"),  
        (276315, 276811, "RunD"),  
        (276831, 277420, "RunE"),  
        (277772, 278808, "RunF"),  
        (278820, 280385, "RunG"),  
        (280919, 284044, "RunH"),  
    ],

    "2018": [
        (315252, 316995, "RunA"),
        (316998, 319312, "RunB"),
        (319313, 320393, "RunC"),
        (320394, 325273, "RunD"),
        (325274, 325765, "RunE"),
    ]
}

#Attach numerical ID to each run name
runmap_numerical = {
    "RunA": 0,
    "RunB": 1,
    "RunC": 2,
    "RunD": 3,
    "RunE": 4,
    "RunF": 5,
    "RunG": 6,
    "RunH": 7,
}

#reversed runmap
runmap_numerical_r = {v: k for k, v in runmap_numerical.items()}

#Used to scale the genweight to prevent a numerical overflow
genweight_scalefactor = 1e-5

catnames = {
    "dimuon_invmass_z_peak_cat5": "dimuons, Z region, cat 5",
    "dimuon_invmass_h_peak_cat5": "dimuons, H SR, cat 5",
    "dimuon_invmass_h_sideband_cat5": "dimuons, H SB, cat 5",

    "dimuon_invmass_z_peak": "dimuons, Z region",
    "dimuon_invmass_h_peak": "dimuons, H SR",
    "dimuon_invmass_h_sideband": "dimuons, H SB",

    "dnn_presel": r"dimuons, $\geq 2$ jets",
    "dimuon": "dimuons",
}


varnames = {
    "Higgs_eta": "$\eta_{\mu\mu}$",
    "Higgs_mass": "$M_{\mu\mu}$",
    "MET_pt": "MET [GeV]",
    "M_jj": "dijet invariant mass [GeV]",
    "M_mmjj": "$M_{\mu\mu j_1 j_2}$",
    "cthetaCS": "$\cos \theta_{CS}$",
    "dEta_jj": "$\Delta \eta(j_1 j_2)$",
    "dEtamm": "$\Delta \eta (\mu \mu)$",
    "dPhimm": "$\Delta \phi(j_1 j_2)",
    "dRmin_mj": "min $\Delta R (\mu j)$",
    "dijet_inv_mass": "dijet invariant mass $M_{jj} [GeV]",
    "dnn_pred2": "signal DNN", 
    "dnnPisa_pred_atanh": "signal Pisa DNN",
    "eta_mmjj": "$\eta_{\mu\mu j_1 j_2}$",
    "hmmthetacs": "$\theta_{CS}$",
    "inv_mass": "$M_{\mu\mu}$",
    "leadingJet_eta": "leading jet $\eta$",
    "leadingJet_pt": "leading jet $p_T$ [GeV]",
    "leading_jet_eta": "leading jet $\eta$",
    "leading_jet_pt": "leading jet $p_T$ [GeV]",
    "leading_jet_pt": "leading jet $p_T$",
    "num_jets": "number of jets",
    "phi_mmjj": "$\phi(\mu\mu,j_1 j_2)$", 
    "pt_balance": "$p_{T,\mu\mu} / p_{T,jj}$",
    "pt_jj": "dijet $p_T$ [GeV]",
    "softJet5": "number of soft EWK jets",
    "subleadingJet_eta": "subleading jet $\eta$",
    "subleadingJet_pt": "subleading jet $p_T$ [GeV]",
    "subleadingJet_qgl": "subleading jet QGL",
    "subleading_jet_pt": "subleading jet $p_T$ [GeV]",
}

analysis_names = {
    "baseline": {"2018": "Autumn18_V19", "2017": "Fall17_17Nov2017_V32", "2016": "Summer16_07Aug2017_V11"},
    "jer": {"2018": "nominal: JER smearing off", "2017": "nominal: JER smearing off", "2016": "nominal: JER smearing off"},
}

#All analysis definitions (cut values etc) should go here
analysis_parameters = {
    "baseline": {

        "nPV": 0,
        "NdfPV": 4,
        "zPV": 24,

        # Will be applied with OR
        "hlt_bits": {
            "2016": ["HLT_IsoMu24", "HLT_IsoTkMu24"],
            "2017": ["HLT_IsoMu27"],
            "2018": ["HLT_IsoMu24"],
            },

        "muon_pt": 20,
        "muon_pt_leading": {"2016": 26.0, "2017": 29.0, "2018": 26.0},
        "muon_eta": 2.4,
        "muon_iso": 0.25,
        "muon_id": {"2016": "medium", "2017": "medium", "2018": "medium"},
        "muon_trigger_match_dr": 0.1,
        "muon_iso_trigger_matched": 0.15,
        "muon_id_trigger_matched": {"2016": "tight", "2017": "tight", "2018": "tight"},

        "do_rochester_corrections": True, 
        "do_lepton_sf": True,
        "do_geofit": True,
        #re-apply JEC (True) or use jet pT directly from nanoAOD (False) 
        "do_jec": {"2016": False, "2017": True, "2018": True},
        "do_jer": {"2016": True, "2017": True, "2018": True},
        "jer_pt_eta_bins" : {
            "jerB1": {"eta" : [0,1.93], "pt" : [0.0,10000.0]},
            "jerB2": {"eta" : [1.93,2.5], "pt" : [0.0,10000.0]},
            "jerEC1": {"eta" : [2.5,3.1], "pt" : [0.0,50.]},
            "jerEC2": {"eta" : [2.5,3.1], "pt" : [50.0,10000.0]},
            "jerF1": {"eta" : [3.1,10.0], "pt" : [0.0,50.0]},
            "jerF2": {"eta" : [3.1,10.0], "pt" : [50.0,10000.0]},
        },
        "split_z_peak":{"2016": False, "2017": False, "2018": False},
        "jec_tag": {"2016": "Summer16_07Aug2017_V11", "2017": "Fall17_17Nov2017_V32", "2018": "Autumn18_V19"}, 
        "jet_mu_dr": 0.4,
        "jet_pt_leading": {"2016": 35.0, "2017": 35.0, "2018": 35.0},
        "jet_pt_subleading": {"2016": 25.0, "2017": 25.0, "2018": 25.0},
        "jet_eta": 4.7,
        "jet_id": {"2016":"loose", "2017":"tight", "2018":"tight"},
        "jet_puid": "loose",
        "jet_puid_pt_max": 50,
        "jet_veto_eta": [2.6, 3.0],
        "jet_veto_raw_pt": 50.0,  
        "jet_btag_medium": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},
        "jet_btag_loose": {"2016": 0.2217, "2017": 0.1522, "2018": 0.1241},
        "do_factorized_jec": True,
        "apply_btag": True,
        "softjet_pt5": 5.0,
        "softjet_pt2": 2.0,
        "softjet_evt_dr2": 0.16, 

        "fsr_dROverEt2": 0.012,
        "fsr_relIso03": 1.8,
        "pt_fsr_over_mu_e": 0.4,
 
        "cat5_dijet_inv_mass": 400.0,
        "cat5_abs_jj_deta_cut": 2.5,

        "masswindow_z_peak": [76, 106],
        "masswindow_h_sideband": [110, 150],
        "masswindow_h_peak": [115.03, 135.03],
        "masswindow_z_peak_jerB1": [76, 106],
        "masswindow_z_peak_jerB2": [76, 106],
        "masswindow_z_peak_jerEC1": [76, 106],
        "masswindow_z_peak_jerEC2": [76, 106],
        "masswindow_z_peak_jerF1": [76, 106],
        "masswindow_z_peak_jerF2": [76, 106],
        "inv_mass_bins": 41,

        "extra_electrons_pt": 20,
        "extra_electrons_eta": 2.5,
        "extra_electrons_iso": 0.4, #Check if we want to apply this
        "extra_electrons_id": "mvaFall17V2Iso_WP90",

        "save_dnn_vars": True,
        "dnn_vars_path": "out/dnn_vars",
        #If true, apply mjj > cut, otherwise inverse
        "vbf_filter_mjj_cut": 350,
        "vbf_filter": {
            "dy_m105_160_mg_01j": True,
            "dy_m105_160_amc_01j": True,
            "dy_m105_160_mg_2j": True,
            "dy_m105_160_amc_2j": True,
            "dy_m105_160_vbf_mg_01j": False,
            "dy_m105_160_vbf_amc_01j": False, 
            "dy_m105_160_vbf_mg_2j": False,
            "dy_m105_160_vbf_amc_2j": False,
        },
        "ggh_nnlops_reweight": {
            "ggh_amc_pythia_125": 1,
            "ggh_amcPS_pythia_125": 1,
            "ggh_powheg_pythia_125": 2,
            "ggh_powhegPS_pythia_125": 2,
        },
        "ZpT_reweight": {
            "2016": {
                #"dy_0j": 2, 
                #"dy_1j": 2, 
                #"dy_2j": 2, 
                #"dy_m105_160_amc": 2, 
                #"dy_m105_160_vbf_amc": 2,
            },
            "2017": {
                #"dy_0j": 1,
                #"dy_1j": 1,
                #"dy_2j": 1,
                #"dy_m105_160_amc": 1,
                #"dy_m105_160_vbf_amc": 1,
            },
            "2018": {
                #"dy_0j": 1,
                #"dy_1j": 1,
                #"dy_2j": 1,
                #"dy_m105_160_amc": 1,
                #"dy_m105_160_vbf_amc": 1,
            },
        },
        #Pisa Group's DNN input variable order for keras
        "dnnPisa_varlist1_order": ['Mqq_log','Rpt','qqDeltaEta','log(ll_zstar)','NSoft5','HTSoft2','minEtaHQ','CS_phi', 'CS_theta','Higgs_pt','log(Higgs_pt)','Higgs_eta','Mqq','QJet0_pt_touse','QJet1_pt_touse','QJet0_eta','QJet1_eta','QJet0_phi','QJet1_phi','QJet0_qgl','QJet1_qgl','year'],
        
        #Irene's DNN input variable order for keras
        "dnn_varlist_order": ['HTSoft5', 'dRmm','dEtamm','M_jj','pt_jj','eta_jj','phi_jj','M_mmjj','eta_mmjj','phi_mmjj','dEta_jj','Zep','minEtaHQ','minPhiHQ','dPhimm','leadingJet_pt','subleadingJet_pt','massErr_rel', 'leadingJet_eta','subleadingJet_eta','leadingJet_qgl','subleadingJet_qgl','cthetaCS','Higgs_pt','Higgs_eta','Higgs_m_125'],
        "dnn_input_histogram_bins": {
            "HTSoft5": (0,10,10),
            "dRmm": (0,5,11),
            "dEtamm": (-2,2,11),
            "dPhimm": (-2,2,11),
            "M_jj": (0,2000,41),
            "pt_jj": (0,400,41),
            "eta_jj": (-5,5,41),
            "phi_jj": (-5,5,41),
            "M_mmjj": (0,2000,11),
            #"eta_mmjj": (-3,3,11),
            #"phi_mmjj": (-3,3,11),
            "dEta_jj": (-3,3,11),
            "Zep": (-2,2,11),
            "minEtaHQ":(-5,5,11),
            "minPhiHQ":(-5,5,11),
            "leadingJet_pt": (0, 200, 41),
            "subleadingJet_pt": (0, 200, 41),
            "massErr_rel":(0,0.5,11),
            "leadingJet_eta": (-5, 5, 41),
            "subleadingJet_eta": (-5, 5, 41),
            "leadingJet_qgl": (0, 1, 41),
            "subleadingJet_qgl": (0, 1, 41),
            "cthetaCS": (-1, 1, 11),
            "Higgs_pt": (0, 200, 41),
            "Higgs_eta": (-3, 3, 41),
            "Higgs_mass": (110, 150, 41),
            "hmmthetacs": (-1, 1, 11),
            "hmmphics": (-4, 4, 11),
            "dnn_pred": (0, 1, 1001),
            #"bdt_ucsd": (-1, 1, 11),
            #"bdt2j_ucsd": (-1, 1, 11),
            #"bdt01j_ucsd": (-1, 1, 11)
        },

        "do_bdt_ucsd": False,
        "do_dnn_pisa": True,
        "do_dnn_cit": True,
    }, #end of baseline
}

#define the histogram binning
histo_bins = {
    "muon_pt": np.linspace(0, 200, 101, dtype=np.float32),
    "muon_eta": np.linspace(-2.5, 2.5, 21, dtype=np.float32),
    "npvs": np.linspace(0, 100, 101, dtype=np.float32),
    "dijet_inv_mass": np.linspace(0, 2000, 11, dtype=np.float32),
    "inv_mass": np.linspace(70, 150, 11, dtype=np.float32),
    "numjet": np.linspace(0, 10, 11, dtype=np.float32),
    "jet_pt": np.linspace(0, 300, 101, dtype=np.float32),
    "jet_eta": np.linspace(-4.7, 4.7, 11, dtype=np.float32),
    "pt_balance": np.linspace(0, 5, 1001, dtype=np.float32),
    "Rpt": np.linspace(0, 1, 11, dtype=np.float32),
    "numjets": np.linspace(0, 10, 11, dtype=np.float32),
    "jet_qgl": np.linspace(0, 1, 11, dtype=np.float32),
    "massErr": np.linspace(0, 10, 101, dtype=np.float32),
    "massErr_rel": np.linspace(0, 0.05, 101, dtype=np.float32),
    "DeepCSV": np.linspace(0, 1, 11, dtype=np.float32),

}

for imass in mass_point:
    histo_bins["dnnPisa_pred_"+str(imass)] = np.linspace(0,5,5001, dtype=np.float32)
    analysis_parameters["baseline"]["dnnPisa_varlist2_order_"+str(imass)] = ['Higgs_m_'+str(imass),'Higgs_mRelReso','Higgs_mReso_'+str(imass)]

for hname, bins in analysis_parameters["baseline"]["dnn_input_histogram_bins"].items():
    histo_bins[hname] = np.linspace(bins[0], bins[1], bins[2], dtype=np.float32)

for masswindow in ["z_peak", "h_peak", "h_sideband","z_peak_jerB1","z_peak_jerB2","z_peak_jerEC1","z_peak_jerEC2","z_peak_jerF1","z_peak_jerF2"]:
    mw = analysis_parameters["baseline"]["masswindow_" + masswindow]
    histo_bins["inv_mass_{0}".format(masswindow)] = np.linspace(mw[0], mw[1], 41, dtype=np.float32)

histo_bins["dnn_pred2"] = {
    "h_peak": np.array([0., 0.905, 0.915, 0.925, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965,0.97, 0.975,0.98, 0.985,1.0], dtype=np.float32),
    "z_peak": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "h_sideband": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "z_peak_jerB1": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "z_peak_jerB2": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "z_peak_jerEC1": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "z_peak_jerEC2": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "z_peak_jerF1": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    "z_peak_jerF2": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
}

histo_bins["dnnPisa_predf"] = {
    "h_peak": np.array([0.0, 0.797, 0.892, 0.939, 0.964, 0.976, 0.983, 0.988, 0.991, 0.994, 0.996, 0.998, 1.0], dtype=np.float32),
    "z_peak": np.array([0.0, 0.797, 0.892, 0.939, 0.964, 0.976, 0.983, 0.988, 0.991, 0.994, 0.996, 0.998, 1.0], dtype=np.float32),
    "h_sideband": np.array([0.0, 0.797, 0.892, 0.939, 0.964, 0.976, 0.983, 0.988, 0.991, 0.994, 0.996, 0.998, 1.0], dtype=np.float32),
}

histo_bins["dnnPisa_pred_atanh"] = {
    "2016":{
        "h_peak": np.array([0, 0.846666666667, 1.30333333333, 1.575, 1.815, 2.03833333333, 2.23666666667, 2.41, 2.575, 2.74, 2.91833333333, 3.12333333333, 3.4, 5.0], dtype=np.float32),
        "z_peak": np.array([0, 0.846666666667, 1.30333333333, 1.575, 1.815, 2.03833333333, 2.23666666667, 2.41, 2.575, 2.74, 2.91833333333, 3.12333333333, 3.4, 5.0], dtype=np.float32),
        "h_sideband": np.array([0, 0.846666666667, 1.30333333333, 1.575, 1.815, 2.03833333333, 2.23666666667, 2.41, 2.575, 2.74, 2.91833333333, 3.12333333333, 3.4, 5.0], dtype=np.float32),
    },
    "2017":{
        "h_peak": np.array([0, 0.846666666667, 1.30333333333, 1.575, 1.815, 2.03833333333, 2.23666666667, 2.41, 2.575, 2.74, 2.91833333333, 3.12333333333, 3.4, 5.0], dtype=np.float32),
        "z_peak": np.array([0, 0.846666666667, 1.30333333333, 1.575, 1.815, 2.03833333333, 2.23666666667, 2.41, 2.575, 2.74, 2.91833333333, 3.12333333333, 3.4, 5.0], dtype=np.float32),
        "h_sideband": np.array([0, 0.846666666667, 1.30333333333, 1.575, 1.815, 2.03833333333, 2.23666666667, 2.41, 2.575, 2.74, 2.91833333333, 3.12333333333, 3.4, 5.0], dtype=np.float32),

    },
    "2018":{
        "h_peak": np.array([0, 0.575, 1.30833333333, 1.67, 1.965, 2.21666666667, 2.42166666667, 2.59166666667, 2.75333333333, 2.91666666667, 3.08666666667, 3.27833333333, 3.52333333333, 5.0], dtype=np.float32),

        "z_peak": np.array([0, 0.575, 1.30833333333, 1.67, 1.965, 2.21666666667, 2.42166666667, 2.59166666667, 2.75333333333, 2.91666666667, 3.08666666667, 3.27833333333, 3.52333333333, 5.0], dtype=np.float32),

        "h_sideband": np.array([0, 0.575, 1.30833333333, 1.67, 1.965, 2.21666666667, 2.42166666667, 2.59166666667, 2.75333333333, 2.91666666667, 3.08666666667, 3.27833333333, 3.52333333333, 5.0], dtype=np.float32),
    },
}
analysis_parameters["baseline"]["histo_bins"] = histo_bins
