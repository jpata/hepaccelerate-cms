import numpy as np

categories = {
    "dimuon": {
        "datacard_processes" : [
            "ggh_amcPS",
            "vbf",
            "wmh",
            "wph",
            "zh",
            "tth",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120",
            #"ewk_lljj_mll105_160",
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
            "ggh_amcPS",
            "vbf",
            "wmh",
            "wph",
            "zh",
            "tth",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120",
            #"ewk_lljj_mll105_160",
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
            "ggh_amcPS",
            "vbf",
            "wmh",
            "wph",
            "zh",
            "tth",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            #"ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_m105_160_amc", "dy_m105_160_vbf_amc",
            "www","wwz","wzz","zzz",
        ],
    },
    "h_peak": {
        "datacard_processes" : [
            "ggh_amcPS",
            "vbf",
            "wmh",
            "wph",
            "zh",
            "tth",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            #"ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_m105_160_amc", "dy_m105_160_vbf_amc",
            "www","wwz","wzz","zzz",
        ],
    }
}
proc_grps = [
        ("vh",["wmh", "wph", "zh"]),
        ("vv", ["wz_3lnu", "ww_2l2nu", "wz_2l2q", "zz"]),
        ("vvv", ["www","wwz","wzz","zzz"]),
        ("stop", ["st_tw_top", "st_tw_antitop"]),
        ("tt", ["ttjets_sl", "ttjets_dl",]),
    ]
combined_signal_samples= ["ggh_amcPS", "vbf", "vh", "tth"]
combined_categories = {
    "dimuon": {
        "datacard_processes" : [
            "ggh_amcPS",
            "vbf",
            "vh",
            "tth",
            #"wz_1l1nu2q",
            "vv", 
            "ewk_lljj_mll50_mjj120",
            #"ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "stop",
            "tt",
            "dy",
            "vvv",
        ],
    },
    "z_peak": {
        "datacard_processes" : [
            "ggh_amcPS",
            "vbf",
            "vh",
            "tth",
            #"wz_1l1nu2q",
            "vv",
            "ewk_lljj_mll50_mjj120",
            #"ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "stop",
            "tt",
            "dy_0j", "dy_1j", "dy_2j",
            "vvv",
        ],
    },
    "h_sideband": {
        "datacard_processes" : [
            "ggh_amcPS",
            "vbf",
            "vh",
            "tth",
            #"wz_1l1nu2q",
            "vv", 
            #"ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "stop",
            "tt",
            "dy_m105_160_amc", "dy_m105_160_vbf_amc",
            "vvv",
        ],
    },
    "h_peak": {
        "datacard_processes" : [
            "ggh_amcPS",
            "vbf",
            "vh",
            "tth",
            #"wz_1l1nu2q",
            "vv", 
            #"ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "stop",
            "tt",
            "dy_m105_160_amc", "dy_m105_160_vbf_amc",
            "vvv",
        ],
    }
}

colors = {
    "dy": (254, 254, 83),
    "ewk": (109, 253, 245),
    "stop": (236, 76, 105),
    "tt": (67, 150, 42),
    "vvv": (247, 206, 205),
    "vv": (100, 105, 98),
    "higgs": (0, 0, 0),
}

process_groups = [
    ("higgs", ["ggh_amcPS", "vbf", "wmh", "wph", "zh", "tth"]),
    ("vv", ["wz_3lnu", "ww_2l2nu", "wz_2l2q", "zz"]),
    ("vvv", ["www","wwz","wzz","zzz"]),
    ("ewk", ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160"]),
    ("stop", ["st_tw_top", "st_tw_antitop"]),
    ("tt", ["ttjets_sl", "ttjets_dl",]),
    ("dy", ["dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc", "dy"]),
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
    "inv_mass"
]

cross_sections = {
    "dy": 2026.96*3, #https://indico.cern.ch/event/841566/contributions/3565385/attachments/1914850/3165328/Drell-Yan_jets_crosssection_September2019.pdf 
    "dy_0j": 4620.52, #https://indico.cern.ch/event/673253/contributions/2756806/attachments/1541203/2416962/20171016_VJetsXsecsUpdate_PH-GEN.pdf
    "dy_1j": 859.59,
    "dy_2j": 338.26,
    "dy_m105_160_mg": 46.9479,
    "dy_m105_160_vbf_mg": 2.02,
    "dy_m105_160_amc": 46.9479, # https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "dy_m105_160_vbf_amc": 46.9479*0.0425242, #https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "ggh_powheg": 0.010571, #48.61 * 0.0002176; https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNHLHE2019
    "ggh_powhegPS": 0.010571,
    "ggh_amcPS": 0.010571,
    "ggh_amcPS_TuneCP5down": 0.010571,
    "ggh_amcPS_TuneCP5up": 0.010571,
    "ggh_amc": 0.010571,
    "vbf": 0.000823,
    "vbf_sync": 0.000823,
    "vbf_powheg_herwig": 0.000823,
    "vbf_powheg": 0.000823,
    "vbf_powhegPS": 0.000823,
    "vbf_amc_herwig": 0.000823,
    "vbf_amcPS_TuneCP5down": 0.000823,
    "vbf_amcPS_TuneCP5up": 0.000823,
    "vbf_amcPS": 0.000823,
    "vbf_amc": 0.000823,
    "wmh": 0.000116,
    "wph": 0.000183,
    "zh": 0.000192,
    "tth": 0.000110,
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
    "ewk_lljj_mll105_160": 0.0508896,

    # Note via Nan L.: the 2016 sample has a different tune, for which Stephane C.
    # computed a new cross-section from MINIAOD using
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToGenXSecAnalyzer
    "ewk_lljj_mll50_mjj120": {"2016": 1.611, "2017": 1.700, "2018": 1.700},

    "ttw": 0.2001,
    "ttz": 0.2529,
    "st_t_top": 3.36,
    "www": 0.2086,
    "wwz": 0.1651,
    "wzz": 0.05565,
    "zzz": 0.01398
}

signal_samples = ["ggh_amcPS", "vbf", "wmh", "wph", "zh", "tth"]
jec_unc = [
    #'AbsoluteFlavMap', 
    'AbsoluteMPFBias', 'AbsoluteSample', 'AbsoluteScale',
    'AbsoluteStat',
#These can be used as a proxy for all the groups
    #'CorrelationGroupFlavor', 'CorrelationGroupIntercalibration',
    #'CorrelationGroupMPFInSitu', 'CorrelationGroupUncorrelated', 'CorrelationGroupbJES',

#These are overlapping, the one closest to our region of interest should be chosen
    #'FlavorPhotonJet', 'FlavorPureBottom', 'FlavorPureCharm', 'FlavorPureGluon',
    #'FlavorPureQuark', 'FlavorQCD',
    'FlavorZJet',
    'TimePtEta',
    'Fragmentation', 'PileUpDataMC',
    #'PileUpEnvelope', 'PileUpMuZero',
    'PileUpPtBB', 'PileUpPtEC1', 'PileUpPtEC2',
    'PileUpPtHF', 'PileUpPtRef', 'RelativeBal', 'RelativeFSR', 'RelativeJEREC1',
    'RelativeJEREC2', 'RelativeJERHF', 'RelativePtBB', 'RelativePtEC1', 'RelativePtEC2',
    'RelativePtHF', 'RelativeSample', 'RelativeStatEC', 'RelativeStatFSR', 'RelativeStatHF',
    'SinglePionECAL', 'SinglePionHCAL']

#These subtotals can be used for cross-checks
#, 'SubTotalAbsolute', 'SubTotalMC', 'SubTotalPileUp',
#    'SubTotalPt', 'SubTotalRelative', 'SubTotalScale', 'Total', 'TotalNoFlavor',
#    'TotalNoFlavorNoTime', 'TotalNoTime']

#Uncomment to use just the total JEC for quick tests
#jec_unc = ["Total"]
shape_systematics = jec_unc + ["jer", "trigger", "id", "iso", "puWeight", "L1PreFiringWeight","DYLHEScaleWeight","EWZLHEScaleWeight"]
common_scale_uncertainties = {
    "lumi": 1.025,
}
scale_uncertainties = {
    "ww_2l2nu": {"VVxsec": 1.10},
    "wz_3lnu": {"VVxsec": 1.10},
    "wz_2l2q": {"VVxsec": 1.10},
    "wz_2l2q": {"VVxsec": 1.10},
    "zz": {"VVxsec": 1.10},
    "wjets": {"WJetsxsec": 1.10},
    "vv" :{"VVxsec": 1.10},
    "stop": {"STxsec": 1.05},
    "tt" : {"TTxsec": 1.05},
    #"dy_m105_160_amc": {"DYxsec": 1.10},
    #"dy_m105_160__vbf_amc": {"DYxsec": 1.10},
    #"ewk_lljj_mll105_160": {"EWZxsec": 1.20},
    #"ewk_lljj_mll50_mjj120": {"EWZxsec": 1.20},
    "ttjets_sl": {"TTxsec": 1.05},
    "ttjets_dl": {"TTxsec": 1.05},
    "st_t_top": {"STxsec": 1.05},
    "st_t_antitop": {"STxsec": 1.05},
    "st_tw_top": {"STxsec": 1.05},
    "st_tw_antitop": {"STxsec": 1.05},
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
    "baseline": {"2018": "Autumn18_V16", "2017": "Fall17_17Nov2017_V32", "2016": "Summer16_07Aug2017_V11"},
    "jer": {"2018": "JER enabled", "2017": "JER enabled", "2016": "JER enabled"},
}

# dataset nickname, datataking era, filename glob pattern, isMC
datasets = [
    ("ggh_amcPS", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS_TuneCP5down", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGluHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS_TuneCP5up", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGluHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_powheg", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/**/*.root", True),
    ("ggh_powhegPS", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),

    ("ggh_amc", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS_TuneCP5down", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS_TuneCP5up", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_powheg", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M-125_13TeV_powheg_pythia8/**/*.root", True),
    ("ggh_powhegPS", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),

    ("ggh_amcPS", "2018", "/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS_TuneCP5down", "2018", "/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_amcPS_TuneCP5up", "2018", "/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh_powhegPS", "2018", "/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),

    ("vbf", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/**/*.root", True),
    ("vbf_amcPS", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amcPS_TuneCP5up", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amcPS_TuneCP5down", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_powhegPS", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("vbf_powheg", "2016", "/store/mc/RunIISummer16NanoAODv5/VBF_HToMuMu_M125_13TeV_powheg_pythia8/**/*.root", True),    
    ("vbf_amc_herwig", "2016","/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/**/*.root",True),
    ("vbf_powheg_herwig", "2016","/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/**/*.root",True),

    ("vbf", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/**/*.root", True),
    ("vbf_powhegPS", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("vbf_amcPS", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amcPS_TuneCP5down", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amcPS_TuneCP5up", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amc", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M125_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amc_herwig", "2017","/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/**/*.root", True),
    ("vbf_powheg_herwig", "2017","/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/**/*.root", True),

    ("vbf", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_powhegPS", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("vbf_amcPS", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amcPS_TuneCP5down", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amcPS_TuneCP5up", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
            
    ("data", "2016", "/store/data/Run2016*/SingleMuon/NANOAOD/Nano1June2019*/**/*.root", False),
    ("data", "2017", "/store/data/Run2017*/SingleMuon/NANOAOD/Nano1June2019-v1/**/*.root", False),
    ("data", "2018", "/store/data/Run2018*/SingleMuon/NANOAOD/Nano1June2019-v1/**/*.root", False),

    ("dy", "2016", "/store/mc/RunIISummer16NanoAODv5/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("dy", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("dy", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    
    ("dy_0j", "2016", "/store/mc/RunIISummer16NanoAODv5/DYToLL_0J_13TeV-amcatnloFXFX-pythia8//**/*.root", True),
    ("dy_0j", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8///**/*.root", True),
    ("dy_0j", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8//**/*.root", True),
    
    ("dy_1j", "2016", "/store/mc/RunIISummer16NanoAODv5/DYToLL_1J_13TeV-amcatnloFXFX-pythia8//**/*.root", True),
    ("dy_1j", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8///**/*.root", True),
    ("dy_1j", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8//**/*.root", True),
    
    ("dy_2j", "2016", "/store/mc/RunIISummer16NanoAODv5/DYToLL_2J_13TeV-amcatnloFXFX-pythia8//**/*.root", True),
    ("dy_2j", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8///**/*.root", True),
    ("dy_2j", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8//**/*.root", True),

    ("dy_m105_160_amc", "2016", "/store/mc/RunIISummer16NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("dy_m105_160_amc", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("dy_m105_160_amc", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/**/*.root", True),

#wrong tune for 2016?
    ("dy_m105_160_vbf_amc", "2016", "/store/mc/RunIISummer16NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("dy_m105_160_vbf_amc", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("dy_m105_160_vbf_amc", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    
    ("dy_m105_160_mg", "2016", "/store/mc/RunIISummer16NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/**/*.root", True),
    ("dy_m105_160_mg", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/**/*.root", True),
    ("dy_m105_160_mg", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8//**/*.root", True),
    
    ("dy_m105_160_vbf_mg", "2016", "/store/mc/RunIISummer16NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/**/*.root", True),
    ("dy_m105_160_vbf_mg", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_13TeV-madgraphMLM-pythia8/**/*.root", True),
    ("dy_m105_160_vbf_mg", "2018", "/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8//**/*.root", True),
    
    ("ttjets_dl", "2016", "/store/mc/RunIISummer16NanoAODv5/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("ttjets_dl", "2017", "/store/mc/RunIIFall17NanoAODv5/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("ttjets_dl", "2018", "/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

    ("ttjets_sl", "2016", "/store/mc/RunIISummer16NanoAODv5/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("ttjets_sl", "2017", "/store/mc/RunIIFall17NanoAODv5/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("ttjets_sl", "2018", "/store/mc/RunIIAutumn18NanoAODv5/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

    ("ttw", "2016", "/store/mc/RunIISummer16NanoAODv5/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/**/*.root", True),
    ("ttw", "2017", "/store/mc/RunIIFall17NanoAODv5/TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8/**/*.root", True),
    ("ttw", "2018", "/store/mc/RunIIAutumn18NanoAODv5/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/**/*.root", True),

    ("ttz", "2016", "/store/mc/RunIISummer16NanoAODv5/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("ttz", "2017", "/store/mc/RunIIFall17NanoAODv5/TTZToLLNuNu_M-10_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("ttz", "2018", "/store/mc/RunIIAutumn18NanoAODv5/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),

    ("ewk_lljj_mll50_mjj120", "2016", "/store/mc/RunIISummer16NanoAODv5/EWK_LLJJ_MLL-50_MJJ-120_13TeV-madgraph-herwigpp/**/*.root", True),
    ("ewk_lljj_mll50_mjj120", "2017", "/store/mc/RunIIFall17NanoAODv5/EWK_LLJJ_MLL-50_MJJ-120_TuneCH3_PSweights_13TeV-madgraph-herwig7/**/*.root", True),
    ("ewk_lljj_mll50_mjj120", "2018", "/store/mc/RunIIAutumn18NanoAODv5/EWK_LLJJ_MLL-50_MJJ-120_TuneCH3_PSweights_13TeV-madgraph-herwig7/**/*.root", True),

    ("ewk_lljj_mll105_160", "2016", "/store/mc/RunIISummer16NanoAODv5/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneEEC5_13TeV-madgraph-herwigpp/**/*.root", True),
    ("ewk_lljj_mll105_160", "2017", "/store/mc/RunIIFall17NanoAODv5/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7/**/*.root", True),
    ("ewk_lljj_mll105_160", "2018", "/store/mc/RunIIAutumn18NanoAODv5/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7/**/*.root", True),

    ("ww_2l2nu", "2016", "/store/mc/RunIISummer16NanoAODv5/WWTo2L2Nu_13TeV-powheg/**/*.root", True),
    ("ww_2l2nu", "2017", "/store/mc/RunIIFall17NanoAODv5/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("ww_2l2nu", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

    ("wz_3lnu", "2016", "/store/mc/RunIISummer16NanoAODv5/WZTo3LNu_TuneCUETP8M1_13TeV-powheg-pythia8/**/*.root", True),
    ("wz_3lnu", "2017", "/store/mc/RunIIFall17NanoAODv5/WZTo3LNu_13TeV-powheg-pythia8/**/*.root", True),
    ("wz_3lnu", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    
    ("wz_2l2q", "2016", "/store/mc/RunIISummer16NanoAODv5/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
    ("wz_2l2q", "2017", "/store/mc/RunIIFall17NanoAODv5/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
    ("wz_2l2q", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),

#2016 ST_t-channel not available yet
#https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=%2FST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8*%2F*NanoAODv5*%2FNANOAODSIM
#   # ("st_top", "2016", "/store/mc/RunIISummer16NanoAODv5//**/*.root", True),
    ("st_t_top", "2017", "/store/mc/RunIIFall17NanoAODv5/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("st_t_top", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

#2016 ST_t-channel not available yet
#https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=%2FST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8*%2F*NanoAODv5*%2FNANOAODSIM
#   # ("st_t_antitop", "2016", "/store/mc/RunIISummer16NanoAODv5//**/*.root", True),
    ("st_t_antitop", "2017", "/store/mc/RunIIFall17NanoAODv5/ST_t-channel_antitop_5f_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("st_t_antitop", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

    ("st_tw_antitop", "2016", "/store/mc/RunIISummer16NanoAODv5/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("st_tw_antitop", "2017", "/store/mc/RunIIFall17NanoAODv5/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("st_tw_antitop", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

    ("st_tw_top", "2016", "/store/mc/RunIISummer16NanoAODv5/ST_tW_top_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("st_tw_top", "2017", "/store/mc/RunIIFall17NanoAODv5/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("st_tw_top", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),

#2018 WZTo1L1Nu2Q_13TeV not available
#https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=%2FWZTo1L1Nu2Q_13TeV_amcatnloFXFX*%2F*NanoAODv5*%2FNANOAODSIM
   ("wz_1l1nu2q", "2016", "/store/mc/RunIISummer16NanoAODv5/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
    ("wz_1l1nu2q", "2017", "/store/mc/RunIIFall17NanoAODv5/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
#    ("wz_1l1nu2q", "2018", "/store/mc/RunIIAutumn18NanoAODv5//**/*.root", True),

    ("zz", "2016", "/store/mc/RunIISummer16NanoAODv5/ZZ_TuneCUETP8M1_13TeV-pythia8/**/*.root", True),
    ("zz", "2017", "/store/mc/RunIIFall17NanoAODv5/ZZ_TuneCP5_13TeV-pythia8/**/*.root", True),
    ("zz", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ZZ_TuneCP5_13TeV-pythia8/**/*.root", True),

    ("wmh", "2016", "/store/mc/RunIISummer16NanoAODv5/WminusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("wmh", "2017", "/store/mc/RunIIFall17NanoAODv5/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/**/*.root", True),
    ("wmh", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WminusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),

    ("wph", "2016", "/store/mc/RunIISummer16NanoAODv5/WplusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("wph", "2017", "/store/mc/RunIIFall17NanoAODv5/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/**/*.root", True),
    ("wph", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WplusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),

    ("tth", "2016", "/store/mc/RunIISummer16NanoAODv5/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("tth", "2017", "/store/mc/RunIIFall17NanoAODv5/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("tth", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),

    ("zh", "2016", "/store/mc/RunIISummer16NanoAODv5/ZH_HToMuMu_ZToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("zh", "2017", "/store/mc/RunIIFall17NanoAODv5/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/**/*.root", True),
    ("zh", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ZH_HToMuMu_ZToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    
    ("www", "2016", "/store/mc/RunIISummer16NanoAODv5/WWW_4F_TuneCUETP8M1_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("www", "2017", "/store/mc/RunIIFall17NanoAODv5/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("www", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    
    ("wwz", "2016", "/store/mc/RunIISummer16NanoAODv5/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("wwz", "2017", "/store/mc/RunIIFall17NanoAODv5/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("wwz", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WWZ_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    
    ("wzz", "2016", "/store/mc/RunIISummer16NanoAODv5/WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("wzz", "2017", "/store/mc/RunIIFall17NanoAODv5/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("wzz", "2018", "/store/mc/RunIIAutumn18NanoAODv5/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    
    ("zzz", "2016", "/store/mc/RunIISummer16NanoAODv5/ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("zzz", "2017", "/store/mc/RunIIFall17NanoAODv5/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True),
    ("zzz", "2018", "/store/mc/RunIIAutumn18NanoAODv5/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/**/*.root", True)
]

# Synchronization datasets/
datasets_sync = [
    #("ggh", "2016", "data/ggh_nano_2016.root", True)
    #("vbf_sync", "2016", "data/vbf_sync_2016.root",True)
    ("vbf_sync", "2016", "data/myNanoProdMc2016_NANO.root",True)
]
    
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
        
        "do_jec": True,
        "do_jer": {"2016": False, "2017": True, "2018": True},
        "jec_tag": {"2016": "Summer16_07Aug2017_V11", "2017": "Fall17_17Nov2017_V32", "2018": "Autumn18_V16"}, 
        "jet_mu_dr": 0.4,
        "jet_pt_leading": {"2016": 35.0, "2017": 35.0, "2018": 35.0},
        "jet_pt_subleading": {"2016": 25.0, "2017": 25.0, "2018": 25.0},
        "jet_eta": 4.7,
        "jet_id": {"2016":"loose", "2017":"tight", "2018":"tight"},
        "jet_puid": "loose",
        "jet_veto_eta": [2.65, 3.139],
        "jet_veto_raw_pt": 50.0,  
        "jet_btag_medium": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},
        "jet_btag_loose": {"2016": 0.2217, "2017": 0.1522, "2018": 0.1241},
        "do_factorized_jec": True,

        "softjet_pt": 5.0,
        "softjet_evt_dr2": 0.16, 

        "cat5_dijet_inv_mass": 400.0,
        "cat5_abs_jj_deta_cut": 2.5,

        "masswindow_z_peak": [76, 106],
        "masswindow_h_sideband": [110, 150],
        "masswindow_h_peak": [115, 135],

        "inv_mass_bins": 41,

        "extra_electrons_pt": 20,
        "extra_electrons_eta": 2.5,
        "extra_electrons_iso": 0.4, #Check if we want to apply this
        "extra_electrons_id": "mvaFall17V1Iso_WP90",

        "save_dnn_vars": True,
        "dnn_vars_path": "out/dnn_vars",

        #If true, apply mjj > cut, otherwise inverse
        "vbf_filter_mjj_cut": 350,
        "vbf_filter": {
            "dy_m105_160_mg": True,
            "dy_m105_160_amc": True,
            "dy_m105_160_vbf_mg": False,
            "dy_m105_160_vbf_amc": False, 
        },
        "ggh_nnlops_reweight": {
            "ggh_amc": 1,
            "ggh_amcPS": 1,
            "ggh_amcPS_TuneCP5down": 1,
            "ggh_amcPS_TuneCP5up": 1,
            "ggh_powheg": 2,
            "ggh_powhegPS": 2,
        },
        "ZpT_reweight": {
            "2016": {
                "dy_0j": 2, 
                "dy_1j": 2, 
                "dy_2j": 2, 
                "dy_m105_160_amc": 2, 
                "dy_m105_160_vbf_amc": 2,
            },
            "2017": {
                "dy_0j": 1,
                "dy_1j": 1,
                "dy_2j": 1,
                "dy_m105_160_amc": 1,
                "dy_m105_160_vbf_amc": 1,
            },
            "2018": {
                "dy_0j": 1,
                "dy_1j": 1,
                "dy_2j": 1,
                "dy_m105_160_amc": 1,
                "dy_m105_160_vbf_amc": 1,
            },
        },
       
        #Pisa Group's DNN input variable order for keras
        "dnnPisa_varlist1_order": ['Mqq_log','Rpt','qqDeltaEta','ll_zstar','NSoft5','minEtaHQ','Higgs_pt','log(Higgs_pt)','Higgs_eta','Mqq','QJet0_pt_touse','QJet1_pt_touse','QJet0_eta','QJet1_eta','QJet0_phi','QJet1_phi','QJet0_qgl','QJet1_qgl'],
        "dnnPisa_varlist2_order": ['Higgs_m','Higgs_mRelReso','Higgs_mReso'],
        #Irene's DNN input variable order for keras
        "dnn_varlist_order": ['HTSoft5', 'dRmm','dEtamm','M_jj','pt_jj','eta_jj','phi_jj','M_mmjj','eta_mmjj','phi_mmjj','dEta_jj','Zep','minEtaHQ','minPhiHQ','dPhimm','leadingJet_pt','subleadingJet_pt','massErr_rel', 'leadingJet_eta','subleadingJet_eta','leadingJet_qgl','subleadingJet_qgl','cthetaCS','Higgs_pt','Higgs_eta','Higgs_mass'],
        "dnn_input_histogram_bins": {
            "HTSoft5": (0,10,10),
            "dRmm": (0,5,11),
            "dEtamm": (-2,2,11),
            "dPhimm": (-2,2,11),
            "M_jj": (0,2000,11),
            "pt_jj": (0,400,11),
            "eta_jj": (-5,5,11),
            "phi_jj": (-5,5,11),
            "M_mmjj": (0,2000,11),
            "eta_mmjj": (-3,3,11),
            "phi_mmjj": (-3,3,11),
            "dEta_jj": (-3,3,11),
            "Zep": (-2,2,11),
            "minEtaHQ":(-5,5,11),
            "minPhiHQ":(-5,5,11),
            "leadingJet_pt": (0, 200, 11),
            "subleadingJet_pt": (0, 200, 11),
            "massErr_rel":(0,0.5,11),
            "leadingJet_eta": (-5, 5, 11),
            "subleadingJet_eta": (-5, 5, 11),
            "leadingJet_qgl": (0, 1, 11),
            "subleadingJet_qgl": (0, 1, 11),
            "cthetaCS": (-1, 1, 11),
            "Higgs_pt": (0, 200, 11),
            "Higgs_eta": (-3, 3, 11),
            "Higgs_mass": (110, 150, 11),
            "dnn_pred": (0, 1, 1001),
            "dnn_pred2": (0, 1, 11),
            "bdt_ucsd": (-1, 1, 11),
            "bdt2j_ucsd": (-1, 1, 11),
            "bdt01j_ucsd": (-1, 1, 11),
            "MET_pt": (0, 200, 11),
            "hmmthetacs": (-1, 1, 11),
            "hmmphics": (-4, 4, 11),
        },

        "categorization_trees": {},
        "do_bdt_ucsd": False,
        "do_dnn_pisa": False,
    },
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
    "pt_balance": np.linspace(0, 5, 11, dtype=np.float32),
    "numjets": np.linspace(0, 10, 11, dtype=np.float32),
    "jet_qgl": np.linspace(0, 1, 11, dtype=np.float32),
    "massErr": np.linspace(0, 10, 101, dtype=np.float32),
    "massErr_rel": np.linspace(0, 0.05, 101, dtype=np.float32)
}
for hname, bins in analysis_parameters["baseline"]["dnn_input_histogram_bins"].items():
    histo_bins[hname] = np.linspace(bins[0], bins[1], bins[2], dtype=np.float32)

for masswindow in ["z_peak", "h_peak", "h_sideband"]:
    mw = analysis_parameters["baseline"]["masswindow_" + masswindow]
    histo_bins["inv_mass_{0}".format(masswindow)] = np.linspace(mw[0], mw[1], 41, dtype=np.float32)

histo_bins["dnn_pred2"] = {
    "h_peak": np.array([0., 0.905, 0.915, 0.925, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965,0.97, 0.975,0.98, 0.985,1.0], dtype=np.float32),
    "z_peak": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], dtype=np.float32),
    "h_sideband": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], dtype=np.float32),
}

analysis_parameters["baseline"]["histo_bins"] = histo_bins
