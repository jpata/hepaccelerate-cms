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
    "dy": 2075.14*3, # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SummaryTable1G25ns; Pg 10: https://indico.cern.ch/event/746829/contributions/3138541/attachments/1717905/2772129/Drell-Yan_jets_crosssection.pdf 
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
    "vbf_powheg_herwig": 0.000823,
    "vbf_powheg1": 0.000823,
    "vbf_powheg2": 0.000823,
    "vbf_amc_herwig": 0.000823,
    "vbf_amc1": 0.000823,
    "vbf_amc2": 0.000823,
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
    'AbsoluteFlavMap', 'AbsoluteMPFBias', 'AbsoluteSample', 'AbsoluteScale',
    'AbsoluteStat',
#These can be used as a proxy for all the groups
    #'CorrelationGroupFlavor', 'CorrelationGroupIntercalibration',
    #'CorrelationGroupMPFInSitu', 'CorrelationGroupUncorrelated', 'CorrelationGroupbJES',

#These are overlapping, the one closest to our region of interest should be chosen
    #'FlavorPhotonJet', 'FlavorPureBottom', 'FlavorPureCharm', 'FlavorPureGluon',
    #'FlavorPureQuark', 'FlavorQCD',
    'FlavorZJet',

    'Fragmentation', 'PileUpDataMC',
    'PileUpEnvelope', 'PileUpMuZero', 'PileUpPtBB', 'PileUpPtEC1', 'PileUpPtEC2',
    'PileUpPtHF', 'PileUpPtRef', 'RelativeBal', 'RelativeFSR', 'RelativeJEREC1',
    'RelativeJEREC2', 'RelativeJERHF', 'RelativePtBB', 'RelativePtEC1', 'RelativePtEC2',
    'RelativePtHF', 'RelativeSample', 'RelativeStatEC', 'RelativeStatFSR', 'RelativeStatHF',
    'SinglePionECAL', 'SinglePionHCAL']
#jec_unc = ["Total"]
#These subtotals can be used for cross-checks
#, 'SubTotalAbsolute', 'SubTotalMC', 'SubTotalPileUp',
#    'SubTotalPt', 'SubTotalRelative', 'SubTotalScale', 'TimePtEta', 'Total', 'TotalNoFlavor',
#    'TotalNoFlavorNoTime', 'TotalNoTime']

jec_unc = ["Total"]
shape_systematics = jec_unc + ["jer", "trigger", "id", "iso", "puWeight", "L1PreFiringWeight"]
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
    "dy_m105_160_amc": {"DYxsec": 1.10},
    "dy_m105_160__vbf_amc": {"DYxsec": 1.10},
    "ewk_lljj_mll105_160": {"EWZxsec": 1.20},
    "ewk_lljj_mll50_mjj120": {"EWZxsec": 1.20},
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
    "oldjec": {"2018": "Autumn18_V8", "2017": "", "2016": ""},
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

    ("vbf", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_powheg1", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("vbf_powheg2", "2016", "/store/mc/RunIISummer16NanoAODv5/VBF_HToMuMu_M125_13TeV_powheg_pythia8/**/*.root", True),    
    ("vbf_amc_herwig", "2016","/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/**/*.root",True),
    ("vbf_powheg_herwig", "2016","/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/**/*.root",True),
    ("vbf", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("vbf_amc1", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_amc2", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M125_13TeV_amcatnlo_pythia8/**/*.root", True),

    ("vbf_amc_herwig", "2017","/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/**/*.root", True),
    ("vbf_powheg_herwig", "2017","/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/**/*.root", True),
    ("vbf", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf_powheg1", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    
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
    ("ewk_lljj_mll50_mjj120", "2017", "/store/mc/RunIIFall17NanoAODv5/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_*_13TeV-madgraph-pythia8/**/*.root", True),
    ("ewk_lljj_mll50_mjj120", "2018", "/store/mc/RunIIAutumn18NanoAODv5/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8//**/*.root", True),

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
    ("ggh", "2016", "data/ggh_nano_2016.root", True)
]
