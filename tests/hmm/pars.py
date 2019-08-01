categories = {
    "dimuon": {
        "datacard_processes" : [
            "ggh", "dy",
        ],
        "plot_processes": {
            "dy": ["dy_0j", "dy_1j", "dy_2j"],
        }
    },
    "z_peak": {
        "datacard_processes" : [
            "ggh",
            "vbf",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_0j", "dy_1j", "dy_2j",
        ],
        "plot_processes": {
            "dy": ["dy_0j", "dy_1j", "dy_2j"],
        }
    },
    "h_sideband": {
        "datacard_processes" : [
            "ggh",
            "vbf",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_m105_160_amc", "dy_m105_160_vbf_amc",
        ],
        "plot_processes": {
            "dy": ["dy_m105_160_amc", "dy_m105_160_vbf_amc"],
        }
    },
    "h_peak": {
        "datacard_processes" : [
            "ggh",
            "vbf",
            #"wz_1l1nu2q",
            "wz_3lnu", 
            "ww_2l2nu", "wz_2l2q", "zz",
            "ewk_lljj_mll50_mjj120",
            "ewk_lljj_mll105_160",
            #"st_top",
            #"st_t_antitop",
            "st_tw_top",
            "st_tw_antitop",
            "ttjets_sl", "ttjets_dl",
            "dy_m105_160_amc", "dy_m105_160_vbf_amc",
        ],
        "plot_processes": {
            "dy": ["dy_m105_160_amc", "dy_m105_160_vbf_amc"],
        }
    }
}

cross_sections = {
    "dy": 2075.14*3, # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SummaryTable1G25ns; Pg 10: https://indico.cern.ch/event/746829/contributions/3138541/attachments/1717905/2772129/Drell-Yan_jets_crosssection.pdf
    "dy_0j": 4620.52, #https://indico.cern.ch/event/673253/contributions/2756806/attachments/1541203/2416962/20171016_VJetsXsecsUpdate_PH-GEN.pdf
    "dy_1j": 859.59,
    "dy_2j": 338.26,
    "dy_m105_160_mg": 46.9479,
    "dy_m105_160_vbf_mg": 2.02,
    "dy_m105_160_amc": 46.9479, # https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "dy_m105_160_vbf_amc": 46.9479*0.0425242, #https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "ggh": 0.010571, #48.61 * 0.0002176; https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNHLHE2019
    "vbf": 0.000823,
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
    "ewk_lljj_mll50_mjj120": 0.128,
}

signal_samples = ["ggh", "vbf"]
jec_unc = ['AbsoluteFlavMap', 'AbsoluteMPFBias', 'AbsoluteSample', 'AbsoluteScale',
    'AbsoluteStat', 'CorrelationGroupFlavor', 'CorrelationGroupIntercalibration',
    'CorrelationGroupMPFInSitu', 'CorrelationGroupUncorrelated', 'CorrelationGroupbJES',

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
 
uncertainties = jec_unc + ["puWeight"]
shape_systematics = jec_unc + ["jer", "puWeight"]
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
    "subleadingJet_pt": "subleading jet $p_T$ [GeV]",
    "subleadingJet_eta": "subleading jet $\eta$",
    "leadingJet_pt": "leading jet $p_T$ [GeV]",
    "leadingJet_eta": "leading jet $\eta$",
}

analysis_names = {
    "baseline": "baseline",
    "redo_jec_V16": "JEC V16",
    "jetpt_l30_sl30": "jet pT 30,30",
}

# dataset nickname, datataking era, filename glob pattern, isMC
datasets = [
    ("ggh", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/**/*.root", True),
    ("ggh", "2017", "/store/mc/RunIIFall17NanoAODv5/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/**/*.root", True),
    ("ggh", "2018", "/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    
    ("vbf", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("vbf", "2017", "/store/mc/RunIIFall17NanoAODv5/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("vbf", "2018", "/store/mc/RunIIAutumn18NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    
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
    ("dy_m105_160_vbf_mg", "2017", "/store/mc/RunIIFall17NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/**/*.root", True),
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

    ("ewk_lljj_mll50_mjj120", "2016", "/store/mc/RunIISummer16NanoAODv5/EWK_LLJJ_MLL-50_MJJ-120_13TeV-madgraph-pythia8/**/*.root", True),
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
]

# Synchronization datasets/
datasets_sync = [
    ("ggh", "2016", "data/ggh_nano_2016.root", True)
]
