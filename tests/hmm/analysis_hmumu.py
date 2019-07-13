import os
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import numba

import argparse
import numpy as np
import copy
import pickle
import shutil

import hepaccelerate.backend_cpu as backend_cpu
from hepaccelerate.utils import choose_backend, LumiData, LumiMask
from hepaccelerate.utils import Dataset
from cmsutils.decisiontree import DecisionTreeNode, DecisionTreeLeaf

import hmumu_utils
from hmumu_utils import run_analysis, load_analysis, make_random_tree, load_puhist_target, compute_significances, optimize_categories
from hmumu_lib import LibHMuMu, RochesterCorrections, LeptonEfficiencyCorrections

import os
from coffea.util import USE_CUPY

def parse_args():
    parser = argparse.ArgumentParser(description='Example HiggsMuMu analysis')
    #parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--async-data', action='store_true', help='Load data on a separate thread')
    parser.add_argument('--action', '-a', action='append', help='List of actions to do', choices=['cache', 'analyze'], required=True)
    parser.add_argument('--nthreads', '-t', action='store', help='Number of CPU threads or workers to use', type=int, default=4, required=False)
    parser.add_argument('--datapath', '-p', action='store', help='Prefix to load NanoAOD data from', default="/nvmedata")
    parser.add_argument('--maxfiles', '-m', action='store', help='Maximum number of files to process', default=-1, type=int)
    parser.add_argument('--chunksize', '-c', action='store', help='Number of files to process simultaneously', default=2, type=int)
    parser.add_argument('--cache-location', action='store', help='Cache location', default='./mycache', type=str)
    parser.add_argument('--out', action='store', help='Output location', default='out', type=str)
    parser.add_argument('--niter', action='store', help='Number of categorization optimization iterations', default=1, type=int)
    parser.add_argument('--pinned', action='store_true', help='Use CUDA pinned memory')
    parser.add_argument('--filter-datasets', action='store', help='Glob pattern to select datasets', default="*")
    parser.add_argument('--do-sync', action='store_true', help='Run synchronization datasets')
    args = parser.parse_args()
    return args

# dataset nickname, datataking era, filename glob pattern, isMC
datasets = [
# Official 2017 NanoAOD
    ("data", "2017", "/store/data/Run2017*/SingleMuon/NANOAOD/Nano14Dec2018-v1/**/*.root", False),
    ("ggh", "2017", "/store/mc/RunIIFall17NanoAODv4/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("vbf", "2017", "/store/mc/RunIIFall17NanoAODv4/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
#    ("tth", "2017", "/store/mc/RunIIFall17NanoAODv4/ttHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
#    ("wmh", "2017", "/store/mc/RunIIFall17NanoAODv4/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
#    ("wph", "2017", "/store/mc/RunIIFall17NanoAODv4/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
#    ("zh", "2017", "/store/mc/RunIIFall17NanoAODv4/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("dy", "2017", "/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", True),
    ("dy_m105_160", "2017", "/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/NANOAODSIM/**/*.root", True),
    ("dy_m105_160_vbf", "2017", "/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", True),
    ("ttjets_dl", "2017", "/store/mc/RunIIFall17NanoAODv4/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
#    ("ttjets_sl", "2017", "/store/mc/RunIIFall17NanoAODv4/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
#    ("ww_2l2nu", "2017", "/store/mc/RunIIFall17NanoAODv4/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
#    ("wz_3lnu", "2017", "/store/mc/RunIIFall17NanoAODv4/WZTo3LNu_13TeV-powheg-pythia8/**/*.root", True),
#    ("wz_2l2q", "2017", "/store/mc/RunIIFall17NanoAODv4/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
#    ("wz_1l1nu2q", "2017", "/store/mc/RunIIFall17NanoAODv4/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
#    ("zz", "2017", "/store/mc/RunIIFall17NanoAODv4/ZZTo2L2Nu_13TeV_powheg_pythia8/**/*.root", True),
  
# 2016 NanoAOD
    ("dy", "2016", "/store/mc/RunIISummer16NanoAODv5/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/**/*.root", True),
    ("ggh", "2016", "/store/mc/RunIISummer16NanoAODv5/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/**/*.root", True),
    ("ttjets_dl", "2016", "/store/mc/RunIISummer16NanoAODv5/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("vbf", "2016", "/store/mc/RunIISummer16NanoAODv5/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("data", "2016", "/store/data/Run2016*/SingleMuon/NANOAOD/Nano1June2019*/**/*.root", False),

# 2018 NanoAOD
    ("dy", "2018", "/store/mc/RunIIAutumn18NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/**/*.root", True),
    ("ggh", "2018", "/store/mc/RunIIAutumn18NanoAODv4/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/**/*.root", True),
    ("ttjets_dl", "2018", "/store/mc/RunIIAutumn18NanoAODv4/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("vbf", "2018", "/store/mc/RunIIAutumn18NanoAODv4/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/**/*.root", True),
    ("data", "2018", "/store/data/Run2018*/SingleMuon/NANOAOD/Nano14Dec2018*/**/*.root", False),
]

# How many NanoAOD files to load to memory simultaneously.
# Larger numbers mean faster runtime, but may run out of memory
chunksizes_mult = {"2016": 1, "2017": 1, "2018": 1}
maxfiles_mult = {"2016": 1, "2017": 1, "2018": 1}

# Synchronization datasets
datasets_sync = [
    ("ggh", "2016", "data/ggh_nano_2016.root", True)
]

#dataset cross sections in picobarns
cross_sections = {
    "dy": 5765.4,
    "ggh": 0.009605,
    "tth": 0.000110,
    "vbf": 0.000823,
    
    "wmh": 0.000116,
    "wph": 0.000183,
    "zh": 0.000192,
    
    "ttjets_dl": 85.656,
    "ttjets_sl": 687.0,
    "ww_2l2nu": 5.595,
    "wz_3lnu":  4.42965,
    "wz_2l2q": 5.595,
    "wz_1l1nu2q": 11.61,
    "zz": 16.523
}

sig_samples = [
    "ggh",
    "vbf", "tth", "zh", "wmh", "wph"
]
bkg_samples = [
    "dy",
    "ttjets_sl", "ttjets_dl", "ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"
]
mc_samples = sig_samples + bkg_samples

from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor

class JetMetCorrections:
    def __init__(self, do_factorized_jec_unc=False):
        extract = extractor()
        
        extract.add_weight_sets(['* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi.jec.txt.gz',
                                 '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi.jec.txt.gz',
                                 '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi.jec.txt.gz',
                                 '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi.jec.txt.gz',
                                 '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz',
                                 '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi.junc.txt.gz',
                                 '* * coffea/tests/samples/Spring16_25nsV10_MC_PtResolution_AK4PFPuppi.jr.txt.gz',
                                 '* * coffea/tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt.gz'])
        
        extract.finalize()
        evaluator = extract.make_evaluator()
        
        jec_names = ['Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi',
                     'Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi',
                     'Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi',
                     'Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi']
        
        self.jec = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names})
        
        test_eta = np.array([0.2, 1.8, 3.4])
        test_Rho = np.array([1.0, 1.2, 1.3])
        test_pt = np.array([100.0, 200.0, 300.0])
        test_A = np.array([5.0, 6.0, 7.0])
        
        #corr = corrector.getCorrection(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt, JetA=test_A)
        #print(corr)
        
        jer_names = ['Spring16_25nsV10_MC_PtResolution_AK4PFPuppi']
        self.jer = JetResolution(**{name: evaluator[name] for name in jer_names})
        #resos = reso.getResolution(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt)
        #print(list(resos))
            
        jersf_names = ['Spring16_25nsV10_MC_SF_AK4PFPuppi']
        self.jersf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})
        #resosfs = resosf.getScaleFactor(JetEta=test_eta) 
        #print(list(resosfs))
        
        junc_names = ['Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi']
        #levels = []
        if do_factorized_jec_unc:
            for name in dir(evaluator):
                if 'Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi' in name:
                    junc_names.append(name)
                    #levels.append(name.split('_')[-1])
        self.jesunc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})
        #juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)
        #juncs = list(juncs)

if __name__ == "__main__":

    # Do you want to use yappi to profile the python code
    do_prof = False

    args = parse_args()

    #use the environment variable for cupy/cuda choice
    args.use_cuda = USE_CUPY

    # Optionally disable pinned memory (will be somewhat slower)
    if args.use_cuda and not args.pinned:
        import cupy
        cupy.cuda.set_allocator(None)
        cupy.cuda.set_pinned_memory_allocator(None)

    #Use sync-only datasets
    if args.do_sync:
        datasets = datasets_sync

    datasets = [ds for ds in datasets if shutil.fnmatch.fnmatch(ds[0], args.filter_datasets)]
    print("selected datasets {0} based on pattern {1}".format([ds[0] for ds in datasets], args.filter_datasets))
    hmumu_utils.NUMPY_LIB, hmumu_utils.ha = choose_backend(args.use_cuda)
    Dataset.numpy_lib = hmumu_utils.NUMPY_LIB
    DecisionTreeNode.NUMPY_LIB = hmumu_utils.NUMPY_LIB
    
    #Categorization where we first cut on leptons, then jets
    dt = DecisionTreeNode("additional_leptons", 0)
    dt.add_child_left(DecisionTreeNode("num_jets", 0))
    dt.add_child_right(DecisionTreeLeaf())
    dt.child_left.add_child_left(DecisionTreeLeaf())
    dt.child_left.add_child_right(DecisionTreeNode("dijet_inv_mass", 400))
    dt.child_left.child_right.add_child_left(DecisionTreeLeaf())
    dt.child_left.child_right.add_child_right(DecisionTreeNode("num_jets_btag", 0))
    dt.child_left.child_right.child_right.add_child_left(DecisionTreeLeaf())
    dt.child_left.child_right.child_right.add_child_right(DecisionTreeLeaf())
    dt.assign_ids()
    varA = dt

    # dt2 = DecisionTreeNode("num_jets_btag", 0)
    # dt2.add_child_left(DecisionTreeNode("additional_leptons", 0))
    # dt2.child_left.add_child_left(DecisionTreeNode("dijet_inv_mass", 400))
    # dt2.child_left.child_left.add_child_right(DecisionTreeLeaf())
    # dt2.child_left.child_left.add_child_left(DecisionTreeNode("dijet_inv_mass", 60))
    # dt2.child_left.child_left.child_left.add_child_left(DecisionTreeLeaf())
    # dt2.child_left.child_left.child_left.add_child_right(DecisionTreeNode("dijet_inv_mass", 110))
    # dt2.child_left.child_left.child_left.child_right.add_child_left(DecisionTreeLeaf())
    # dt2.child_left.child_left.child_left.child_right.add_child_right(DecisionTreeLeaf())
    # dt2.child_left.add_child_right(DecisionTreeNode("additional_leptons", 1))
    # dt2.child_left.child_right.add_child_left(DecisionTreeLeaf())
    # dt2.child_left.child_right.add_child_right(DecisionTreeLeaf())
    # dt2.add_child_right(DecisionTreeNode("additional_leptons", 0))
    # dt2.child_right.add_child_right(DecisionTreeLeaf())
    # dt2.child_right.add_child_left(DecisionTreeNode("num_jets", 4))
    # dt2.child_right.child_left.add_child_right(DecisionTreeLeaf())
    # dt2.child_right.child_left.add_child_left(DecisionTreeLeaf())
    # dt2.assign_ids()
    # varB = dt2

    # make a hundred random trees as a starting point
    # varlist = {
    #     "dijet_inv_mass": [50, 100, 150, 200, 250, 300, 350, 400, 500, 600],
    #     "num_jets": [0, 1, 2, 3, 4, 5],
    #     "num_jets_btag": [0, 1, 2],
    #     "leading_mu_abs_eta": [0.5, 1.0, 1.5, 2.0],
    #     "additional_leptons": [0,1,3,4]
    # }
    # rand_trees = {"rand{0}".format(i): make_random_tree(varlist, 5) for i in range(1)}

    analysis_parameters = {
        "baseline": {

            "nPV": 0,

            # From HmmAnalyzer code
            #"NdfPV": 0,
            #"zPV": 99999,
            # True cuts
            "NdfPV": 4,
            "zPV": 24,

            # Will be applied with OR
            "hlt_bits": {
                "2016": ["HLT_IsoMu24", "HLT_IsoTkMu24"],
                "2017": ["HLT_IsoMu27"],
                "2018": ["HLT_IsoMu24"],
                },

            "muon_pt": 20,
            "muon_pt_leading": {"2016": 26.0, "2017": 30.0, "2018": 26.0},
            "muon_eta": 2.4,
            "muon_iso": 0.25,
            "muon_id": {"2016": "medium", "2017": "medium", "2018": "medium"},
            "muon_trigger_match_dr": 0.1,
            
            "do_rochester_corrections": True,
            "do_lepton_sf": True,
            
            "do_jec": False, 
            "jet_mu_dr": 0.4,
            "jet_pt": {"2016": 25.0, "2017": 30.0, "2018": 30.0},
            "jet_eta": 4.7,
            "jet_id": "tight",
            "jet_puid": "loose",
            "jet_btag": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},

            "inv_mass_bins": 41,

            "extra_electrons_pt": 20,
            "extra_electrons_eta": 2.5,
            "extra_electrons_iso": 0.4,
            "extra_electrons_id": "mvaFall17V1Iso_WP90",


            "dnn_varlist_order": ['softJet5', 'dRmm', 'dEtamm', 'dPhimm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'Zep', 'dRmin_mj', 'dRmax_mj', 'dRmin_mmj', 'dRmax_mmj', 'leadingJet_pt', 'subleadingJet_pt', 'leadingJet_eta', 'subleadingJet_eta', 'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS', 'Higgs_pt', 'Higgs_eta'],
            "dnn_input_histogram_bins": {
                "softJet5": (0,10,10),
                "dRmm": (0,5,20),
                "dEtamm": (-2,2,20),
                "dPhimm": (-2,2,20),
                "M_jj": (0,400,20),
                "pt_jj": (0,400,20),
                "eta_jj": (-5,5,20),
                "phi_jj": (-5,5,20),
                "M_mmjj": (0,400,20),
                "eta_mmjj": (-3,3,20),
                "phi_mmjj": (-3,3,20),
                "dEta_jj": (-3,3,20),
                "Zep": (-2,2,20),
                "dRmin_mj": (0,5,20),
                "dRmax_mj": (0,5,20),
                "dRmin_mmj": (0,5,20),
                "dRmax_mmj": (0,5,20),
                "leadingJet_pt": (0, 200, 20),
                "subleadingJet_pt": (0, 200, 20),
                "leadingJet_eta": (-5, 5, 20),
                "subleadingJet_eta": (-5, 5, 20),
                "leadingJet_qgl": (-1, 1, 20),
                "subleadingJet_qgl": (-1, 1, 20),
                "cthetaCS": (-1, 1, 20),
                "Higgs_pt": (0, 200, 20),
                "Higgs_eta": (-3, 3, 20),
            },

            "categorization_trees": {}
        },
    }
  
    #add additional analyses for benchmarking purposes 
    if args.niter > 1:
        for i in range(args.niter-1):
            analysis_parameters["v{0}".format(i)] = copy.deepcopy(analysis_parameters["baseline"])

    #analysis_parameters["baseline"]["categorization_trees"].update(rand_trees)

    lumimask = {
        "2016": LumiMask("data/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt", np, backend_cpu),
        "2017": LumiMask("data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt", np, backend_cpu),
        "2018": LumiMask("data/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt", np, backend_cpu),
    }

    lumidata = {
        "2016": LumiData("data/lumi2016.csv"),
        "2017": LumiData("data/lumi2017.csv"),
        "2018": LumiData("data/lumi2018.csv")
    }

    pu_corrections = {
        "2016": load_puhist_target("data/pileup/RunII_2016_data.root"),
        "2017": load_puhist_target("data/pileup/RunII_2017_data.root"),
        "2018": load_puhist_target("data/pileup/RunII_2018_data.root")
    }
    
    libhmm = LibHMuMu()
    rochester_corr = {
        "2016": RochesterCorrections(libhmm, "data/RoccoR2016.txt"),
        "2017": RochesterCorrections(libhmm, "data/RoccoR2017v1.txt"),
        "2018": RochesterCorrections(libhmm, "data/RoccoR2018.txt")
    }

    ratios_dataera = {
        "2016": [0.5548, 1.0 - 0.5548],
        "2017": 1.0,
        "2018": 1.0
    }

    lepsf_iso = {
        "2016": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2016/RunBCDEF_SF_ISO.root", "data/leptonSF/2016/RunGH_SF_ISO.root"],
            ["NUM_LooseRelIso_DEN_MediumID_eta_pt", "NUM_LooseRelIso_DEN_MediumID_eta_pt"],
            ratios_dataera["2016"]),
        "2017": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2017/RunBCDEF_SF_ISO_syst.root"],
            ["NUM_LooseRelIso_DEN_MediumID_pt_abseta"], [1.0]),
        "2018": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2018/RunABCD_SF_ISO.root"],
            ["NUM_LooseRelIso_DEN_MediumID_pt_abseta"], [1.0]),
    }
    lepsf_id = {
        "2016": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2016/RunBCDEF_SF_ID.root", "data/leptonSF/2016/RunGH_SF_ID.root"],
            ["NUM_MediumID_DEN_genTracks_eta_pt", "NUM_MediumID_DEN_genTracks_eta_pt"],
            ratios_dataera["2016"]),
        "2017": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2017/RunBCDEF_SF_ID_syst.root"],
            ["NUM_MediumID_DEN_genTracks_pt_abseta"], [1.0]),
        "2018": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2018/RunABCD_SF_ID.root"],
            ["NUM_MediumID_DEN_TrackerMuons_pt_abseta"], [1.0])
    }
    lepsf_trig = {
        "2016": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2016/EfficienciesAndSF_RunBtoF.root", "data/leptonSF/2016/EfficienciesAndSF_RunGtoH.root"],
            ["IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"],
            ratios_dataera["2016"]),
        "2017": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root"],
            ["IsoMu27_PtEtaBins/pt_abseta_ratio"],
            [1.0]
        ),
        "2018": LeptonEfficiencyCorrections(libhmm,
            ["data/leptonSF/2018/EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root"],
            ["IsoMu24_PtEtaBins/pt_abseta_ratio"],
            [1.0]
        ),
     }
    jetmet_corrections = JetMetCorrections()

    #Run baseline analysis
    outpath = "{0}/baseline".format(args.out)
    try:
        os.makedirs(outpath)
    except FileExistsError as e:
            pass

    with open('{0}/parameters.pickle'.format(outpath), 'wb') as handle:
        pickle.dump(analysis_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if do_prof:
        import yappi
        filename = 'analysis.prof'
        yappi.set_clock_type('cpu')
        yappi.start(builtins=True)
   
    #disable GPU for tensorflow
    if not args.use_cuda: 
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        import tensorflow as tf
    else:
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))
    
    #load DNN model
    import keras
    dnn_model = keras.models.load_model("data/dnn_model.h5")

    run_analysis(args, outpath, datasets, analysis_parameters,
        {k: args.chunksize*v for k, v in chunksizes_mult.items()},
        {k: args.maxfiles*v for k, v in maxfiles_mult.items()},
        lumidata, lumimask, pu_corrections, rochester_corr,
        lepsf_iso, lepsf_id, lepsf_trig, dnn_model,
        jetmet_corrections)

    # if "analyze" in args.action: 
    #     ans = analysis_parameters["baseline"]["categorization_trees"].keys()
    #     r = load_analysis(mc_samples, outpath, cross_sections, ans)
    #     Zs = compute_significances(sig_samples, bkg_samples, r, ans)
    #     print(Zs[:10])
    #     with open('{0}/sigs.pickle'.format(outpath), 'wb') as handle:
    #         pickle.dump(Zs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     #best_tree = copy.deepcopy(analysis_parameters["baseline"]["categorization_trees"][Zs[0][0]])
    #     starting_tree = copy.deepcopy(rand_trees["rand0"])
    #     optimize_categories(sig_samples, bkg_samples, varlist, datasets, lumidata, lumimask, pu_corrections, cross_sections, args, analysis_parameters, starting_tree)

    if do_prof:
        stats = yappi.get_func_stats()
        stats.save(filename, type='callgrind')
