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

def parse_args():
    parser = argparse.ArgumentParser(description='Example HiggsMuMu analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--async-data', action='store_true', help='Load data on a separate thread')
    parser.add_argument('--action', '-a', action='append', help='List of actions to do', choices=['cache', 'analyze'], required=True)
    parser.add_argument('--nthreads', '-t', action='store', help='Number of CPU threads or workers to use', type=int, default=4, required=False)
    parser.add_argument('--datapath', '-p', action='store', help='Prefix to load NanoAOD data from', default="/nvmedata")
    parser.add_argument('--maxfiles', '-m', action='store', help='Maximum number of files to process', default=-1, type=int)
    parser.add_argument('--chunksize', '-c', action='store', help='Number of files to process simultaneously', default=1, type=int)
    parser.add_argument('--cache-location', action='store', help='Cache location', default='', type=str)
    parser.add_argument('--out', action='store', help='Output location', default='out', type=str)
    parser.add_argument('--niter', action='store', help='Number of categorization optimization iterations', default=1, type=int)
    parser.add_argument('--pinned', action='store_true', help='Use CUDA pinned memory')
    parser.add_argument('--filter-datasets', action='store', help='Glob pattern to select datasets', default="*")
    args = parser.parse_args()
    return args

#dataset nickname, filename glob pattern, isMC
datasets = [
    ("data_2017", "/store/data/Run2017*/SingleMuon/NANOAOD/Nano14Dec2018-v1/**/*.root", False),
#    ("data_2018", "/store/data/Run2018*/SingleMuon/NANOAOD/Nano14Dec2018_ver2-v1/**/*.root", False),
    ("ggh", "/store/mc/RunIIFall17NanoAODv4/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("vbf", "/store/mc/RunIIFall17NanoAODv4/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("tth", "/store/mc/RunIIFall17NanoAODv4/ttHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("wmh", "/store/mc/RunIIFall17NanoAODv4/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("wph", "/store/mc/RunIIFall17NanoAODv4/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
    ("zh", "/store/mc/RunIIFall17NanoAODv4/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/NANOAODSIM/*12Apr2018_Nano14Dec2018*/**/*.root", True),
   ("dy", "/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", True),
    ("dy_vbf", "/store/mc/RunIIFall17NanoAODv4/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", True),
    ("ttjets_dl", "/store/mc/RunIIFall17NanoAODv4/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/**/*.root", True),
    ("ttjets_sl", "/store/mc/RunIIFall17NanoAODv4/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("ww_2l2nu", "/store/mc/RunIIFall17NanoAODv4/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/**/*.root", True),
    ("wz_3lnu", "/store/mc/RunIIFall17NanoAODv4/WZTo3LNu_13TeV-powheg-pythia8/**/*.root", True),
    ("wz_2l2q", "/store/mc/RunIIFall17NanoAODv4/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
    ("wz_1l1nu2q", "/store/mc/RunIIFall17NanoAODv4/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/**/*.root", True),
    ("zz", "/store/mc/RunIIFall17NanoAODv4/ZZTo2L2Nu_13TeV_powheg_pythia8/**/*.root", True),
]

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

if __name__ == "__main__":

    do_prof = False

    args = parse_args()
    if args.use_cuda and not args.pinned:
        import cupy
        cupy.cuda.set_allocator(None)
        cupy.cuda.set_pinned_memory_allocator(None)

    datasets = [ds for ds in datasets if shutil.fnmatch.fnmatch(ds[0], args.filter_datasets)]
    print("selected datasets {0} based on pattern {1}".format([ds[0] for ds in datasets], args.filter_datasets))
    hmumu_utils.NUMPY_LIB, hmumu_utils.ha = choose_backend(args.use_cuda)
    Dataset.numpy_lib = hmumu_utils.NUMPY_LIB
    DecisionTreeNode.NUMPY_LIB = hmumu_utils.NUMPY_LIB
    
    varlist = {
        "dijet_inv_mass": [50, 100, 150, 200, 250, 300, 350, 400, 500, 600],
        "num_jets": [0, 1, 2, 3, 4, 5],
        "num_jets_btag": [0, 1, 2],
        "leading_mu_abs_eta": [0.5, 1.0, 1.5, 2.0],
        "additional_leptons": [0,1,3,4]
    }

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

    #make a hundred random trees as a starting point
    rand_trees = {"rand{0}".format(i): make_random_tree(varlist, 5) for i in range(1)}

    analysis_parameters = {
        "baseline": {
            "NdfPV": 4,
            "zPV": 24,
            "nPV": 0,
            "muon_pt": 20,
            "muon_pt_leading": 30,
            "muon_eta": 2.4,
            "muon_iso": 0.25,
            "muon_id": "medium",
            "muon_trigger_match_dr": 0.1,
            
            "do_rochester_corrections": True,
            "do_lepton_sf": True,

            "jet_mu_dr": 0.4,
            "jet_pt": 30.0,
            "jet_eta": 4.7,
            "jet_id": "tight",
            "jet_puid": "loose",
            "jet_btag": 0.4941,

            "inv_mass_bins": 41,

            "extra_electrons_pt": 5,
            "extra_electrons_eta": 2.5,
            "extra_electrons_id": "mvaFall17V1Iso_WP90",


            "dnn_varlist_order": ['softJet5', 'dRmm', 'dEtamm', 'dPhimm', 'M_jj', 'pt_jj', 'eta_jj', 'phi_jj', 'M_mmjj', 'eta_mmjj', 'phi_mmjj', 'dEta_jj', 'Zep', 'dRmin_mj', 'dRmax_mj', 'dRmin_mmj', 'dRmax_mmj', 'leadingJet_pt', 'subleadingJet_pt', 'leadingJet_eta', 'subleadingJet_eta', 'leadingJet_qgl', 'subleadingJet_qgl', 'cthetaCS', 'Higgs_pt', 'Higgs_eta'],
            "dnn_input_histogram_bins": {
                "softJet5": (0,10,10),
                "dRmm": (0,2,20),
                "dEtamm": (-2,2,20),
                "dPhimm": (-2,2,20),
                "M_jj": (0,400,20),
                "pt_jj": (0,400,20),
                "eta_jj": (-3,3,20),
                "phi_jj": (-3,3,20),
                "M_mmjj": (0,400,20),
                "eta_mmjj": (-3,3,20),
                "phi_mmjj": (-3,3,20),
                "dEta_jj": (-3,3,20),
                "Zep": (-2,2,20),
                "dRmin_mj": (0,2,20),
                "dRmax_mj": (0,2,20),
                "dRmin_mmj": (0,2,20),
                "dRmax_mmj": (0,2,20),
                "leadingJet_pt": (0, 200, 20),
                "subleadingJet_pt": (0, 200, 20),
                "leadingJet_eta": (0, 200, 20),
                "subleadingJet_eta": (0, 200, 20),
                "leadingJet_qgl": (0, 200, 20),
                "subleadingJet_qgl": (0, 200, 20),
                "cthetaCS": (-1, 1, 20),
                "Higgs_pt": (0, 200, 20),
                "Higgs_eta": (-3, 3, 20),
            },

            "categorization_trees": {"varA": copy.deepcopy(dt)}
        },
    }
  
    #add additional analyses for benchmarking purposes 
    if args.niter > 1:
        for i in range(args.niter-1):
            analysis_parameters["v{0}".format(i)] = copy.deepcopy(analysis_parameters["baseline"])

    #analysis_parameters["baseline"]["categorization_trees"].update(rand_trees)

    lumimask = LumiMask("data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt", np, backend_cpu)
    lumidata = LumiData("data/lumi2017.csv")
    pu_corrections_2017 = load_puhist_target("data/RunII_2017_data.root")
    
    libhmm = LibHMuMu()
    rochester_corr = RochesterCorrections(libhmm, "data/RoccoR2017v1.txt")
    lepsf_iso = LeptonEfficiencyCorrections(libhmm, "data/leptonSF/RunBCDEF_SF_ISO.root", "NUM_LooseRelIso_DEN_MediumID_pt_abseta")
    lepsf_id = LeptonEfficiencyCorrections(libhmm, "data/leptonSF/RunBCDEF_SF_ID.root", "NUM_MediumID_DEN_genTracks_pt_abseta")
    lepsf_trig = LeptonEfficiencyCorrections(libhmm, "data/leptonSF/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root", "IsoMu27_PtEtaBins/pt_abseta_ratio")

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
    
    import keras
    #load DNN model
    dnn_model = keras.models.load_model("data/dnn_model.h5")

    run_analysis(args, outpath, datasets, analysis_parameters, lumidata, lumimask, pu_corrections_2017, rochester_corr, lepsf_iso, lepsf_id, lepsf_trig, dnn_model)

    # if "analyze" in args.action: 
    #     ans = analysis_parameters["baseline"]["categorization_trees"].keys()
    #     r = load_analysis(mc_samples, outpath, cross_sections, ans)
    #     Zs = compute_significances(sig_samples, bkg_samples, r, ans)
    #     print(Zs[:10])
    #     with open('{0}/sigs.pickle'.format(outpath), 'wb') as handle:
    #         pickle.dump(Zs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     #best_tree = copy.deepcopy(analysis_parameters["baseline"]["categorization_trees"][Zs[0][0]])
    #     starting_tree = copy.deepcopy(rand_trees["rand0"])
    #     optimize_categories(sig_samples, bkg_samples, varlist, datasets, lumidata, lumimask, pu_corrections_2017, cross_sections, args, analysis_parameters, starting_tree)

    if do_prof:
        stats = yappi.get_func_stats()
        stats.save(filename, type='callgrind')
