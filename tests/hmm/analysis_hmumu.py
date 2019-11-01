import os
import numba

import argparse
import numpy as np
import copy
import pickle
import shutil
import resource

import hepaccelerate.backend_cpu as backend_cpu
from hepaccelerate.utils import choose_backend, LumiData, LumiMask
from hepaccelerate.utils import Dataset, Results

import hmumu_utils
from hmumu_utils import run_analysis, run_cache, create_dataset_jobfiles, load_puhist_target, seed_generator
from hmumu_lib import LibHMuMu, RochesterCorrections, LeptonEfficiencyCorrections, GBREvaluator, MiscVariables, NNLOPSReweighting, hRelResolution, ZpTReweighting

import os
import getpass

import json
import glob
import sys

from coffea.util import USE_CUPY
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

from pars import datasets, datasets_sync

def parse_args():
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis')
    parser.add_argument('--async-data', action='store_true', help='Load data on a separate thread, faster but disable for debugging')
    parser.add_argument('--action', '-a', action='append', help='List of analysis steps to do', choices=['cache', 'analyze', 'merge'], required=False, default=None)
    parser.add_argument('--nthreads', '-t', action='store', help='Number of CPU threads or workers to use', type=int, default=4, required=False)
    parser.add_argument('--datapath', '-p', action='store', help='Input file path that contains the CMS /store/... folder, e.g. /mnt/hadoop', required=False, default="/storage/user/jpata")
    parser.add_argument('--maxchunks', '-m', action='store', help='Maximum number of files to process for each dataset', default=1, type=int)
    parser.add_argument('--chunksize', '-c', action='store', help='Number of files to process simultaneously (larger is faster, but uses more memory)', default=1, type=int)
    parser.add_argument('--cache-location', action='store', help='Cache location', default='./mycache', type=str)
    parser.add_argument('--out', action='store', help='Output location', default='out', type=str)
    parser.add_argument('--datasets', action='append', help='Dataset names process', type=str, required=False)
    parser.add_argument('--eras', action='append', help='Data eras to process', type=str, required=False)
    parser.add_argument('--pinned', action='store_true', help='Use CUDA pinned memory')
    parser.add_argument('--do-sync', action='store_true', help='run only synchronization datasets')
    parser.add_argument('--do-fsr', action='store_true', help='add FSR recovery')
    parser.add_argument('--do-factorized-jec', action='store_true', help='Enables factorized JEC, disables most validation plots')
    parser.add_argument('--do-profile', action='store_true', help='Profile the code with yappi')
    parser.add_argument('--disable-tensorflow', action='store_true', help='Disable loading and evaluating the tensorflow model')
    parser.add_argument('--jobfiles', action='store', help='Jobfiles to process by the "cache" or "analyze" step', default=None, nargs='+', required=False)
    parser.add_argument('--jobfiles-load', action='store', help='Load the list of jobfiles to process from this file', default=None, required=False)
    
    args = parser.parse_args()

    if args.action is None:
        args.action = ['analyze', 'merge']
    
    if args.eras is None:
        args.eras = ["2016", "2017", "2018"]
    return args

class JetMetCorrections:
    def __init__(self,
        jec_tag,
        jec_tag_data,
        jer_tag,
        jmr_vals,
        do_factorized_jec=True):

        extract = extractor()
        
        #For MC
        extract.add_weight_sets([
            '* * data/jme/{0}_L1FastJet_AK4PFchs.txt'.format(jec_tag),
            '* * data/jme/{0}_L2L3Residual_AK4PFchs.txt'.format(jec_tag),
            '* * data/jme/{0}_L2Relative_AK4PFchs.txt'.format(jec_tag),
            '* * data/jme/{0}_L3Absolute_AK4PFchs.txt'.format(jec_tag),
            '* * data/jme/{0}_UncertaintySources_AK4PFchs.junc.txt'.format(jec_tag),
            '* * data/jme/{0}_Uncertainty_AK4PFchs.junc.txt'.format(jec_tag),
        ])

        if jer_tag:
            extract.add_weight_sets([
            '* * data/jme/{0}_PtResolution_AK4PFchs.jr.txt'.format(jer_tag),
            '* * data/jme/{0}_SF_AK4PFchs.jersf.txt'.format(jer_tag)])
        #For data, make sure we don't duplicate
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                extract.add_weight_sets([
                '* * data/jme/{0}_L1FastJet_AK4PFchs.txt'.format(tag),
                '* * data/jme/{0}_L2L3Residual_AK4PFchs.txt'.format(tag),
                '* * data/jme/{0}_L2Relative_AK4PFchs.txt'.format(tag),
                '* * data/jme/{0}_L3Absolute_AK4PFchs.txt'.format(tag)
                ])
                tags_done += [tag]
        
        extract.finalize()
        evaluator = extract.make_evaluator()
        
        jec_names_mc = [
            '{0}_L1FastJet_AK4PFchs'.format(jec_tag),
            '{0}_L2Relative_AK4PFchs'.format(jec_tag),
            '{0}_L2L3Residual_AK4PFchs'.format(jec_tag),
            '{0}_L3Absolute_AK4PFchs'.format(jec_tag)]
        self.jec_mc = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names_mc})

        self.jec_data = {}
        for run, tag in jec_tag_data.items():
            jec_names_data = [
                '{0}_L1FastJet_AK4PFchs'.format(tag),
                '{0}_L2Relative_AK4PFchs'.format(tag),
                '{0}_L2L3Residual_AK4PFchs'.format(tag),
                '{0}_L3Absolute_AK4PFchs'.format(tag)]
            self.jec_data[run] = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names_data})
      
        self.jer = None 
        self.jersf = None 
        if jer_tag: 
            jer_names = ['{0}_PtResolution_AK4PFchs'.format(jer_tag)]
            self.jer = JetResolution(**{name: evaluator[name] for name in jer_names})
            jersf_names = ['{0}_SF_AK4PFchs'.format(jer_tag)]
            self.jersf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})

        junc_names = ['{0}_Uncertainty_AK4PFchs'.format(jec_tag)]
        #levels = []
        if do_factorized_jec:
            for name in dir(evaluator):

                #factorized sources
                if '{0}_UncertaintySources_AK4PFchs'.format(jec_tag) in name:
                    junc_names.append(name)

        self.jesunc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

class AnalysisCorrections:
    def __init__(self, args, do_tensorflow=True, gpu_memory_fraction=0.2):
        self.lumimask = {
            "2016": LumiMask("data/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt", np, backend_cpu),
            "2017": LumiMask("data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt", np, backend_cpu),
            "2018": LumiMask("data/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt", np, backend_cpu),
        }

        self.lumidata = {
            "2016": LumiData("data/lumi2016.csv"),
            "2017": LumiData("data/lumi2017.csv"),
            "2018": LumiData("data/lumi2018.csv")
        }

        self.pu_corrections = {
            "2016": load_puhist_target("data/pileup/RunII_2016_data.root"),
            "2017": load_puhist_target("data/pileup/RunII_2017_data.root"),
            "2018": load_puhist_target("data/pileup/RunII_2018_data.root")
        }
        
        #Load the C++ helper library
        try:
            self.libhmm = LibHMuMu()
        except OSError as e:
            print(e, file=sys.stderr)
            print("Could not load the C++ helper library, please make sure it's compiled by doing", file=sys.stderr)
            print("  $ cd tests/hmm && make -j4", file=sys.stderr)
            print("If there are still issues, please make sure ROOT is installed")
            print("and the library is compiled against the same version as seen at runtime", file=sys.stderr)
            sys.exit(1)

        print("Loading Rochester corrections")
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/RochcorMuon
        self.rochester_corrections = {
            "2016": RochesterCorrections(self.libhmm, "data/RoccoR2016.txt"),
            "2017": RochesterCorrections(self.libhmm, "data/RoccoR2017.txt"),
            "2018": RochesterCorrections(self.libhmm, "data/RoccoR2018.txt")
        }

        #Weight ratios for computing scale factors
        self.ratios_dataera = {
            "2016": [0.5548, 1.0 - 0.5548],
            "2017": 1.0,
            "2018": 1.0
        }

        print("Loading lepton SF")
        self.lepsf_iso = {
            "2016": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2016/RunBCDEF_SF_ISO.root", "data/leptonSF/2016/RunGH_SF_ISO.root"],
                ["NUM_LooseRelIso_DEN_MediumID_eta_pt", "NUM_LooseRelIso_DEN_MediumID_eta_pt"],
                self.ratios_dataera["2016"]),
            "2017": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2017/RunBCDEF_SF_ISO_syst.root"],
                ["NUM_LooseRelIso_DEN_MediumID_pt_abseta"], [1.0]),
            "2018": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2018/RunABCD_SF_ISO.root"],
                ["NUM_LooseRelIso_DEN_MediumID_pt_abseta"], [1.0]),
        }
        self.lepsf_id = {
            "2016": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2016/RunBCDEF_SF_ID.root", "data/leptonSF/2016/RunGH_SF_ID.root"],
                ["NUM_MediumID_DEN_genTracks_eta_pt", "NUM_MediumID_DEN_genTracks_eta_pt"],
                self.ratios_dataera["2016"]),
            "2017": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2017/RunBCDEF_SF_ID_syst.root"],
                ["NUM_MediumID_DEN_genTracks_pt_abseta"], [1.0]),
            "2018": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2018/RunABCD_SF_ID.root"],
                ["NUM_MediumID_DEN_TrackerMuons_pt_abseta"], [1.0])
        }
        self.lepsf_trig = {
            "2016": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2016/EfficienciesAndSF_RunBtoF.root", "data/leptonSF/2016/EfficienciesAndSF_RunGtoH.root"],
                ["IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"],
                self.ratios_dataera["2016"]),
            "2017": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root"],
                ["IsoMu27_PtEtaBins/pt_abseta_ratio"],
                [1.0]
            ),
            "2018": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2018/EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root"],
                ["IsoMu24_PtEtaBins/pt_abseta_ratio"],
                [1.0]
            ),
        }

        print("Loading JEC...")
        #JEC files copied from
        #https://github.com/cms-jet/JECDatabase/tree/master/textFiles
        #Need to rename JECDatabase files as follows:
        #  *_UncertaintySources_AK4PFchs.txt -> *_UncertaintySources_AK4PFchs.junc.txt
        #  *_Uncertainty_AK4PFchs.txt -> *_Uncertainty_AK4PFchs.junc.txt
        #JER files from
        #  *_PtResolution_AK4PFchs.txt -> *_PtResolution_AK4PFchs.jr.txt
        #  *_SF_AK4PFchs.txt -> *_SF_AK4PFchs.jersf.txt
        self.jetmet_corrections = {
            "2016": {
                "Summer16_07Aug2017_V11":
                    JetMetCorrections(
                    jec_tag="Summer16_07Aug2017_V11_MC",
                    jec_tag_data={
                        "RunB": "Summer16_07Aug2017BCD_V11_DATA",
                        "RunC": "Summer16_07Aug2017BCD_V11_DATA",
                        "RunD": "Summer16_07Aug2017BCD_V11_DATA",
                        "RunE": "Summer16_07Aug2017EF_V11_DATA",
                        "RunF": "Summer16_07Aug2017EF_V11_DATA",
                        "RunG": "Summer16_07Aug2017GH_V11_DATA",
                        "RunH": "Summer16_07Aug2017GH_V11_DATA",
                    },
                    jer_tag="Summer16_25nsV1_MC",
                    jmr_vals=[1.0, 1.2, 0.8],
                    do_factorized_jec=True),
            },
            "2017": {
                "Fall17_17Nov2017_V32":
                    JetMetCorrections(
                    jec_tag="Fall17_17Nov2017_V32_MC",
                    jec_tag_data={
                        "RunB": "Fall17_17Nov2017B_V32_DATA",
                        "RunC": "Fall17_17Nov2017C_V32_DATA",
                        "RunD": "Fall17_17Nov2017DE_V32_DATA",
                        "RunE": "Fall17_17Nov2017DE_V32_DATA",
                        "RunF": "Fall17_17Nov2017F_V32_DATA",
                    },
                    jer_tag="Fall17_V3_MC",
                    jmr_vals=[1.09, 1.14, 1.04],
                    do_factorized_jec=True),
            },
            "2018": {
                "Autumn18_V16":
                    JetMetCorrections(
                    jec_tag="Autumn18_V16_MC",
                    jec_tag_data={
                        "RunA": "Autumn18_RunA_V16_DATA",
                        "RunB": "Autumn18_RunB_V16_DATA",
                        "RunC": "Autumn18_RunC_V16_DATA",
                        "RunD": "Autumn18_RunD_V16_DATA",
                    },
                    jer_tag="Autumn18_V7_MC",
                    jmr_vals=[1.0, 1.2, 0.8],
                    do_factorized_jec=True),
            }
        }

        self.dnn_model = None
        self.dnn_normfactors = None
        self.dnnPisa_models = []
        self.dnnPisa_normfactors1 = None
        self.dnnPisa_normfactors2 = None
        if do_tensorflow:
            print("Loading tensorflow model")
            #disable GPU for tensorflow
            import tensorflow as tf
            config = tf.ConfigProto()
            config.intra_op_parallelism_threads=args.nthreads
            config.inter_op_parallelism_threads=args.nthreads

            if not args.use_cuda: 
                os.environ["CUDA_VISIBLE_DEVICES"]="-1"
            else:
                from keras.backend.tensorflow_backend import set_session
                import tensorflow as tf
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = False
                config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

            from keras.backend.tensorflow_backend import set_session
            set_session(tf.Session(config=config))

            #load DNN model
            import keras
            self.dnn_model = keras.models.load_model("data/DNN27vars_sig_vbf_ggh_bkg_dyvbf_dy105To160_ewk105To160_split_60_40_mod10_191008.h5")
            self.dnn_normfactors = np.load("data/DNN27vars_sig_vbf_ggh_bkg_dyvbf_dy105To160_ewk105To160_split_60_40_mod10_191008.npy")

            if args.use_cuda:
                import cupy
                self.dnn_normfactors = cupy.array(self.dnn_normfactors[0]), cupy.array(self.dnn_normfactors[1])
            
            for imodel in range(4):
                json_path = "data/PisaDNN/model_preparation/model_toexport_evt"+str(imodel)+".json"
                with open(json_path, 'r') as file_handle:
                    dnnPisa_model = keras.models.model_from_json(file_handle.read())
                dnnPisa_model.load_weights("data/PisaDNN/model_preparation/model_toexport_evt"+str(imodel)+".h5")
                self.dnnPisa_models += [dnnPisa_model]
                self.dnnPisa_normfactors1 = np.load("data/PisaDNN/model_preparation/helphelp_node1.npy")
                self.dnnPisa_normfactors2 = np.load("data/PisaDNN/model_preparation/helphelp_node2.npy")
                if args.use_cuda:
                    import cupy
                    self.dnnPisa_normfactors1 = cupy.array(self.dnnPisa_normfactors1[0]), cupy.array(self.dnnPisa_normfactors1[1])
                    self.dnnPisa_normfactors2 = cupy.array(self.dnnPisa_normfactors2[0]), cupy.array(self.dnnPisa_normfactors2[1])
        print("Loading UCSD BDT model")
        self.bdt_ucsd = GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2016/TMVAClassification_BDTG.weights.2jet_bveto_withmass.xml")
        self.bdt2j_ucsd = {
            "2016": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2016/TMVAClassification_BDTG.weights.2jet_bveto.xml"),
            "2017": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2017/TMVAClassification_BDTG.weights.2jet_bveto.xml"),
            "2018": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2018/TMVAClassification_BDTG.weights.2jet_bveto.xml")
        }
        self.bdt01j_ucsd = {
            "2016": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2016/TMVAClassification_BDTG.weights.01jet.xml"),
            "2017": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2017/TMVAClassification_BDTG.weights.01jet.xml"),
            "2018": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2018/TMVAClassification_BDTG.weights.01jet.xml")
        }
        self.miscvariables = MiscVariables(self.libhmm)
        print("Loading NNLOPSReweighting...")
        self.nnlopsreweighting = NNLOPSReweighting(self.libhmm, "data/nnlops/NNLOPS_reweight.root")
        print("Loading hRelResolution...")
        self.hrelresolution = hRelResolution(self.libhmm, "data/PisaDNN/muonresolution.root")
        print("Loading ZpTReweighting...")
        self.zptreweighting = ZpTReweighting(self.libhmm)


        puid_maps = "data/puidSF/PUIDMaps.root"
        print("Extracting PU ID weights from "+puid_maps)
        puid_extractor = extractor()
        puid_extractor.add_weight_sets(["* * {0}".format(puid_maps)])
        puid_extractor.finalize()
        self.puidreweighting = puid_extractor.make_evaluator()


def check_and_recreate_filename_cache(cache_filename, cache_location, datapath, datasets):
    if os.path.isfile(cache_filename):
        print("Cache file {0} already exists, we will not overwrite it to be safe.".format(cache_filename), file=sys.stderr)
        print("Delete it or change --cache-location and try again.", file=sys.stderr)
        sys.exit(1)
        print("--action cache and no jobfiles specified, creating datasets.json dump of all filenames")
    
    if not os.path.isdir(cache_location):
        os.makedirs(cache_location)
    filenames_cache = {}

    for dataset in datasets:
        dataset_name, dataset_era, dataset_globpattern, is_mc = dataset
        filenames_all = glob.glob(datapath + dataset_globpattern, recursive=True)
        filenames_all = [fn for fn in filenames_all if not "Friend" in fn]
        filenames_cache[dataset_name + "_" + dataset_era] = [
            fn.replace(datapath, "") for fn in filenames_all]

        if len(filenames_all) == 0:
            raise Exception("Dataset {0} matched 0 files from glob pattern {1}, verify that the data files are located in {2}".format(
                dataset_name, dataset_globpattern, datapath
            ))
    
    #save all dataset filenames to a json file 
    print("Creating a json dump of all the dataset filenames based on data found in {0}".format(datapath))
    with open(cache_filename, "w") as fi:
        fi.write(json.dumps(filenames_cache, indent=2))

def create_all_jobfiles(datasets, cache_filename, datapath, chunksize, outpath):
    #Create a list of job files for processing
    jobfile_data = []
    print("Loading list of filenames from {0}".format(cache_filename))
    if not os.path.isfile(cache_filename):
        raise Exception("Cached dataset list of filenames not found in {0}, please run this code with --action cache".format(
            cache_filename))
    filenames_cache = json.load(open(cache_filename, "r"))

    seed_gen = seed_generator()
    for dataset in sorted(datasets):
        dataset_name, dataset_era, dataset_globpattern, is_mc = dataset
        try:
            filenames_all = filenames_cache[dataset_name + "_" + dataset_era]
        except KeyError as e:
            print("Could not load {0} from {1}, please make sure this dataset has been added to cache".format(
                dataset_name + "_" + dataset_era, cache_filename), file=sys.stderr)
            raise e

        filenames_all_full = [datapath + "/" + fn for fn in filenames_all]
        print("Saving dataset {0}_{1} with {2} files in {3} files per chunk to jobfiles".format(
            dataset_name, dataset_era, len(filenames_all_full), chunksize))
        jobfile_dataset = create_dataset_jobfiles(dataset_name, dataset_era,
            filenames_all_full, is_mc, chunksize, outpath, seed_gen)
        jobfile_data += jobfile_dataset
        print("Dataset {0}_{1} consists of {2} chunks".format(
            dataset_name, dataset_era, len(jobfile_dataset)))

    assert(len(jobfile_data) > 0)
    assert(len(jobfile_data[0]["filenames"]) > 0)


def load_jobfiles(datasets, jobfiles_load_from_file, jobfiles, maxchunks, outpath):
    #Load from file
    if not (jobfiles_load_from_file is None):
        jobfiles = [l.strip() for l in open(jobfiles_load_from_file).readlines()]

    #Check for existing jobfiles
    if jobfiles is None:
        print("You did not specify to process specific dataset chunks, assuming you want to process all chunks")
        print("If this is not true, please specify e.g. --jobfiles data_2018_0.json data_2018_1.json ...")
        jobfiles = []
        for dataset in datasets:
            dataset_name, dataset_era, dataset_globpattern, is_mc = dataset
            jobfile_pattern = outpath + "/jobfiles/{0}_{1}_*.json".format(dataset_name, dataset_era)
            jobfiles_dataset = glob.glob(jobfile_pattern)
            if len(jobfiles_dataset) == 0:
                raise Exception("Could not find any jobfiles matching pattern {0}".format(jobfile_pattern))

            if maxchunks > 0:
                jobfiles_dataset = jobfiles_dataset[:maxchunks]
            jobfiles += jobfiles_dataset
    
    #Now actually load the jobfiles 
    assert(len(jobfiles) > 0)
    print("You specified --jobfiles {0}, processing only these dataset chunks".format(" ".join(jobfiles))) 
    jobfile_data = []
    for f in jobfiles:
        jobfile_data += [json.load(open(f))]

    chunkstr = " ".join(["{0}_{1}_{2}".format(
        ch["dataset_name"], ch["dataset_era"], ch["dataset_num_chunk"])
        for ch in jobfile_data])
    print("Will process {0} dataset chunks: {1}".format(len(jobfile_data), chunkstr))
    assert(len(jobfile_data) > 0)
    return jobfile_data

def merge_partial_results(dataset_name, dataset_era, outpath, outpath_partial):
    results = []
    partial_results = glob.glob(outpath_partial + "/{0}_{1}_*.pkl".format(dataset_name, dataset_era))
    print("Merging {0} partial results for dataset {1}_{2}".format(len(partial_results), dataset_name, dataset_era))
    for res_file in partial_results:
        res = pickle.load(open(res_file, "rb"))
        results += [res]
    results = sum(results, Results({}))
    try:
        os.makedirs(outpath + "/results")
    except FileExistsError as e:
        pass
    result_filename = outpath + "/results/{0}_{1}.pkl".format(dataset_name, dataset_era)
    print("Saving results to {0}".format(result_filename))
    with open(result_filename, "wb") as fi:
        pickle.dump(results, fi, protocol=pickle.HIGHEST_PROTOCOL) 
    return

def main(args, datasets):
    do_prof = args.do_profile
    do_tensorflow = not args.disable_tensorflow

    #use the environment variable for cupy/cuda choice
    args.use_cuda = USE_CUPY

    # Optionally disable pinned memory (will be somewhat slower)
    if args.use_cuda:
        import cupy
        if not args.pinned:
            cupy.cuda.set_allocator(None)
            cupy.cuda.set_pinned_memory_allocator(None)

    #Use sync-only datasets
    if args.do_sync:
        datasets = datasets_sync

    #Filter datasets by era
    datasets_to_process = []
    for ds in datasets:
        if args.datasets is None or ds[0] in args.datasets:
            if args.eras is None or ds[1] in args.eras:
                datasets_to_process += [ds]
                print("Will consider dataset", ds)
    if len(datasets) == 0:
        raise Exception("No datasets considered, please check the --datasets and --eras options")
    datasets = datasets_to_process

    hmumu_utils.NUMPY_LIB, hmumu_utils.ha = choose_backend(args.use_cuda)
    Dataset.numpy_lib = hmumu_utils.NUMPY_LIB
    NUMPY_LIB = hmumu_utils.NUMPY_LIB 

    outpath_partial = "{0}/partial_results".format(args.out)
    try:
        os.makedirs(outpath_partial)
    except FileExistsError as e:
        print("Output path {0} already exists, not recreating".format(outpath_partial))

    #save the parameters as a pkl file
    from pars import analysis_parameters
    for analysis_name in analysis_parameters.keys():
        analysis_parameters[analysis_name]["do_factorized_jec"] = args.do_factorized_jec
        analysis_parameters[analysis_name]["dnn_vars_path"] = "{0}/dnn_vars".format(args.out)
 
    with open('{0}/parameters.pkl'.format(outpath_partial), 'wb') as handle:
        pickle.dump(analysis_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Recreate dump of all filenames
    cache_filename = args.cache_location + "/datasets.json"
    if ("cache" in args.action) and (args.jobfiles is None):

        check_and_recreate_filename_cache(cache_filename, args.cache_location, args.datapath, datasets)

    #Create the jobfiles
    if ("cache" in args.action or "analyze" in args.action) and (args.jobfiles is None):
        create_all_jobfiles(datasets, cache_filename, args.datapath, args.chunksize, args.out)

    #For each dataset, find out which chunks we want to process
    if "cache" in args.action or "analyze" in args.action:
        jobfile_data = load_jobfiles(datasets, args.jobfiles_load, args.jobfiles, args.maxchunks, args.out)

    #Start the profiler only in the actual data processing
    if do_prof:
        import yappi
        yappi.set_clock_type('cpu')
        yappi.start(builtins=True)

    if "cache" in args.action:
        print("Running the 'cache' step of the analysis, ROOT files will be opened and branches will be uncompressed")
        run_cache(args, outpath_partial, jobfile_data, analysis_parameters)
   
    #Run the physics analysis on all specified jobfiles  
    if "analyze" in args.action:
        print("Running the 'analyze' step of the analysis, processing the events into histograms with all systematics")
        analysis_corrections = AnalysisCorrections(args, do_tensorflow)
        run_analysis(args, outpath_partial, jobfile_data, analysis_parameters, analysis_corrections)
    
    if do_prof:
        stats = yappi.get_func_stats()
        stats.save("analysis.prof", type='callgrind')

    #Merge the partial results (pieces of each dataset)
    if "merge" in args.action:
        with ProcessPoolExecutor(max_workers=args.nthreads) as executor:
            for dataset in datasets:
                dataset_name, dataset_era, dataset_globpattern, is_mc = dataset
                fut = executor.submit(merge_partial_results, dataset_name, dataset_era, args.out, outpath_partial)
        print("done merging")

    #print memory usage
    total_memory = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    total_memory += resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("maxrss={0} MB".format(total_memory/1024))

if __name__ == "__main__":
    args = parse_args()
    main(args, datasets)
