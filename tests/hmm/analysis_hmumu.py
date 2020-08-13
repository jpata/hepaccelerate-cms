"""Main entry point for the Caltech HiggsMuMu accelerated analysis code.
Process multiple years from NanoAOD to final results with on-the-fly systematics in a few hours!
"""
import os

import argparse
import numpy as np
import pickle
import resource

#Load the acceleration backend
import hepaccelerate.backend_cpu as backend_cpu
from hepaccelerate.utils import choose_backend, LumiData, LumiMask
from hepaccelerate.utils import Dataset, Results

import hmumu_utils
from hmumu_utils import run_analysis, create_dataset_jobfiles, load_puhist_target, seed_generator
#Load legacy ROOT-based C++ corrections that are not yet available in coffea 
from hmumu_lib import LibHMuMu, RochesterCorrections, LeptonEfficiencyCorrections, GBREvaluator, BTagCalibration
from hmumu_lib import MiscVariables, NNLOPSReweighting, hRelResolution, ZpTReweighting

import json
import glob
import sys
import yaml

#We use coffea for jetmet and btag corrections
from coffea.util import USE_CUPY
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.btag_tools import BTagScaleFactor
from concurrent.futures import ProcessPoolExecutor

from typing import List, Dict


def parse_args():
    """Parses the command-line arguments.
    
    Returns:
        Namespace: The parsed arguments for the HiggsMuMu analysis
    """
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis')
    parser.add_argument('--async-data', action='store_true', help='Load data on a separate thread, faster but disable for debugging')
    parser.add_argument('--action', '-a', action='append', help='List of analysis steps to do', choices=['analyze', 'merge'], required=False, default=None)
    parser.add_argument('--nthreads', '-t', action='store', help='Number of CPU threads or workers to use', type=int, default=4, required=False)
    parser.add_argument('--datapath', '-p', action='store', help='Input file path that contains the CMS /store/... folder, e.g. /mnt/hadoop', required=False, default="/storage/user/jpata")
    parser.add_argument('--maxchunks', '-m', action='store', help='Maximum number of files to process for each dataset', default=1, type=int)
    parser.add_argument('--chunksize', '-c', action='store', help='Number of files to process simultaneously (larger is faster, but uses more memory)', default=1, type=int)
    parser.add_argument('--out', action='store', help='Output location', default='out', type=str)
    parser.add_argument('--datasets', action='append', help='Dataset names process', type=str, required=False)
    parser.add_argument('--datasets-yaml', action='store', help='Dataset definition file', type=str, required=True)
    parser.add_argument('--cachepath', action='store', help='Location of the skimmed NanoAOD files', type=str, required=False)
    parser.add_argument('--eras', action='append', help='Data eras to process', type=str, required=False)
    parser.add_argument('--do-sync', action='store_true', help='run only synchronization datasets')
    parser.add_argument('--do-fsr', action='store_true', help='add FSR recovery')
    parser.add_argument('--do-factorized-jec', action='store_true', help='Enables factorized JEC, disables most validation plots')
    parser.add_argument('--do-profile', action='store_true', help='Profile the code with yappi')
    parser.add_argument('--disable-tensorflow', action='store_true', help='Disable loading and evaluating the tensorflow model')
    parser.add_argument('--jobfiles', action='store', help='Jobfiles to process', default=None, nargs='+', required=False)
    parser.add_argument('--jobfiles-load', action='store', help='Load the list of jobfiles to process from this file', default=None, required=False)
    
    args = parser.parse_args()

    if args.action is None:
        args.action = ['analyze', 'merge']
    
    if args.eras is None:
        args.eras = ["2016", "2017", "2018"]
    return args


class BTagWeights:

    """Loads and computes b-tagging weights.
    More documentation is available in https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
    
    Attributes:
        evaluator (coffea.lookup_tools.evaluator): The b-tagging weights loaded to memory and ready for evaluation 
    """
    
    def __init__(self,
        tag_name):
        """Loads the b-tagging weights and creates the evaluator
        
        Args:
            tag_name (str): The filename to load
        """
        btag_extractor = extractor()
        btag_extractor.add_weight_sets(['* * data/btagSF/{0}.csv'.format(tag_name)])
        btag_extractor.finalize()
        self.evaluator = btag_extractor.make_evaluator() 


class JetMetCorrections:

    """Loads and computes JEC and JER corrections.
    
    Attributes:
        jec_data (dict of str->coffea.jetmet_tools.FactorizedJetCorrector): per-run jet energy corrections for real data
        jec_mc (coffea.jetmet_tools.FactorizedJetCorrector): jet energy corrections for MC
        jer (coffea.jetmet_tools.JetResolution): the jet energy resolution correction object
        jersf (coffea.jetmet_tools.JetResolutionScaleFactor): the data-to-mc scale factor for jet energy resolution
        jesunc (coffea.jetmet_tools.JetCorrectionUncertainty): data-to-MC scale factor for jet energy scale uncertainties
    """
    
    def __init__(
        self,
        jec_tag: str,
        jec_tag_data: dict,
        jer_tag: str,
        do_factorized_jec=True):
        """Loads the JEC and JER corrections corresponding to the given tags from txt files.
        
        Args:
            jec_tag (str): Tag for the jet energy corrections for MC simulation
            jec_tag_data (dict): per-run JEC tags for real data
            jer_tag (str): tag for the jet energy resolution
            do_factorized_jec (bool, optional): If True, loads and enables the factorized jet
                energy corrections instead of the total
        """
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

    """Stores the various external calibrations, corrections and MVA models for the analysis.
    
    Attributes:
        bdt01j_ucsd (dict of str->GBREvaluator): UCSD BDT for the 0 and 1-jet bins for different years
        bdt2j_ucsd (dict of str->GBREvaluator): UCSD BDT for the 2-jet bin for different years
        bdt_ucsd (GBREvaluator): Baseline UCSD BDT
        btag_weights (dict of str->BTagWeights): b-tagging corrections for different years
        dnn_model (keras.Model): Caltech baseline DNN model
        dnn_normfactors (numpy.array): normalization factors for the Caltech DNN
        dnnPisa_models (list of keras.Model): The set of Pisa DNN models
        dnnPisa_normfactors1 (numpy.array): Normalization factors for node1 of the Pisa DNN
        dnnPisa_normfactors2 (numpy.array): Normalization factors for node2 of the Pisa DNN
        hrelresolution (hmumu_lib.hRelResolution): The Higgs resolution corrections
        jetmet_corrections (analysis_hmumu.JetMetCorrections): The jet energy scale (JES) and resolution (JER) corrections and uncertainties/scale-factors
        lepeff_trig_data (dict of str->hmumu_lib.LeptonEfficiencyCorrections): Lepton efficiency corrections for data for each year 
        lepeff_trig_mc (dict of str->hmumu_lib.LeptonEfficiencyCorrections): Lepton efficiency corrections for data for each year 
        lepsf_id (dict of str->hmumu_lib.LeptonEfficiencyCorrections): Lepton ID corrections for each year 
        lepsf_iso (dict of str->hmumu_lib.LeptonEfficiencyCorrections): Lepton iso corrections for each year 
        libhmm (hmumu_lib.LibHMuMu): The Python wrapper for the C++ helper library defined in hmumu_lib
        lumidata (LumiData): The integrated luminosity information for each lumi block, loaded from a csv file
        lumimask (LumiMask): The JSON of good luminosity blocks
        miscvariables (hmumu_lib.MiscVariables): The python wrapper for the C++ helper library for various tricky variables
        nnlopsreweighting (hmumu_lib.NNLOPSReweighting): The python wrapper for the C++ helper library for NNLO Parton Shower reweighting
        pu_corrections (dict of str->Histogram): Data PU histograms for each year
        puidreweighting (coffea.lookup_tools.evaluator): pileup ID reweighting evaluator from coffea
        ratios_dataera (dict of str->list): Luminosity ratios for the datataking periods used for weighting lepton scale factors
        rochester_corrections (dict of str->hmumu_lib.RochesterCorrections): Per-era python wrapper for the C++ helper library for Rochester corrections to muon momentum 
        zptreweighting (hmumu_lib.ZpTReweighting): Python wrapper for the C++ library for Z-boson pT reweighting
    """
    
    def __init__(self, args, do_tensorflow=True, gpu_memory_fraction=0.2):
        """Summary
        
        Args:
            args (Namespace): Command line arguments from analysis_hmumu.parse_args
            do_tensorflow (bool, optional): If True, enable the use of tensorflow-based DNN evaluation
            gpu_memory_fraction (float, optional): What fraction of GPU memory to allocate to tensorflow
        """
        self.lumimask = {
            "2016": LumiMask("data/Cert_271036-284044_13TeV_ReReco_07Aug2017_Collisions16_JSON.txt", np, backend_cpu),
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

        #Luminosity weight ratios for computing lepton scale factors
        self.ratios_dataera = {
            #BCDEF, GH
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
        self.lepeff_trig_data = {
            "2016": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2016/EfficienciesAndSF_RunBtoF.root", "data/leptonSF/2016/EfficienciesAndSF_RunGtoH.root"],
                ["IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA", "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA"],
                self.ratios_dataera["2016"]),
            "2017": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root"],
                ["IsoMu27_PtEtaBins/efficienciesDATA/pt_abseta_DATA"],
                [1.0]
            ),
            "2018": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2018/EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root"],
                ["IsoMu24_PtEtaBins/efficienciesDATA/pt_abseta_DATA"],
                [1.0]
            ),
        }

        self.lepeff_trig_mc = {
            "2016": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2016/EfficienciesAndSF_RunBtoF.root", "data/leptonSF/2016/EfficienciesAndSF_RunGtoH.root"],
                ["IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC", "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC"],
                self.ratios_dataera["2016"]),
            "2017": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root"],
                ["IsoMu27_PtEtaBins/efficienciesMC/pt_abseta_MC"],
                [1.0]
            ),
            "2018": LeptonEfficiencyCorrections(self.libhmm,
                ["data/leptonSF/2018/EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root"],
                ["IsoMu24_PtEtaBins/efficienciesMC/pt_abseta_MC"],
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
                    do_factorized_jec=True),
            },
            "2018": {
                "Autumn18_V19":
                    JetMetCorrections(
                    jec_tag="Autumn18_V19_MC",
                    jec_tag_data={
                        "RunA": "Autumn18_RunA_V19_DATA",
                        "RunB": "Autumn18_RunB_V19_DATA",
                        "RunC": "Autumn18_RunC_V19_DATA",
                        "RunD": "Autumn18_RunD_V19_DATA",
                    },
                    jer_tag="Autumn18_V7_MC",
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
            config = tf.compat.v1.ConfigProto()
            config.intra_op_parallelism_threads=1
            config.inter_op_parallelism_threads=1

            if not args.use_cuda: 
                os.environ["CUDA_VISIBLE_DEVICES"]="-1"
            else:
                config.gpu_options.allow_growth = False
                config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

            #load DNN model
            import keras
            self.dnn_model = keras.models.load_model("data/DNN27vars_sig_vbf_ggh_bkg_dyvbf_dy105To160_ewk105To160_split_60_40_mod10_200109.h5")
            self.dnn_normfactors = np.load("data/DNN27vars_sig_vbf_ggh_bkg_dyvbf_dy105To160_ewk105To160_split_60_40_mod10_200109.npy")

            if args.use_cuda:
                import cupy
                self.dnn_normfactors = cupy.array(self.dnn_normfactors[0]), cupy.array(self.dnn_normfactors[1])
            
            for imodel in range(4):
                dnnPisa_model = keras.models.load_model("data/PisaDNN/model_preparation/nn_evt"+str(imodel)+"_february.h5")
                self.dnnPisa_models += [dnnPisa_model]
                self.dnnPisa_normfactors1 = np.load("data/PisaDNN/model_preparation/Febhelp_node1.npy")
                self.dnnPisa_normfactors2 = np.load("data/PisaDNN/model_preparation/Febhelp_node2.npy")
                if args.use_cuda:
                    import cupy
                    self.dnnPisa_normfactors1 = cupy.array(self.dnnPisa_normfactors1[0]), cupy.array(self.dnnPisa_normfactors1[1])
                    self.dnnPisa_normfactors2 = cupy.array(self.dnnPisa_normfactors2[0]), cupy.array(self.dnnPisa_normfactors2[1])
                    
        print("Loading UCSD BDT model")
        #self.bdt_ucsd = GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2016/TMVAClassification_BDTG.weights.2jet_bveto_withmass.xml")
        #self.bdt2j_ucsd = {
        #    "2016": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2016/TMVAClassification_BDTG.weights.2jet_bveto.xml"),
        #    "2017": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2017/TMVAClassification_BDTG.weights.2jet_bveto.xml"),
        #    "2018": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2018/TMVAClassification_BDTG.weights.2jet_bveto.xml")
        #}
        #self.bdt01j_ucsd = {
        #    "2016": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2016/TMVAClassification_BDTG.weights.01jet.xml"),
        #    "2017": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2017/TMVAClassification_BDTG.weights.01jet.xml"),
        #    "2018": GBREvaluator(self.libhmm, "data/Hmm_BDT_xml/2018/TMVAClassification_BDTG.weights.01jet.xml")
        #}
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

        systs = ["jes", "lfstats1", "lfstats2", "hfstats1", "hfstats2", "cferr1", "cferr2", "lf", "hf"]
        systs_sdir=[]
        for sdir in ["up", "down"]:
            for syst in systs:
                systs_sdir += [sdir + "_" + syst]
            
        print("Extracting b-tag weights...")
        self.btag_weights = {
            "DeepCSV_2016": BTagCalibration(self.libhmm, "DeepCSV", "data/btagSF/DeepCSV_2016LegacySF_V1.csv" , systs_sdir),
            "DeepCSV_2017": BTagCalibration(self.libhmm, "DeepCSV", "data/btagSF/DeepCSV_94XSF_V5_B_F.csv", systs_sdir),
            "DeepCSV_2018": BTagCalibration(self.libhmm, "DeepCSV", "data/btagSF/DeepCSV_102XSF_V1.csv", systs_sdir)
            #"DeepCSV_2016": BTagScaleFactor("data/btagSF/DeepCSV_2016LegacySF_V1.csv", BTagScaleFactor.RESHAPE, 'iterativefit,iterativefit,iterativefit', keep_df=True),
            #"DeepCSV_2017": BTagScaleFactor("data/btagSF/DeepCSV_94XSF_V5_B_F.csv", BTagScaleFactor.RESHAPE, 'iterativefit,iterativefit,iterativefit', keep_df=True),
            #"DeepCSV_2018": BTagScaleFactor("data/btagSF/DeepCSV_102XSF_V1.csv", BTagScaleFactor.RESHAPE, 'iterativefit,iterativefit,iterativefit', keep_df=True)
            #"DeepCSV_2016": BTagWeights(tag_name="DeepCSV_2016LegacySF_V1"),
            #"DeepCSV_2017": BTagWeights(tag_name="DeepCSV_94XSF_V4_B_F"),
            #"DeepCSV_2018": BTagWeights(tag_name="DeepCSV_102XSF_V1")
        }


def check_and_recreate_filename_cache(cache_filename: str, datapath: str, datasets: List[Dict[str, str]], use_merged: bool):
    """Creates the list of all filenames for each dataset.
    This can involve a substantial crawling of the filesystem, so we save (cache) the results in a JSON file.
    
    Args:
        cache_filename (str): Path to a json file that contains the full list of filenames for each dataset
        datapath (str): Base directory from where to load the datasets
        datasets (List[Dict[str, str]]): List of all the datasets to process
        use_merged (bool): if True, use the skimmed+merged files from 'files_merged' for each dataset, otherwise use raw NanoAOD
    
    """

    #Check if the cache file already exists 
    if os.path.isfile(cache_filename):
        print("Cache file {0} already exists, we will not overwrite it to be safe.".format(cache_filename), file=sys.stderr)
        print("Delete it to rescan the filesystem for dataset files.", file=sys.stderr)
        return

    #It didn't, so we need to crawl the filesystem for each dataset
    filenames_cache = {}
    for dataset in datasets:
        dataset_name = dataset["name"]
        dataset_era = dataset["era"]
        if use_merged:
            dataset_globpattern = dataset["files_merged"]
        else:
            dataset_globpattern = dataset["files_nano_in"]

        filenames_all = glob.glob(datapath + dataset_globpattern, recursive=True)

        #Remove any Friend ntuples - probably not needed
        filenames_all = [fn for fn in filenames_all if "Friend" not in fn]

        filenames_cache[dataset_name + "_" + dataset_era] = [
            fn.replace(datapath, "") for fn in filenames_all]

        #We didn't find any files for this dataset, most likely there is a problem
        if len(filenames_all) == 0:
            raise Exception("Dataset {0} matched 0 files from glob pattern {1}, verify that the data files are located in {2}".format(
                dataset_name, dataset_globpattern, datapath
            ))

    #save all dataset filenames to a json file 
    print("Creating a json dump of all the dataset filenames based on data found in {0}".format(datapath))
    with open(cache_filename, "w") as fi:
        fi.write(json.dumps(filenames_cache, indent=2))

    return


def create_all_jobfiles(datasets: List[Dict], cache_filename: str, datapath: str, chunksize: str, outpath: str):
    """Splits the dataset into job descriptions, specifying how many files will be processed per job.
    The job descriptions will be saved to small JSON fioles for batch processing.
    
    Args:
        datasets (List[Dataset]): The dataset for which to create the job files
        cache_filename (str): Input json filename where the filenames for each dataset are loaded from
        datapath (str): Path to load the data from
        chunksize (int): Number of files to process in each job
        outpath (str): Path with the output directory where the jobfiles will be stored
    
    """
    jobfile_path = outpath + "/jobfiles"
    if os.path.isdir(jobfile_path):
        print("Jobfiles directory {0} already exists, skipping jobfile creation".format(jobfile_path))
        return
    os.makedirs(jobfile_path)

    #Create a list of job files for processing
    jobfile_data = []
    print("Loading list of filenames from {0}".format(cache_filename))
    if not os.path.isfile(cache_filename):
        raise Exception("Cached dataset list of filenames not found in {0}, please run this code with --action cache".format(
            cache_filename))
    filenames_cache = json.load(open(cache_filename, "r"))

    seed_gen = seed_generator()
    for dataset in sorted(datasets, key=lambda x: (x["name"], x["era"])):
        dataset_name = dataset["name"]
        dataset_era = dataset["era"]
        is_mc = dataset["is_mc"]
        
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
            filenames_all_full, is_mc, chunksize, jobfile_path, seed_gen)
        jobfile_data += jobfile_dataset
        print("Dataset {0}_{1} consists of {2} chunks".format(
            dataset_name, dataset_era, len(jobfile_dataset)))

    assert(len(jobfile_data) > 0)
    assert(len(jobfile_data[0]["filenames"]) > 0)


def load_jobfiles(
    datasets: List[Dict[str, str]],
    jobfiles_load_from_file: str,
    jobfiles: List[str],
    maxchunks: int,
    outpath: str):

    """Loads the specified job files.
    Each job file consists of filenames to process, plus additional information such as the dataset name, era and such.
    
    Args:
        datasets (List[Dataset]): List of datasets for which to load the jobfiles
        jobfiles_load_from_file (str): Filename to load the jobfiles from, if None, use filenames from 
        jobfiles (list of str): List of filenames from which to load job data
        maxchunks (int): Max number of files per dataset to process
        outpath (str): Path where jobfiles are loaded from
    
    Returns: List[Dict] of the job files to process
    """

    #Load list of jobfiles from a single text file
    if not (jobfiles_load_from_file is None):
        jobfiles = [l.strip() for l in open(jobfiles_load_from_file).readlines()]

    #Check for existing jobfiles in {outpath}/jobfiles/*.json
    if jobfiles is None:
        print("You did not specify to process specific jobfiles, assuming you want to process all")
        print("If this is not true, please specify e.g. --jobfiles data_2018_0.json data_2018_1.json ...")
        jobfiles = []
        for dataset in datasets:
            dataset_name = dataset["name"]
            dataset_era = dataset["era"]

            jobfile_pattern = outpath + "/jobfiles/{0}_{1}_*.json".format(dataset_name, dataset_era)
            jobfiles_dataset = glob.glob(jobfile_pattern)
            if len(jobfiles_dataset) == 0:
                raise Exception("Could not find any jobfiles matching pattern {0}".format(jobfile_pattern))

            if maxchunks > 0:
                jobfiles_dataset = jobfiles_dataset[:maxchunks]
            jobfiles += jobfiles_dataset
    
    #Now actually load the job description data
    assert(len(jobfiles) > 0)
    print("You specified --jobfiles {0}, processing only these jobfiles".format(" ".join(jobfiles))) 
    jobfile_data = []
    for f in jobfiles:
        jobfile_data += [json.load(open(f))]

    chunkstr = " ".join(["{0}_{1}_{2}".format(
        ch["dataset_name"], ch["dataset_era"], ch["dataset_num_chunk"])
        for ch in jobfile_data])
    print("Will process {0} jobfiles: {1}".format(len(jobfile_data), chunkstr))
    assert(len(jobfile_data) > 0)
    return jobfile_data


def merge_partial_results(dataset_name: str, dataset_era: str, outpath: str, outpath_partial: str):
    """Merges the output from separate jobs for each dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_era (str): Dataset era
        outpath (str): Directory with the output results
        outpath_partial (str): Directory with the partial input results
    """
    results = []
    partial_results = glob.glob(outpath_partial + "/{0}_{1}_*.pkl".format(dataset_name, dataset_era))
    print("Merging {0} partial results for dataset {1}_{2}".format(len(partial_results), dataset_name, dataset_era))

    #Load all thge partial results
    for res_file in partial_results:
        res = pickle.load(open(res_file, "rb"))
        results += [res]

    #Merge the partial results
    results = sum(results, Results({}))

    #Create output directory if it does not exist
    try:
        os.makedirs(outpath + "/results")
    except FileExistsError:
        print("Output directory {} already exists".format(outpath))

    result_filename = outpath + "/results/{0}_{1}.pkl".format(dataset_name, dataset_era)
    print("Saving results to {0}".format(result_filename))
    with open(result_filename, "wb") as fi:
        pickle.dump(results, fi, protocol=pickle.HIGHEST_PROTOCOL) 


def main(args):
    do_prof = args.do_profile
    do_tensorflow = not args.disable_tensorflow

    # use the environment variable for cupy/cuda choice
    args.use_cuda = USE_CUPY

    datasets = yaml.load(open(args.datasets_yaml), Loader=yaml.FullLoader)["datasets"]
    
    # Filter datasets by era
    datasets_to_process = []
    for ds in datasets:
        if args.datasets is None or ds["name"] in args.datasets:
            if args.eras is None or ds["era"] in args.eras:
                datasets_to_process += [ds]
    if len(datasets_to_process) == 0:
        raise Exception("No datasets considered, please check the --datasets and --eras options")
    datasets = datasets_to_process

    # Choose either the CPU or GPU(CUDA) backend
    hmumu_utils.NUMPY_LIB, hmumu_utils.ha = choose_backend(args.use_cuda)
    Dataset.numpy_lib = hmumu_utils.NUMPY_LIB

    outpath_partial = "{0}/partial_results".format(args.out)
    try:
        os.makedirs(outpath_partial)
    except FileExistsError:
        print("Output path {0} already exists, not recreating".format(outpath_partial))

    # save the parameters as a pkl file
    from pars import analysis_parameters
    for analysis_name in analysis_parameters.keys():
        analysis_parameters[analysis_name]["do_factorized_jec"] = args.do_factorized_jec
        analysis_parameters[analysis_name]["dnn_vars_path"] = "{0}/dnn_vars".format(args.out)
    with open('{0}/parameters.pkl'.format(outpath_partial), 'wb') as handle:
        pickle.dump(analysis_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Recreate dump of all filenames
    cache_filename = "{0}/datasets.json".format(args.out)

    use_skim = False
    if args.cachepath is None:
        print("--cachepath not specified, will process unskimmed NanoAOD, which is somewhat slower!")
        print("Please see the README.md on how to skim the NanoAOD")
        datapath = args.datapath
    else:
        print("Processing skimmed NanoAOD")
        datapath = args.cachepath
        use_skim = True
    check_and_recreate_filename_cache(cache_filename, datapath, datasets, use_skim)

    # Create the jobfiles
    if args.jobfiles is None:
        create_all_jobfiles(datasets, cache_filename, datapath, args.chunksize, args.out)

    # For each dataset, find out which chunks we want to process
    if "analyze" in args.action:
        jobfile_data = load_jobfiles(datasets, args.jobfiles_load, args.jobfiles, args.maxchunks, args.out)

    # If we want to check what part of the code is slow, start the profiler only in the actual data processing
    if do_prof:
        import yappi
        yappi.set_clock_type('cpu')
        yappi.start(builtins=True)

    # Run the physics analysis on all specified jobfiles  
    if "analyze" in args.action:
        print("Running the 'analyze' step of the analysis, processing the events into histograms with all systematics")
        analysis_corrections = AnalysisCorrections(args, do_tensorflow)
        run_analysis(args, outpath_partial, jobfile_data, analysis_parameters, analysis_corrections)
    
    if do_prof:
        stats = yappi.get_func_stats()
        stats.save("analysis.prof", type='callgrind')

    # Merge the partial results (pieces of each dataset)
    if "merge" in args.action:
        with ProcessPoolExecutor(max_workers=args.nthreads) as executor:
            for dataset in datasets:
                dataset_name = dataset["name"]
                dataset_era = dataset["era"]
                executor.submit(merge_partial_results, dataset_name, dataset_era, args.out, outpath_partial)
        print("done merging")

    # print memory usage for debugging
    total_memory = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    total_memory += resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("maxrss={0} MB".format(total_memory/1024))


if __name__ == "__main__":
    args = parse_args()
    main(args)
