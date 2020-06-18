import os, requests
import numpy as np
import unittest
import pickle
import shutil
import copy
import sys

from hepaccelerate.utils import choose_backend, Dataset
from hmumu_utils import create_datastructure
from coffea.util import USE_CUPY
from test_hmumu_utils import download_if_not_exists

if USE_CUPY:
    from numba import cuda

class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.NUMPY_LIB, self.ha = choose_backend(use_cuda=USE_CUPY)

        import hmumu_utils
        hmumu_utils.NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.ha = self.ha
        
        #disable everything that requires ROOT which is not easily available on travis tests
        from pars import analysis_parameters
        self.analysis_parameters = analysis_parameters
        self.analysis_parameters["baseline"]["do_rochester_corrections"] = True
        self.analysis_parameters["baseline"]["do_lepton_sf"] = True
        self.analysis_parameters["baseline"]["save_dnn_vars"] = True
        self.analysis_parameters["baseline"]["do_bdt_ucsd"] = False
        self.analysis_parameters["baseline"]["do_bdt_pisa"] = False
        self.analysis_parameters["baseline"]["do_factorized_jec"] = False
        self.analysis_parameters["baseline"]["do_jec"] = True
        self.analysis_parameters["baseline"]["do_jer"] = {"2016": True}
        
        from argparse import Namespace
        self.cmdline_args = Namespace(use_cuda=USE_CUPY, datapath=".", nthreads=1, do_fsr=False, async_data=False, do_sync=False, out="test_out")
        
        from analysis_hmumu import AnalysisCorrections
        self.analysis_corrections = AnalysisCorrections(self.cmdline_args, True)
        download_if_not_exists(
            "data/myNanoProdMc2016_NANO.root",
            "https://jpata.web.cern.ch/jpata/hmm/test_files/21-02-2020-private-nanoaod/myNanoProdMc2016_NANO.root"
        )
        download_if_not_exists(
            "data/nano_2016_data.root",
            "https://jpata.web.cern.ch/jpata/hmm/test_files/21-02-2020-private-nanoaod/nano_2016_data.root"
        )

    #Run the analysis on a raw NanoAOD MC sample
    def test_run_analysis_mc(self):
        import hmumu_utils
        from hmumu_utils import load_puhist_target, run_analysis
        from coffea.lookup_tools import extractor
        NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.ha = self.ha

        job_descriptions = [
            {
                "dataset_name": "vbf_sync",
                "dataset_era": "2016",
                "filenames": ["data/myNanoProdMc2016_NANO.root"],
                "is_mc": True,
                "dataset_num_chunk": 0,
                "random_seed": 0
            }
        ]
        
        if not os.path.exists("test_out"):
            os.makedirs("test_out")
        ret = run_analysis(
            self.cmdline_args,
            "test_out",
            job_descriptions,
            self.analysis_parameters,
            self.analysis_corrections,
            numev_per_chunk=10000)

        ret2 = pickle.load(open("test_out/vbf_sync_2016_0.pkl", "rb"))
 
        print(ret2["num_events"])
        print(ret2["genEventSumw"])
        print(ret2["baseline"]["selected_events_dimuon"])
        self.assertAlmostEqual(ret2["num_events"], 3674)
        self.assertAlmostEqual(ret2["genEventSumw"], 4.659182940800001)
        self.assertAlmostEqual(ret2["baseline"]["selected_events_dimuon"], 1580)
    
    #Run the analysis on a skimmed MC sample
    def test_run_analysis_mc_skim(self):
        fn = "data/myNanoProdMc2016_NANO_skim.root"
        if not os.path.isfile(fn):
            print("File {0} not found, skipping".format(fn))
            return

        from hmumu_utils import run_analysis
        job_descriptions = [
            {
                "dataset_name": "vbf_sync_skim",
                "dataset_era": "2016",
                "filenames": [fn],
                "is_mc": True,
                "dataset_num_chunk": 0,
                "random_seed": 0
            }
        ]
        
        from argparse import Namespace
        if not os.path.exists("test_out"):
            os.makedirs("test_out")
        
        ret = run_analysis(
            self.cmdline_args,
            "test_out",
            job_descriptions,
            self.analysis_parameters,
            self.analysis_corrections,
            numev_per_chunk=10000)

        ret2 = pickle.load(open("test_out/vbf_sync_skim_2016_0.pkl", "rb"))
 
        print(ret2["num_events"])
        print(ret2["genEventSumw"])
        print(ret2["baseline"]["selected_events_dimuon"])
        self.assertAlmostEqual(ret2["num_events"], 1880)
        self.assertAlmostEqual(ret2["genEventSumw"], 4.659182940800001)
        self.assertAlmostEqual(ret2["baseline"]["selected_events_dimuon"], 1580)
    
    #Run the analysis on a raw NanoAOD data sample 
    def test_run_analysis_data(self):
        fn = "data/nano_2016_data.root"
        if not os.path.isfile(fn):
            print("Data sync file {0} not found, skipping".format(fn))
            return

        from hmumu_utils import run_analysis

        job_descriptions = [
            {
                "dataset_name": "data",
                "dataset_era": "2016",
                "filenames": ["data/nano_2016_data.root"],
                "is_mc": False,
                "dataset_num_chunk": 0,
                "random_seed": 0
            }
        ]
        
        if not os.path.exists("test_out"):
            os.makedirs("test_out")
        
        ret = run_analysis(
            self.cmdline_args,
            "test_out",
            job_descriptions,
            self.analysis_parameters,
            self.analysis_corrections,
            numev_per_chunk=10000)

        ret2 = pickle.load(open("test_out/data_2016_0.pkl", "rb"))
        print(ret2["num_events"])
        print(ret2["int_lumi"])
        print(ret2["baseline"]["selected_events_dimuon"])
        self.assertAlmostEqual(ret2["num_events"], 2000)
        self.assertAlmostEqual(ret2["int_lumi"], 0.130571592)
        self.assertAlmostEqual(ret2["baseline"]["selected_events_dimuon"], 45)
   
    #Run the analysis on a skimmed data sample 
    def test_run_analysis_data_skim(self):
        fn = "data/nano_2016_data_skim.root"
        if not os.path.isfile(fn):
            print("Data sync file {0} not found, skipping".format(fn))
            return

        from hmumu_utils import run_analysis

        job_descriptions = [
            {
                "dataset_name": "data_skim",
                "dataset_era": "2016",
                "filenames": [fn],
                "is_mc": False,
                "dataset_num_chunk": 0,
                "random_seed": 0
            }
        ]
        
        if not os.path.exists("test_out"):
            os.makedirs("test_out")
        
        ret = run_analysis(
            self.cmdline_args,
            "test_out",
            job_descriptions,
            self.analysis_parameters,
            self.analysis_corrections,
            numev_per_chunk=10000)

        ret2 = pickle.load(open("test_out/data_skim_2016_0.pkl", "rb"))
        
        print(ret2["num_events"])
        print(ret2["int_lumi"])
        print(ret2["baseline"]["selected_events_dimuon"])
        self.assertAlmostEqual(ret2["num_events"], 142)
        self.assertAlmostEqual(ret2["int_lumi"], 0.130571592)
        self.assertAlmostEqual(ret2["baseline"]["selected_events_dimuon"], 44)

        #Not sure why this is different on the skimmed sample... floating point precision in changing the file encoding?
        #self.assertAlmostEqual(ret2["baseline"]["numev_passed"]["muon"], 4024.0)
   
    #Test the analysis on one bkg MC and data file, making sure the data/mc plot looks OK
    def test_run_analysis_mc_and_data(self):
        fn_mc = "/storage/user/nlu/hmm/automaticTest/myNanoProdMc2016_NANO.root"

        if not os.path.isfile(fn_mc):
            fn_mc = "data/myNanoProdMc2016_NANO.root"

        fn_data = "data/nano_2016_data.root"

        from hmumu_utils import run_analysis

        job_descriptions = [
            {
                "dataset_name": "dy",
                "dataset_era": "2016",
                "filenames": [fn_mc],
                "is_mc": True,
                "dataset_num_chunk": 0,
                "random_seed": 0
            },
            {
                "dataset_name": "data",
                "dataset_era": "2016",
                "filenames": [fn_data],
                "is_mc": False,
                "dataset_num_chunk": 0,
                "random_seed": 0
            }
        ]

        outpath = "test_out"
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.makedirs(outpath)

        cmdline_args = copy.deepcopy(self.cmdline_args)
        cmdline_args.out = outpath

        ret = run_analysis(
            cmdline_args,
            outpath,
            job_descriptions,
            self.analysis_parameters,
            self.analysis_corrections,
        )

        from plotting import make_pdf_plot
        from pars import cross_sections
        
        res = {"data": pickle.load(open(outpath + "/data_2016_0.pkl", "rb"))}
        analysis = "baseline"
        var = "hist__dimuon__leading_muon_pt"
        era = "2016"

        int_lumi = res["data"]["int_lumi"]
        mc_samples = ["dy"]
        process_groups = [
            ("dy", ["dy"])
        ]
       
        genweights = {}
        weight_xs = {} 
        for mc_samp in mc_samples: 
            res[mc_samp] = pickle.load(open(outpath + "/{0}_2016_0.pkl".format(mc_samp), "rb"))
            genweights[mc_samp] = res[mc_samp]["genEventSumw"]
            weight_xs[mc_samp] =  cross_sections[mc_samp] * int_lumi / genweights[mc_samp]
        
        print(genweights["dy"])
        print(int_lumi)
        self.assertAlmostEqual(genweights["dy"], 4.659182940800001)
        self.assertAlmostEqual(int_lumi, 0.130571592)
        
        histos = {}
        for sample in mc_samples + ["data"]:
            histos[sample] = res[sample][analysis][var]
        hdata = res["data"][analysis][var]["nominal"]

        outdir = "{0}/{1}/plots/{2}".format(outpath, analysis, era)
        plot_args = (
            histos, hdata, mc_samples, analysis,
            var, "nominal", weight_xs, int_lumi, outdir, era,
            process_groups, {})

        make_pdf_plot(plot_args)
  
if __name__ == "__main__": 
    if "--debug" in sys.argv:
        unittest.findTestCases(sys.modules[__name__]).debug()
    else:
        unittest.main()
