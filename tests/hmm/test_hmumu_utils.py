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

if USE_CUPY:
    from numba import cuda

def download_file(filename, url):
    """
    Download an URL to a file
    """
    print("downloading {0}".format(url))
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        # Write response data to file
        iblock = 0
        for block in response.iter_content(4096):
            if iblock % 1000 == 0:
                sys.stdout.write(".");sys.stdout.flush()
            iblock += 1
            fout.write(block)

def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        download_file(filename, url)
        return True
    return False

class TestAnalysisSmall(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.NUMPY_LIB, self.ha = choose_backend(use_cuda=USE_CUPY)

        import hmumu_utils
        hmumu_utils.NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.ha = self.ha

        download_if_not_exists(
            "data/myNanoProdMc2016_NANO.root",
            "https://jpata.web.cern.ch/jpata/hmm/test_files/myNanoProdMc2016_NANO.root"
        )

        #Load a simple sync dataset
        self.datastructures = create_datastructure("vbf_sync", True, "2016", do_fsr=True)
        self.dataset = Dataset(
            "vbf_sync",
            ["data/myNanoProdMc2016_NANO.root"],
            self.datastructures,
            datapath="",
            treename="Events",
            is_mc=True)
        self.dataset.num_chunk = 0
        self.dataset.era = "2016"
        self.dataset.load_root()

        self.dataset.numpy_lib = self.NUMPY_LIB
        self.dataset.move_to_device(self.NUMPY_LIB)
        
        #disable everything that requires ROOT which is not easily available on travis tests
        from pars import analysis_parameters
        self.analysis_parameters = analysis_parameters
        self.analysis_parameters["baseline"]["do_rochester_corrections"] = False
        self.analysis_parameters["baseline"]["do_lepton_sf"] = False
        self.analysis_parameters["baseline"]["save_dnn_vars"] = False
        self.analysis_parameters["baseline"]["do_bdt_ucsd"] = False
        self.analysis_parameters["baseline"]["do_bdt_pisa"] = False
        self.analysis_parameters["baseline"]["do_factorized_jec"] = False
        self.analysis_parameters["baseline"]["do_jec"] = True
        self.analysis_parameters["baseline"]["do_jer"] = {"2016": True}
        
        from argparse import Namespace
        self.cmdline_args = Namespace(use_cuda=USE_CUPY, datapath=".", do_fsr=False, nthreads=1, async_data=False, do_sync=False, out="test_out")
        
        if os.path.isfile("tests/hmm/libhmm.so"):
            from analysis_hmumu import AnalysisCorrections
            self.analysis_corrections = AnalysisCorrections(self.cmdline_args, True)
        else:
            print("Could not load analysis corrections with ROOT, skipping this in further tests")
            self.analysis_corrections = None

    def setUp(self):
        pass

    def testDataset(self):
        nev = self.dataset.numevents()
        print("Loaded dataset from {0} with {1} events".format(self.dataset.filenames[0], nev))
        assert(nev>0)

    def test_get_genpt(self):
        from hmumu_utils import get_genpt_cpu, get_genpt_cuda
        NUMPY_LIB = self.NUMPY_LIB

        muons = self.dataset.structs["Muon"][0]
        genpart = self.dataset.structs["GenPart"][0]
        muons_genpt = NUMPY_LIB.zeros(muons.numobjects(), dtype=NUMPY_LIB.float32)
        if USE_CUPY:
            get_genpt_cuda[32,1024](muons.offsets, muons.genPartIdx, genpart.offsets, genpart.pt, muons_genpt)
            cuda.synchronize()
        else:
            get_genpt_cpu(muons.offsets, muons.genPartIdx, genpart.offsets, genpart.pt, muons_genpt)
        muons_genpt = NUMPY_LIB.asnumpy(muons_genpt)
        self.assertAlmostEqual(NUMPY_LIB.sum(muons_genpt), 11943932)
        self.assertListEqual(list(muons_genpt[:10]), [105.0, 30.4375, 0.0, 0.0, 140.5, 28.625, 102.75, 41.25, 120.5, 80.5])

    def test_fix_muon_fsrphoton_index(self):
        from hmumu_utils import fix_muon_fsrphoton_index
        NUMPY_LIB = self.NUMPY_LIB
        
        muons = self.dataset.structs["Muon"][0]
        fsrphotons = self.dataset.structs["FsrPhoton"][0]
        
        out_muons_fsrPhotonIdx = np.zeros_like(NUMPY_LIB.asnumpy(muons.fsrPhotonIdx))
        fix_muon_fsrphoton_index(
            NUMPY_LIB.asnumpy(fsrphotons.offsets),
            NUMPY_LIB.asnumpy(muons.offsets),
            NUMPY_LIB.asnumpy(fsrphotons.dROverEt2),
            NUMPY_LIB.asnumpy(fsrphotons.muonIdx),
            NUMPY_LIB.asnumpy(muons.fsrPhotonIdx),
            out_muons_fsrPhotonIdx
        )

    def test_analyze_function(self):
        import hmumu_utils
        from hmumu_utils import analyze_data, load_puhist_target
        from analysis_hmumu import JetMetCorrections
        from coffea.lookup_tools import extractor
        NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.ha = self.ha

        analysis_parameters = self.analysis_parameters

        puid_maps = "data/puidSF/PUIDMaps.root"
        puid_extractor = extractor()
        puid_extractor.add_weight_sets(["* * {0}".format(puid_maps)])
        puid_extractor.finalize()
        
        kwargs = {
            "pu_corrections": {"2016": load_puhist_target("data/pileup/RunII_2016_data.root")},
            "puidreweighting": puid_extractor.make_evaluator(),
            "jetmet_corrections": {
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
            },
            "do_fsr": True
        }

        ret = self.dataset.analyze(
            analyze_data,
            use_cuda = USE_CUPY,
            parameter_set_name = "baseline",
            parameters = analysis_parameters["baseline"],
            dataset_era = self.dataset.era,
            dataset_name = self.dataset.name,
            dataset_num_chunk = self.dataset.num_chunk,
            is_mc = self.dataset.is_mc,
            **kwargs
        )
        h = ret["hist__dimuon_invmass_z_peak_cat5__M_mmjj"]
        
        nev_zpeak_nominal = np.sum(h["nominal"].contents)

        if not USE_CUPY:
            self.assertAlmostEqual(nev_zpeak_nominal, 0.012636297, places=4)
        
        self.assertTrue("Total__up" in h.keys())
        self.assertTrue("Total__down" in h.keys())
        self.assertTrue("jer__up" in h.keys())
        self.assertTrue("jer__down" in h.keys())

if __name__ == "__main__":
    if "--debug" in sys.argv:
        unittest.findTestCases(sys.modules[__name__]).debug()
    else:
        unittest.main()
