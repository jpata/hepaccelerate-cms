import os, requests
import numpy as np
import unittest
from hepaccelerate.utils import choose_backend, Dataset
from hmumu_utils import create_datastructure

use_cuda = False

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

class TestAnalysisMC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.NUMPY_LIB, self.ha = choose_backend(use_cuda=use_cuda)
        download_if_not_exists(
            "data/myNanoProdMc2016_NANO.root",
            "https://jpata.web.cern.ch/jpata/hmm/test_files/myNanoProdMc2016_NANO.root"
        )
        self.datastructures = create_datastructure("vbf_sync", True, "2016", do_fsr=True)
        self.dataset = Dataset(
            "vbf_sync",
            ["data/myNanoProdMc2016_NANO.root"],
            self.datastructures,
            cache_location=".",
            datapath="",
            treename="Events",
            is_mc=True)
        self.dataset.num_chunk = 0
        self.dataset.era = "2016"
        self.dataset.load_root()

    def setUp(self):
        pass

    def testDataset(self):
        nev = self.dataset.numevents()
        print("Loaded dataset from {0} with {1} events".format(self.dataset.filenames[0], nev))
        assert(nev>0)

    def test_get_genpt(self):
        from hmumu_utils import get_genpt_cpu
        NUMPY_LIB = self.NUMPY_LIB

        muons = self.dataset.structs["Muon"][0]
        genpart = self.dataset.structs["GenPart"][0]
        muons_genpt = NUMPY_LIB.zeros(muons.numobjects(), dtype=NUMPY_LIB.float32)
        get_genpt_cpu(muons.offsets, muons.genPartIdx, genpart.offsets, genpart.pt, muons_genpt)
        self.assertAlmostEqual(NUMPY_LIB.sum(muons_genpt), 11943932)
        self.assertListEqual(list(muons_genpt[:10]), [105.0, 30.4375, 0.0, 0.0, 140.5, 28.625, 102.75, 41.25, 120.5, 80.5])

    def test_fix_muon_fsrphoton_index(self):
        from hmumu_utils import fix_muon_fsrphoton_index
        NUMPY_LIB = self.NUMPY_LIB
        
        muons = self.dataset.structs["Muon"][0]
        fsrphotons = self.dataset.structs["FsrPhoton"][0]
        
        out_muons_fsrPhotonIdx = NUMPY_LIB.array(muons.fsrPhotonIdx)
        fix_muon_fsrphoton_index(
            fsrphotons.offsets,
            muons.offsets,
            fsrphotons.dROverEt2,
            fsrphotons.muonIdx,
            muons.fsrPhotonIdx,
            out_muons_fsrPhotonIdx
        )
        print(muons.fsrPhotonIdx)
        print(out_muons_fsrPhotonIdx)

    def test_analyze_data(self):
        import hmumu_utils
        from hmumu_utils import analyze_data, load_puhist_target
        from analysis_hmumu import JetMetCorrections
        from pars import analysis_parameters
        from coffea.lookup_tools import extractor
        NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.NUMPY_LIB = self.NUMPY_LIB
        hmumu_utils.ha = self.ha

        #disable everything that requires ROOT which is not easily available on travis tests
        analysis_parameters["baseline"]["do_rochester_corrections"] = False
        analysis_parameters["baseline"]["do_lepton_sf"] = False
        analysis_parameters["baseline"]["save_dnn_vars"] = False
        analysis_parameters["baseline"]["do_bdt_ucsd"] = False
        analysis_parameters["baseline"]["do_bdt_pisa"] = False
        analysis_parameters["baseline"]["do_factorized_jec"] = False
        analysis_parameters["baseline"]["do_jec"] = True
        analysis_parameters["baseline"]["do_jer"] = {"2016": True}

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
            use_cuda = use_cuda,
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
        self.assertAlmostEqual(nev_zpeak_nominal, 0.013123948)
        
        self.assertTrue("Total__up" in h.keys())
        self.assertTrue("Total__down" in h.keys())
        self.assertTrue("jer__up" in h.keys())
        self.assertTrue("jer__down" in h.keys())

if __name__ == "__main__":
    import sys
    if "--debug" in sys.argv:
        unittest.findTestCases(sys.modules[__name__]).debug()
    else:
        unittest.main()
