import numpy as np
import uproot
from argparse import Namespace
import os

from hepaccelerate.utils import choose_backend
import hmumu_utils
from hmumu_utils import JetTransformer
from hmumu_utils import create_datastructure, create_dataset, get_genpt_cpu
from analysis_hmumu import AnalysisCorrections

def make_plots(filename, filename_nano_postproc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t1 = uproot.open(filename).get("Jets")
    #Produce with NanoAODTools
    #python scripts/nano_postproc.py ./ /storage/group/allcit/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/100000/359F045D-D71C-E84E-9BD1-0BEA8E6228C5.root -I PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetmetUncertainties jetmetUncertainties2018 --friend
    t2 = uproot.open(filename_nano_postproc).get("Friends")
    data1 = {
        "pt_raw": t1.array("pt_raw"),
        "pt_gen": t1.array("pt_gen"),
        "corr_JEC": t1.array("corr_JEC"),
        "corr_JER": t1.array("corr_JER"),
        "pt_nom": t1.array("pt_nom"),
    }
    
    data2 = {
        "pt_raw": t2.array("Jet_pt_raw").content,
        "corr_JEC": t2.array("Jet_corr_JEC").content,
        "corr_JER": t2.array("Jet_corr_JER").content,
        "pt_nom": t2.array("Jet_pt_nom").content
    }

    plt.figure()
    bins = np.linspace(0,200,1000)
    plt.hist(data1["pt_raw"], bins=bins, histtype="step", label="our code");
    plt.hist(data2["pt_raw"], bins=bins, histtype="step", label="NanoAODTools");
    plt.xlabel("raw pt")
    plt.savefig("pt_raw.pdf")

    plt.figure()
    bins = np.linspace(0,2,1000)
    plt.hist(data1["corr_JEC"], bins=bins, histtype="step", label="our code");
    plt.hist(data2["corr_JEC"], bins=bins, histtype="step", label="NanoAODTools");
    plt.xlabel("JEC correction")
    plt.savefig("corr_JEC.pdf")

    plt.figure()
    bins = np.linspace(0,2,1000)
    plt.hist(data1["corr_JER"], bins=bins, histtype="step", label="our code");
    plt.hist(data2["corr_JER"], bins=bins, histtype="step", label="NanoAODTools");
    plt.xlabel("JER correction")
    plt.savefig("corr_JER.pdf")

if __name__ == "__main__":
    use_cuda = False
    NUMPY_LIB, ha = choose_backend(use_cuda)
    hmumu_utils.NUMPY_LIB = np
    hmumu_utils.ha = ha
    
    job_desc = {
        "dataset_name": "ggh_amcPS",
        "is_mc": True,
        "dataset_era": "2018",
        "filenames": ["/store/mc/RunIIAutumn18NanoAODv5/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/100000/359F045D-D71C-E84E-9BD1-0BEA8E6228C5.root", ],
        "dataset_num_chunk": 0,
    }
    cache_location = "/storage/user/nlu/hmm/cache2"
    datapath = "/storage/group/allcit/"
    
    datastructures = create_datastructure(job_desc["dataset_name"], job_desc["is_mc"], job_desc["dataset_era"])
    
    ds = create_dataset(
        job_desc["dataset_name"],
        job_desc["filenames"],
        datastructures,
        cache_location,
        datapath,
        job_desc["is_mc"])
    
    ds.era = job_desc["dataset_era"]
    ds.numpy_lib = NUMPY_LIB
    
    ds.filenames = [datapath + fn for fn in ds.filenames]
    ds.load_root(verbose=True)
    
    jets = ds.structs["Jet"][0]
    genJet = ds.structs["GenJet"][0]
    jets.attrs_data["genpt"] = NUMPY_LIB.zeros(jets.numobjects(), dtype=NUMPY_LIB.float32)
    get_genpt_cpu(jets.offsets, jets.genJetIdx, genJet.offsets, genJet.pt, jets.attrs_data["genpt"])
    scalars = ds.eventvars[0]
    
    args = Namespace(nthreads=1, use_cuda=use_cuda)
    
    corrections = AnalysisCorrections(args, do_tensorflow=False)
    parameters = {}
   
    #Run the nominal JEC + JER 
    jet_transformer = JetTransformer(
        jets, scalars,
        parameters,
        corrections.jetmet_corrections[job_desc["dataset_era"]]["Autumn18_V16"],
        NUMPY_LIB, ha, use_cuda, job_desc["is_mc"])
   
    #save output 
    with uproot.recreate("jets_ha.root") as f:
        f["Jets"] = uproot.newtree({
            "pt_raw": "float32",
            "pt_gen": "float32",
            "corr_JEC": "float32",
            "corr_JER": "float32",
            "pt_nom": "float32",
        })
        f["Jets"].extend({
            "pt_raw": jet_transformer.raw_pt,
            "pt_gen": jets.genpt,
            "corr_JEC": jet_transformer.corr_jec,
            "corr_JER": jet_transformer.jer_nominal,
            "pt_nom": jet_transformer.pt_jec_jer,
        })

    #make comparison plots
    fn_nano_postproc = "/storage/user/jpata/hmm/dev/CMSSW_10_3_4/src/PhysicsTools/NanoAODTools/359F045D-D71C-E84E-9BD1-0BEA8E6228C5_Friend.root"
    if os.path.isfile(fn_nano_postproc):
        make_plots("jets_ha.root", fn_nano_postproc)
    else:
        print("Could not find baseline nanoaod postprocessing file {0}, please produce it".format(fn_nano_postproc))
