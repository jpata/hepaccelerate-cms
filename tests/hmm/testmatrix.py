import os, shutil

from analysis_hmumu import datasets, parse_args, main

def setup_baseargs():
    args = parse_args()

    args.cache_location = "./mycache"
    args.chunksize = 1
    args.maxchunks = 4
    args.out = "./out"
    args.datasets = ["data", "dy_m105_160_amc"]
    args.eras = ["2018"]
    return args

def test_cache_clean():
    args = setup_baseargs()
    args.action = ["cache"]
    shutil.rmtree(args.cache_location)
    main(args, datasets)
    assert(os.path.isfile("mycache/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/120000/0A403E5F-9E83-DA44-BEEB-5ED12F995877.GenJet.eta.mmap"))

def test_cache():
    args = setup_baseargs()
    args.action = ["cache"]
    main(args, datasets)
    assert(os.path.isfile("mycache/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/120000/0A403E5F-9E83-DA44-BEEB-5ED12F995877.GenJet.eta.mmap"))

def test_analyze():
    args = setup_baseargs()
    args.action = ["analyze", "merge"]

    shutil.rmtree(args.out)
    main(args, datasets)

    for i in range(args.maxchunks):
        assert(os.path.isfile("{0}/partial_results/data_2018_{1}.pkl".format(args.out, i)))

    assert(os.path.isfile("{0}/results/data_2018.pkl".format(args.out)))

def test_analyze_chunk2():
    args = setup_baseargs()
    args.action = ["analyze", "merge"]
    args.out = "./out2"
    args.chunksize = 2
    args.maxchunks = 2

    shutil.rmtree(args.out)
    main(args, datasets)

    for i in range(args.maxchunks):
        assert(os.path.isfile("{0}/partial_results/data_2018_{1}.pkl".format(args.out, i)))

    assert(os.path.isfile("{0}/results/data_2018.pkl".format(args.out)))

if __name__ == "__main__":
    test_cache()
    test_analyze()
    test_analyze_chunk2()

    for dataset in ["data", "dy_m105_160_amc"]:
        for era in ["2018"]:
            assert(
                shutil.disk_usage("{0}/results/{1}_{2}.pkl".format("out", dataset, era)) ==
                shutil.disk_usage("{0}/results/{1}_{2}.pkl".format("out2", dataset, era)))



