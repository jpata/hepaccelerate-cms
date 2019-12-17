import yaml, sys

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from analysis_hmumu import merge_partial_results

if __name__ == "__main__":
    datasets = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)["datasets"]

    with ProcessPoolExecutor(max_workers=2) as executor:
        for dataset in datasets:
            dataset_name = dataset["name"]
            dataset_era = dataset["era"]
            is_mc = dataset["is_mc"]
            fut = executor.submit(merge_partial_results, dataset_name, dataset_era, "out_merged", sys.argv[2])
