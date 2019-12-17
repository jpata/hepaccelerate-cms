import json
import sys
import hepaccelerate
import hmumu_utils
import boto3
import tempfile
import os

if __name__ == "__main__":

    inp = sys.argv[1]
    iline = int(sys.argv[2])
    out = sys.argv[3]

    USE_CUPY = False
    NUMPY_LIB, ha = hepaccelerate.choose_backend(use_cuda=USE_CUPY)
    
    hmumu_utils.NUMPY_LIB = NUMPY_LIB
    hmumu_utils.ha = ha
    
    #disable everything that requires ROOT which is not easily available on travis tests
    from pars import analysis_parameters
    
    from argparse import Namespace
    cmdline_args = Namespace(use_cuda=USE_CUPY, datapath=".", do_fsr=False, nthreads=1, async_data=False, do_sync=False, out=out)
   
    from analysis_hmumu import AnalysisCorrections
    analysis_corrections = AnalysisCorrections(cmdline_args, True)
    
    from hmumu_utils import run_analysis
    infile_list = open(inp).readlines()[iline].strip()
    job_descriptions = []
    for line in open(infile_list).readlines():
        job_descriptions += [json.load(open(line.strip()))]

    s3_client = boto3.client('s3')

    input_file_idx = 0
    for jd in job_descriptions:
        newfns = []
        for fn in jd["filenames"]:
            newfn = "input_{0}.root".format(input_file_idx)
            print(fn, newfn)
            s3_client.download_file("hepaccelerate-hmm-skim-merged", fn, newfn)
            newfns += [newfn]
            input_file_idx += 1
        jd["filenames"] = newfns
        
        ret = hmumu_utils.run_analysis(
            cmdline_args,
            out,
            [jd],
            analysis_parameters,
            analysis_corrections,
            numev_per_chunk=10000)

        for fn in newfns:
            os.remove(fn)
