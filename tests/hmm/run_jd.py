import json
import sys
import hepaccelerate
import hmumu_utils

if __name__ == "__main__":

    inp = sys.argv[1]
    out = sys.argv[2]

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
    job_descriptions = json.load(open(inp))
    
    ret = hmumu_utils.run_analysis(
        cmdline_args,
        out,
        job_descriptions,
        analysis_parameters,
        analysis_corrections,
        numev_per_chunk=10000)
    print(ret)
