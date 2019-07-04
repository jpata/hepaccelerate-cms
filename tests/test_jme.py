from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor
import numba
from numba import cuda
import cupy
from math import sqrt

import numpy as np
import time
import os

from coffea.util import USE_CUPY

extract = extractor()

extract.add_weight_sets(['* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi.jec.txt.gz',
                         '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi.jec.txt.gz',
                         '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi.jec.txt.gz',
                         '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi.jec.txt.gz',
                         '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz',
                         '* * coffea/tests/samples/Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi.junc.txt.gz',
                         '* * coffea/tests/samples/Spring16_25nsV10_MC_PtResolution_AK4PFPuppi.jr.txt.gz',
                         '* * coffea/tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt.gz'])

extract.finalize()
evaluator = extract.make_evaluator()

jec_names = ['Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi',
             'Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi',
             'Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi',
             'Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi']

corrector = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names})

#generate some random jet data
N = int(os.environ.get("TESTJME_N"))
test_eta = np.random.randn(N)
test_Rho = np.abs(np.random.randn(N))
test_pt = np.abs(100.0 + 50*np.random.randn(N))
test_A = 4.0 + np.random.randn(N)

jer_names = ['Spring16_25nsV10_MC_PtResolution_AK4PFPuppi']
reso = JetResolution(**{name: evaluator[name] for name in jer_names})
jersf_names = ['Spring16_25nsV10_MC_SF_AK4PFPuppi']
resosf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})
junc_names = ['Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi']
levels = []
for name in dir(evaluator):
    if 'Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi' in name:
        junc_names.append(name)
        levels.append(name.split('_')[-1])
junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

t0 = time.time()
corr = corrector.getCorrection(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt, JetA=test_A)
resos = reso.getResolution(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt)
resosfs = resosf.getScaleFactor(JetEta=test_eta)

if USE_CUPY:
   test_eta = cupy.array(test_eta) 
   test_pt = cupy.array(test_pt) 

juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)

t1 = time.time()
print("time", USE_CUPY, N, t1 - t0)
