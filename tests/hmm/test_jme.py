from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor
import numpy as np

class JetMetCorrections:
    def __init__(self,
        jec_tag,
        jec_tag_data,
        jer_tag,
        do_factorized_jec_unc=False):

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

        if jer_tag:
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
            '{0}_L3Absolute_AK4PFchs'.format(jec_tag),
        ]

        self.jec_mc = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names_mc})

        self.jec_data = {}
        for run, tag in jec_tag_data.items():
            jec_names_data = [
                '{0}_L1FastJet_AK4PFchs'.format(tag),
                '{0}_L2Relative_AK4PFchs'.format(tag),
                '{0}_L2L3Residual_AK4PFchs'.format(tag),
                '{0}_L3Absolute_AK4PFchs'.format(tag),
            ]
            self.jec_data[run] = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names_data})
      
        #self.jer = None 
        #self.jersf = None 
        #if jer_tag: 
        #    jer_names = ['{0}_PtResolution_AK4PFchs'.format(jer_tag)]
        #    self.jer = JetResolution(**{name: evaluator[name] for name in jer_names})
        #    jersf_names = ['{0}_SF_AK4PFchs'.format(jer_tag)]
        #    self.jersf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})

        #junc_names = ['{0}_Uncertainty_AK4PFchs'.format(jec_tag)]
        ##levels = []
        #if do_factorized_jec_unc:
        #    for name in dir(evaluator):

        #        #factorized sources
        #        if '{0}_UncertaintySources_AK4PFchs'.format(jec_tag) in name:
        #            junc_names.append(name)

        #self.jesunc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})
        
jme = JetMetCorrections(
jec_tag="Autumn18_V8_MC",
jec_tag_data={
#    "RunA": "Autumn18_RunA_V8_DATA",
    "RunB": "Autumn18_RunB_V8_DATA",
#    "RunC": "Autumn18_RunC_V8_DATA",
#    "RunD": "Autumn18_RunD_V8_DATA",
},
jer_tag=None)

raw_pt = np.array([100.0, 120.0])
print(jme.jec_data["RunB"].getCorrection(JetPt=raw_pt, Rho=[3.0, 2.0], JetEta=[2.0, 1.0], JetA=[1.0, 2.0]))
print(raw_pt)
print(jme.jec_mc.getCorrection(JetPt=[100.0, 120.0], Rho=[3.0, 2.0], JetEta=[2.0, 1.0], JetA=[1.0, 2.0]))
