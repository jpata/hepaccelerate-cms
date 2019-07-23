from root_numpy.tmva import add_classification_events, evaluate_reader
from root_numpy import ROOT_VERSION
from ROOT import TMVA, TFile, TCut

reader = TMVA.Reader()

def arr():
    return array('f', [0.])

reader2j.AddVariable("hmmpt", arr() )
reader2j.AddVariable("hmmrap", arr() )
reader2j.AddVariable("hmmthetacs", arr() )
reader2j.AddVariable("hmmphics", arr() )
reader2j.AddVariable("j1pt", arr())
reader2j.AddVariable("j1eta", arr())
reader2j.AddVariable("j2pt", arr())
reader2j.AddVariable("detajj", arr())
reader2j.AddVariable("dphijj", arr())
reader2j.AddVariable("mjj", arr())
reader2j.AddVariable("met", arr())
reader2j.AddVariable("zepen", arr())
reader2j.AddVariable("njets", arr())
reader2j.AddVariable("drmj", arr()) 
reader2j.AddVariable("m1ptOverMass", arr())
reader2j.AddVariable("m2ptOverMass", arr())
reader2j.AddVariable("m1eta", arr())
reader2j.AddVariable("m2eta", arr())
reader2j.AddSpectator("hmerr", arr())
reader2j.AddSpectator("weight", arr())
reader2j.AddSpectator("hmass", arr())
reader2j.AddSpectator("nbjets", arr())
reader2j.AddSpectator("bdtucsd_inclusive", arr())
reader2j.AddSpectator("bdtucsd_01jet", arr())
reader2j.AddSpectator("bdtucsd_2jet", arr())

for n in range(n_vars):
    reader.AddVariable('f{0}'.format(n), )
reader.BookMVA('TMVAClassification_BDTG.weights.2jet_bveto', 'TMVAClassification_BDTG.weights.2jet_bveto.xml')

X = np.zeros((1, 24), dtype='f')
Z = evaluate_reader(reader, 'TMVAClassification_BDTG.weights.2jet_bveto', X)
