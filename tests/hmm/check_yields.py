import pickle
import sys
from pars import catnames, varnames, analysis_names, shape_systematics, controlplots_shape, datasets, genweight_scalefactor
from pars import cross_sections, categories
from pars import signal_samples, shape_systematics, common_scale_uncertainties, scale_uncertainties

filename = sys.argv[1]
sample_name = sys.argv[2]

res = pickle.load(open(filename, "rb"))
int_lumi = 35000.0
genweight = res["genEventSumw"]
weight_xs = cross_sections[sample_name] * int_lumi / genweight

print("int_lumi", int_lumi)
print("genweight", genweight)

#for k in sorted(res["baseline"].keys()):
#    if k.startswith("hist__"):
#        print(k)

histos = res["baseline"]["hist__dimuon__npvs"]
for k in histos.keys():
    hnom = histos[k]* weight_xs
    print(k, hnom.contents[10], sum(hnom.contents))
