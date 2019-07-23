import numpy as np
import os
from hepaccelerate.utils import Histogram, Results

from collections import OrderedDict
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import copy

def histstep(ax, edges, contents, **kwargs):
    ymins = []
    ymaxs = []
    xmins = []
    xmaxs = []
    for istep in range(len(edges)-1):
        xmins += [edges[istep]]
        xmaxs += [edges[istep+1]]
        ymins += [contents[istep]]
        if istep + 1 < len(contents):
            ymaxs += [contents[istep+1]]

    if not "color" in kwargs:
        kwargs["color"] = next(ax._get_lines.prop_cycler)['color']

    ymaxs += [ymaxs[-1]]
    l0 = ax.hlines(ymins, xmins, xmaxs, **kwargs)
    l1 = ax.vlines(xmaxs, ymins, ymaxs, color=l0.get_color(), linestyles=l0.get_linestyle())
    return l0

def midpoints(arr):
    return arr[:-1] + np.diff(arr)/2.0

def plot_hist_step(ax, edges, contents, errors, kwargs_step={}, kwargs_errorbar={}):
    line = histstep(ax, edges, contents, **kwargs_step)
    ax.errorbar(midpoints(edges), contents, errors, lw=0, elinewidth=1, color=line.get_color()[0], **kwargs_errorbar)

def plot_hist_ratio(hists_mc, hist_data):
    plt.figure(figsize=(4,4), dpi=100)

    ax1 = plt.axes([0.0, 0.23, 1.0, 0.8])
       
    hmc_tot = np.zeros_like(hist_data.contents)
    hmc_tot2 = np.zeros_like(hist_data.contents)
    for h in hists_mc:
        plot_hist_step(ax1, h.edges, hmc_tot + h.contents,
            np.sqrt(hmc_tot2 + h.contents_w2),
            kwargs_step={"label": getattr(h, "label", None)}
        )
        hmc_tot += h.contents
        hmc_tot2 += h.contents_w2
#    plot_hist_step(h["edges"], hmc_tot, np.sqrt(hmc_tot2), kwargs_step={"color": "gray", "label": None})
    ax1.errorbar(
        midpoints(hist_data.edges), hist_data.contents,
        np.sqrt(hist_data.contents_w2), marker="o", lw=0,
        elinewidth=1.0, color="black", ms=3, label=getattr(hist_data, "label", None))
    
    ax1.set_yscale("log")
    ax1.set_ylim(1e-2, 100*np.max(hist_data.contents))
    
    #ax1.get_yticklabels()[-1].remove()
    
    ax2 = plt.axes([0.0, 0.0, 1.0, 0.16], sharex=ax1)

    ratio = hist_data.contents / hmc_tot
    ratio_err = np.sqrt(hist_data.contents_w2) /hmc_tot
    ratio[np.isnan(ratio)] = 0

    plt.errorbar(midpoints(hist_data.edges), ratio, ratio_err, marker="o", lw=0, elinewidth=1, ms=3, color="black")
    plt.ylim(0.5, 1.5)
    plt.axhline(1.0, color="black")
    return ax1, ax2

def load_hist(hist_dict):
    return Histogram.from_dict({
        "edges": np.array(hist_dict["edges"]),
        "contents": np.array(hist_dict["contents"]),
        "contents_w2": np.array(hist_dict["contents_w2"]),
    })

def mask_inv_mass(hist):
    bin_idx1 = np.searchsorted(hist["edges"], 120) - 1
    bin_idx2 = np.searchsorted(hist["edges"], 130) + 1
    hist["contents"][bin_idx1:bin_idx2] = 0.0
    hist["contents_w2"][bin_idx1:bin_idx2] = 0.0

def create_variated_histos(
    hdict,
    baseline="nominal",
    variations=["puWeight", "jes", "jer"]):
 
    if not baseline in hdict.keys():
        raise KeyError("baseline histogram missing")
    
    hbase = copy.deepcopy(hdict[baseline])
    ret = Results(OrderedDict())
    ret["nominal"] = Histogram.from_dict(hbase)
    for variation in variations:
        for vdir in ["up", "down"]:
            print("create_variated_histos", variation, vdir)
            sname = "{0}__{1}".format(variation, vdir)
            if sname.endswith("__up"):
                sname2 = sname.replace("__up", "Up")
            elif sname.endswith("__down"):
                sname2 = sname.replace("__down", "Down")

            if sname not in hdict:
                print("systematic", sname, "not found, taking baseline") 
                hret = hbase
            else:
                hret = hdict[sname]
            ret[sname2] = Histogram.from_dict(hret)
    return ret

def create_datacard(dict_procs, parameter_name, all_processes, histname, baseline, variations, weight_xs):
    
    ret = Results(OrderedDict())
    event_counts = {}
 
    for proc in all_processes:
        print("create_datacard", proc)
        rr = dict_procs[proc][parameter_name][histname]
        _variations = variations

        #don't produce variated histograms for data
        if proc == "data":
            _variations = []

        variated_histos = create_variated_histos(rr, baseline, _variations)

        for syst_name, histo in variated_histos.items():
            if proc != "data":
                histo = histo * weight_xs[proc]       

            if syst_name == "nominal":

                event_counts[proc] = np.sum(histo.contents)
                print(proc, syst_name, np.sum(histo.contents))
                if np.sum(histo.contents) < 0.00000001:
                    print("abnormally small mc", np.sum(histo.contents), np.sum(variated_histos["nominal"].contents))

            #create histogram name for combine datacard
            hist_name = "{0}__{2}".format(proc, histname, syst_name)
            if hist_name == "data__nominal":
                hist_name = "data"
            hist_name = hist_name.replace("__nominal", "")
            
            if hist_name in ret:
                import pdb;pdb.set_trace()
            ret[hist_name] = copy.deepcopy(histo)
    
    return ret, event_counts

def save_datacard(dc, outfile):
    fi = uproot.recreate(outfile)
    for histo_name in dc.keys():
        fi[histo_name] = to_th1(dc[histo_name], histo_name)
    fi.close()

def create_datacard_combine(
    dict_procs, parameter_name,
    all_processes,
    signal_processes,
    histname, baseline,
    weight_xs,
    variations,
    common_scale_uncertainties,
    scale_uncertainties,
    txtfile_name
    ):
    
    dc, event_counts = create_datacard(dict_procs, parameter_name, all_processes, histname, baseline, variations, weight_xs)
    rootfile_name = txtfile_name.replace(".txt", ".root")
    
    save_datacard(dc, rootfile_name)
 
    all_processes.pop(all_processes.index("data"))

    shape_uncertainties = {v: 1.0 for v in variations}
    cat = Category(
        name=histname,
        processes=list(all_processes),
        signal_processes=signal_processes,
        common_shape_uncertainties=shape_uncertainties,
        common_scale_uncertainties=common_scale_uncertainties,
        scale_uncertainties=scale_uncertainties,
     )
    
    categories = [cat]

    filenames = {}
    for cat in categories:
        filenames[cat.full_name] = rootfile_name

    PrintDatacard(categories, event_counts, filenames, txtfile_name)

from uproot_methods.classes.TH1 import from_numpy

def to_th1(hdict, name):
    content = np.array(hdict.contents)
    content_w2 = np.array(hdict.contents_w2)
    edges = np.array(hdict.edges)
    
    #remove inf/nan just in case
    content[np.isinf(content)] = 0
    content_w2[np.isinf(content_w2)] = 0

    content[np.isnan(content)] = 0
    content_w2[np.isnan(content_w2)] = 0
    
    #update the error bars
    centers = (edges[:-1] + edges[1:]) / 2.0
    th1 = from_numpy((content, edges))
    th1._fName = name
    th1._fSumw2 = np.array(hdict.contents_w2)
    th1._fTsumw2 = np.array(hdict.contents_w2).sum()
    th1._fTsumwx2 = np.array(hdict.contents_w2 * centers).sum()

    return th1

class Category:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.full_name = self.name
        self.rebin = kwargs.get("rebin", 1)
        self.do_limit = kwargs.get("do_limit", True)


        self.cuts = kwargs.get("cuts", [])

        self.processes = kwargs.get("processes", [])
        self.data_processes = kwargs.get("data_processes", [])
        self.signal_processes = kwargs.get("signal_processes", [])
        
        #[process][systematic] -> scale factor in datacard
        self.shape_uncertainties = {}
        self.scale_uncertainties = {}

        #[syst] -> scale factor, common for all processes
        common_shape_uncertainties = kwargs.get("common_shape_uncertainties", {})
        common_scale_uncertainties = kwargs.get("common_scale_uncertainties", {})
        for proc in self.processes:
            self.shape_uncertainties[proc] = {}
            self.scale_uncertainties[proc] = {}
            for systname, systval in common_shape_uncertainties.items():
                self.shape_uncertainties[proc][systname] = systval
            for systname, systval in common_scale_uncertainties.items():
                self.scale_uncertainties[proc][systname] = systval

        #Load the process-dependent shape uncertainties
        self.proc_shape_uncertainties = kwargs.get("shape_uncertainties", {})
        for proc, v in self.proc_shape_uncertainties.items():
            self.shape_uncertainties[proc].update(v)
        
        self.proc_scale_uncertainties = kwargs.get("scale_uncertainties", {})
        for proc, v in self.proc_scale_uncertainties.items():
            if not (proc in self.scale_uncertainties):
                self.scale_uncertainties[proc] = {}
            self.scale_uncertainties[proc].update(v)

def PrintDatacard(categories, event_counts, filenames, ofname):
    dcof = open(ofname, "w")
    
    number_of_bins = len(categories)
    number_of_backgrounds = 0
    
    backgrounds = []    
    signals = []    
    for cat in categories:
        for proc in cat.processes:
            if (proc in cat.signal_processes):
                signals += [proc]
            else:
                backgrounds += [proc]
    
    backgrounds = set(backgrounds)
    signals = set(signals)
    number_of_backgrounds = len(backgrounds)
    number_of_signals = len(signals)
    analysis_categories = list(set([c.full_name for c in categories]))

    dcof.write("imax {0}\n".format(number_of_bins))
    dcof.write("jmax {0}\n".format(number_of_backgrounds + number_of_signals - 1))
    dcof.write("kmax *\n")
    dcof.write("---------------\n")

    for cat in categories:
#old format
#        dcof.write("shapes * {0} {1} $PROCESS__$CHANNEL $PROCESS__$CHANNEL__$SYSTEMATIC\n".format(
        dcof.write("shapes * {0} {1} $PROCESS $PROCESS__$SYSTEMATIC\n".format(
            cat.full_name,
            os.path.basename(filenames[cat.full_name])
        ))

    dcof.write("---------------\n")

    dcof.write("bin\t" +  "\t".join(analysis_categories) + "\n")
    dcof.write("observation\t" + "\t".join("-1" for _ in analysis_categories) + "\n")
    dcof.write("---------------\n")

    bins        = []
    processes_0 = []
    processes_1 = []
    rates       = []

    for cat in categories:
        for i_sample, sample in enumerate(cat.processes):
            bins.append(cat.full_name)
            processes_0.append(sample)
            if sample in cat.signal_processes:
                i_sample = -i_sample
            processes_1.append(str(i_sample))
            rates.append("{0}".format(event_counts[sample]))
    
    #Write process lines (names and IDs)
    dcof.write("bin\t"+"\t".join(bins)+"\n")
    dcof.write("process\t"+"\t".join(processes_0)+"\n")
    dcof.write("process\t"+"\t".join(processes_1)+"\n")
    dcof.write("rate\t"+"\t".join(rates)+"\n")
    dcof.write("---------------\n")

    # Gather all shape uncerainties
    all_shape_uncerts = []
    all_scale_uncerts = []
    for cat in categories:
        for proc in cat.processes:
            all_shape_uncerts.extend(cat.shape_uncertainties[proc].keys())
            all_scale_uncerts.extend(cat.scale_uncertainties[proc].keys())
    # Uniquify
    all_shape_uncerts = sorted(list(set(all_shape_uncerts)))
    all_scale_uncerts = sorted(list(set(all_scale_uncerts)))

    #print out shape uncertainties
    for syst in all_shape_uncerts:
        dcof.write(syst + "\t shape \t")
        for cat in categories:
            for proc in cat.processes:
                if (proc in cat.shape_uncertainties.keys() and
                    syst in cat.shape_uncertainties[proc].keys()):
                    dcof.write(str(cat.shape_uncertainties[proc][syst]))
                else:
                    dcof.write("-")
                dcof.write("\t")
        dcof.write("\n")


    #print out scale uncertainties
    for syst in all_scale_uncerts:
        dcof.write(syst + "\t lnN \t")
        for cat in categories:
            for proc in cat.processes:
                if (proc in cat.scale_uncertainties.keys() and
                    syst in cat.scale_uncertainties[proc].keys()):
                    dcof.write(str(cat.scale_uncertainties[proc][syst]))
                else:
                    dcof.write("-")
                dcof.write("\t")
        dcof.write("\n")

    #create nuisance groups for easy manipulation and freezing
    nuisance_groups = {}
    for nuisance_group, nuisances in nuisance_groups.items():
        good_nuisances = []
        for nui in nuisances:
            good_nuisances += [nui]
        dcof.write("{0} group = {1}\n".format(nuisance_group, " ".join(good_nuisances)))
    
    #dcof.write("* autoMCStats 20\n")
    #
    #shapename = os.path.basename(datacard.output_datacardname)
    #shapename_base = shapename.split(".")[0]
    dcof.write("\n")
    dcof.write("# Execute with:\n")
    dcof.write("# combine -n {0} -M FitDiagnostics -t -1 {1} \n".format(cat.full_name, os.path.basename(ofname)))

def make_pdf_plot(args):
    res, hd, mc_samples, analysis, var, weight, weight_xs, int_lumi, outdir = args

    hmc = []
    for mc_samp in mc_samples:
        h = load_hist(res[mc_samp][analysis][var][weight])
        h = h * weight_xs[mc_samp]
        h.label = "{0} ({1:.1E})".format(mc_samp, np.sum(h.contents))
        #if var == "hist_inv_mass_d":
        #    h.contents[0] = 0
        #    h.contents_w2[0] = 0
        hmc += [h]

    hd.label = "data ({0:.1E})".format(np.sum(hd.contents))

    if var == "hist_inv_mass_d":
        mask_inv_mass(hd)
    #    hd.contents[0] = 0
    #    hd.contents_w2[0] = 0

    plt.figure(figsize=(4,4))
    a1, a2 = plot_hist_ratio(hmc, hd)
    a2.grid(which="both", linewidth=0.5)
    # Ratio axis ticks
    ts = a2.set_yticks([0.5, 1.0, 1.5], minor=False)
    ts = a2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], minor=True)

    #a2.set_yticks(np.linspace(0.5,1.5, ))
    if var.startswith("hist_numjet"):
        a1.set_xticks(hd["edges"])

    a1.text(0.01,0.99, r"CMS internal, $L = {0:.1f}\ pb^{{-1}}$".format(int_lumi),
        horizontalalignment='left', verticalalignment='top', transform=a1.transAxes
    )
    a1.set_title(" ".join([var, weight]))
    handles, labels = a1.get_legend_handles_labels()
    a1.legend(handles[::-1], labels[::-1], frameon=False, fontsize=4, loc=1)

    plt.savefig(outdir + "/{0}_{1}_{2}.pdf".format(analysis, var, weight), bbox_inches="tight")

if __name__ == "__main__":

    #in picobarns
    #from https://docs.google.com/presentation/d/1OMnGnSs8TIiPPVOEKV8EbWS8YBgEsoMH0r0Js5v5tIQ/edit#slide=id.g3f663e4489_0_20
    cross_sections = {
        "dy": 5765.4,
        "dy_0j": 4620.52,
        "dy_1j": 859.59,
        "dy_2j": 338.26,
        "dy_m105_160_mg": 46.9479,
        "dy_m105_160_vbf_mg": 2.02,
        "dy_m105_160_amc": 41.81,
        "dy_m105_160_vbf_amc": 41.81*0.0425242,
        "ggh": 0.009605,
        "vbf": 0.000823,
        "ttjets_dl": 85.656,
        "ttjets_sl": 687.0,
        "ww_2l2nu": 5.595,
        "wz_3lnu":  4.42965,
        "wz_2l2q": 5.595,
        "wz_1l1nu2q": 11.61,
        "zz": 16.523,
        "st_top": 136.02,
        "st_t_antitop": 80.95,
        "st_tw_top": 35.85,
        "st_tw_antitop": 35.85,
        "ewk_lljj_mll105_160": 0.0508896, 
    }

    import json

    mc_samples_combine_H = [
        "ggh",
        "vbf",
        #"wz_1l1nu2q",
        "wz_3lnu",
       "ww_2l2nu", "wz_2l2q", "zz",
       "ewk_lljj_mll105_160",
       #"st_top",
       #"st_t_antitop",
       "st_tw_top",
       "st_tw_antitop",
       "ttjets_sl", "ttjets_dl",
       "dy_m105_160_amc", "dy_m105_160_vbf_amc",
    ]

    mc_samples_combine_Z = [
        "ggh",
        "vbf",
        #"wz_1l1nu2q",
        "wz_3lnu", 
       "ww_2l2nu", "wz_2l2q", "zz",
       "ewk_lljj_mll105_160",
        #"st_top",
        #"st_t_antitop",
       "st_tw_top",
       "st_tw_antitop",
       "ttjets_sl", "ttjets_dl",
        "dy_0j", "dy_1j", "dy_2j",
    ]
    mc_samples_load = list(set(mc_samples_combine_H + mc_samples_combine_Z))
    signal_samples = ["ggh", "vbf"]
    shape_systematics = ["jes", "jer", "puWeight"]
    common_scale_uncertainties = {
        "lumi": 1.025,
    }
    scale_uncertainties = {
        "ww_2l2nu": {"VVxsec": 1.10},
        "wz_3lnu": {"VVxsec": 1.10},
        "wz_2l2q": {"VVxsec": 1.10},
        "wz_2l2q": {"VVxsec": 1.10},
        "zz": {"VVxsec": 1.10},
        "wjets": {"WJetsxsec": 1.10},
        "dy_m105_160_amc": {"DYxsec": 1.10},
        "dy_m105_160__vbf_amc": {"DYxsec": 1.10},
        "ttjets_sl": {"TTxsec": 1.05},
        "ttjets_dl": {"TTxsec": 1.05},
        "st_t_top": {"STxsec": 1.05},
        "st_t_antitop": {"STxsec": 1.05},
        "st_tw_top": {"STxsec": 1.05},
        "st_tw_antitop": {"STxsec": 1.05},
    }


    #for era in ["2016", "2017", "2018"]:
    for era in ["2018"]:
        res = {}
        genweights = {}
        weight_xs = {}
        datacard_args = []
        
        analysis = "baseline"
        input_folder = "out2"
        dd = "{0}/{1}".format(input_folder, analysis) 
        res["data"] = json.load(open(dd + "/data_{0}.json".format(era)))
        for mc_samp in mc_samples_load:
            res[mc_samp] = json.load(open(dd + "/{0}_{1}.json".format(mc_samp, era)))

        analyses = [k for k in res["data"].keys() if not k in ["cache_metadata", "num_events"]]

        for analysis in analyses:
            print("processing analysis {0}".format(analysis))
            outdir = "{0}/{1}/plots/{2}".format(input_folder, analysis, era)
            outdir_datacards = "{0}/{1}/datacards/{2}".format(input_folder, analysis, era)
            try:
                os.makedirs(outdir)
            except FileExistsError as e:
                pass
            try:
                os.makedirs(outdir_datacards)
            except FileExistsError as e:
                pass

            #in inverse picobarns
            int_lumi = res["data"]["baseline"]["int_lumi"]
            for mc_samp in mc_samples_load:
                genweights[mc_samp] = res[mc_samp]["genEventSumw"]
                weight_xs[mc_samp] = cross_sections[mc_samp] * int_lumi / genweights[mc_samp]
            
            histnames = [h for h in res["data"]["baseline"].keys() if h.startswith("hist__")]
            #for var in [k for k in res["vbf"][analysis].keys() if k.startswith("hist_")]:
            for var in histnames:
                if var in ["hist_puweight", "hist__dijet_inv_mass_gen"]:
                    continue
                if "110_150" in var:
                    mc_samples = mc_samples_combine_H
                else:
                    mc_samples = mc_samples_combine_Z
                create_datacard_combine(
                    res,
                    analysis,
                    ["data"] + mc_samples,
                    signal_samples,
                    var,
                    "nominal",
                    weight_xs,
                    shape_systematics,
                    common_scale_uncertainties,
                    scale_uncertainties,
                    outdir_datacards + "/{0}.txt".format(var)
                )

                weight_scenarios = ["nominal"]
                for weight in weight_scenarios:
                    try:
                        hdata = load_hist(res["data"][analysis][var]["nominal"])
                    except KeyError:
                        print("Histogram {0} not found for data, skipping".format(var))
                        continue
                    datacard_args += [(res, hdata, mc_samples, analysis, var, weight, weight_xs, int_lumi, outdir)]
        ret = list(map(make_pdf_plot, datacard_args))
