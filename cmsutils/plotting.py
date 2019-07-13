import matplotlib.pyplot as plt
import numpy as np
import os
from hepaccelerate.utils import Histogram

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
            hmc_tot2 + np.sqrt(h.contents_w2),
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

if __name__ == "__main__":

    #in picobarns
    #from https://docs.google.com/presentation/d/1OMnGnSs8TIiPPVOEKV8EbWS8YBgEsoMH0r0Js5v5tIQ/edit#slide=id.g3f663e4489_0_20
    cross_sections = {
        "dy": 5765.4,
        "ggh": 0.009605,
        "vbf": 0.000823,
        "ttjets_dl": 85.656,
        "ttjets_sl": 687.0,
        "ww_2l2nu": 5.595,
        "wz_3lnu":  4.42965,
        "wz_2l2q": 5.595,
        "wz_1l1nu2q": 11.61,
        "zz": 16.523
    }

    import json


    for era in ["2016", "2017", "2018"]:
        res = {}
        genweights = {}
        weight_xs = {}
        
        dd = "out/baseline" 
        outdir = "out/baseline/plots/{0}".format(era)
        try:
            os.makedirs(outdir)
        except FileExistsError as e:
            pass

        #mc_samples = ["vbf", "ggh", "wz_1l1nu2q", "wz_3lnu", "wz_2l2q", "ww_2l2nu", "zz", "ttjets_dl", "ttjets_sl", "dy"]
        mc_samples = ["vbf", "ggh", "ttjets_dl", "dy"]

        res["data"] = json.load(open(dd + "/data_{0}.json".format(era)))
        for mc_samp in mc_samples:
            res[mc_samp] = json.load(open(dd + "/{0}_{1}.json".format(mc_samp, era)))

        #in inverse picobarns
        int_lumi = res["data"]["baseline"]["int_lumi"]

        for mc_samp in mc_samples:
            genweights[mc_samp] = res[mc_samp]["gen_sumweights"]
            weight_xs[mc_samp] = cross_sections[mc_samp] * int_lumi / genweights[mc_samp]
            print(mc_samp, genweights[mc_samp], cross_sections[mc_samp])
        print(genweights)

        for analysis in ["baseline"]:
            for var in [k for k in res["dy"][analysis].keys() if k.startswith("hist_")]:
            #for var in [
            #    "hist__dimuon__inv_mass",
            #    "hist__dimuon__leading_muon_pt",
            #    #"hist__dimuon__subleading_muon_pt",
            #    #"hist__dimuon__npvs"
            #    ]:
                if var in ["hist_puweight"]:
                    continue

                weight_scenarios = ["nominal", "leptonsf"]
                if var == "hist__dimuon__npvs":
                    weight_scenarios += ["puWeight_off"]
                for weight in weight_scenarios:
                    print(analysis, var, weight)
                    try:
                        hd = load_hist(res["data"][analysis][var]["nominal"])
                    except KeyError:
                        print("Histogram {0} not found for data, skipping".format(var))
                        continue

                    hmc = []
                    print("data", np.sum(hd.contents))
                    for mc_samp in mc_samples:
                        h = load_hist(res[mc_samp][analysis][var][weight])
                        h = h * weight_xs[mc_samp]
                        h.label = "{0} ({1:.1E})".format(mc_samp, np.sum(h.contents))
                        print(var, weight, mc_samp, np.sum(h.contents))
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
