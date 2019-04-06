import matplotlib.pyplot as plt
import numpy as np

def midpoints(arr):
    return arr[:-1] - np.diff(arr)/2.0

def plot_hist_step(edges, contents, errors, kwargs_step={}, kwargs_errorbar={}):
    line = plt.step(edges[:-1], contents, where="pre", **kwargs_step)
    plt.errorbar(midpoints(edges), contents, errors, lw=0, elinewidth=1, color=line[0].get_color(), **kwargs_errorbar)

def plot_hist_ratio(hists_mc, hist_data):
    plt.figure(figsize=(4,4), dpi=100)

    ax1 = plt.axes([0.0, 0.23, 1.0, 0.8])
       
    hmc_tot = np.zeros_like(hist_data["contents"])
    hmc_tot2 = np.zeros_like(hist_data["contents"])
    for h in hists_mc:
        plot_hist_step(h["edges"], hmc_tot + h["xsw"]*h["contents"], hmc_tot2 + h["xsw"]*np.sqrt(h["contents_w2"]), kwargs_step={"label": h.get("label", None)})
        hmc_tot += h["xsw"]*h["contents"]
        hmc_tot2 += h["xsw"]*h["contents_w2"]
#    plot_hist_step(h["edges"], hmc_tot, np.sqrt(hmc_tot2), kwargs_step={"color": "gray", "label": None})
    plt.errorbar(midpoints(hist_data["edges"]), hist_data["contents"], np.sqrt(hist_data["contents_w2"]), marker="o", lw=0, elinewidth=1.0, color="black", ms=3, label=hist_data.get("label", "data"))
    
    plt.yscale("log")
    plt.ylim(1e-2, 10*np.max(hist_data["contents"]))
    
    #ax1.get_yticklabels()[-1].remove()
    
    ax2 = plt.axes([0.0, 0.0, 1.0, 0.16], sharex=ax1)

    ratio = hist_data["contents"] / hmc_tot
    ratio_err = np.sqrt(hist_data["contents_w2"]) /hmc_tot
    ratio[np.isnan(ratio)] = 0

    plt.errorbar(midpoints(hist_data["edges"]), ratio, ratio_err, marker="o", lw=0, elinewidth=1, ms=3, color="black")
    plt.ylim(0.5, 1.5)
    plt.axhline(1.0, color="black")
    return ax1, ax2

def load_hist(hist_dict):
    return {
        "edges": np.array(hist_dict["edges"]),
        "contents": np.array(hist_dict["contents"]),
        "contents_w2": np.array(hist_dict["contents_w2"]),
    }

if __name__ == "__main__":

    #in picobarns
    #from https://docs.google.com/presentation/d/1OMnGnSs8TIiPPVOEKV8EbWS8YBgEsoMH0r0Js5v5tIQ/edit#slide=id.g3f663e4489_0_20
    cross_sections = {
        "dy": 5765.4,
        "ggh": 0.009605,
        "vbf": 0.000823,
        "ttjets_dl": 85.656,
        "ww_2l2nu": 5.595,
        "wz_3lnu":  4.42965,
        "wz_2l2q": 5.595,
        "zz": 16.523
    }

    import json

    res = {}
    genweights = {}
    weight_xs = {}
    mc_samples = ["ggh", "zz", "wz_2l2q", "ww_2l2nu", "wz_3lnu", "ttjets_dl", "vbf", "dy"]

    res["data"] = json.load(open("out/data_2017.json"))
    for mc_samp in mc_samples:
        res[mc_samp] = json.load(open("out/{0}.json".format(mc_samp)))

    #in inverse picobarns
    int_lumi = res["data"]["an1"]["int_lumi"]

    for mc_samp in mc_samples:
        genweights[mc_samp] = res[mc_samp]["gen_sumweights"]
        weight_xs[mc_samp] = cross_sections[mc_samp] * int_lumi / genweights[mc_samp]
        print(mc_samp, genweights[mc_samp], cross_sections[mc_samp])

    for analysis in [k for k in res["dy"].keys() if k.startswith("an")]:

        for var in [k for k in res["dy"][analysis].keys() if k.startswith("hist_")]:
            if var in ["hist_puweight"]:
                continue

            #weights = res["dy"][analysis][var].keys()
            for weight in ["nominal", "puWeight"]:
                hd = load_hist(res["data"][analysis][var]["nominal"])

                hmc = []
                for mc_samp in mc_samples:
                    h = load_hist(res[mc_samp][analysis][var][weight])
                    h["xsw"] = weight_xs[mc_samp]
                    h["label"] = "{0} ({1:.1E})".format(mc_samp, h["xsw"]*np.sum(h["contents"]))
                    print(var, weight, mc_samp, h["xsw"]*np.sum(h["contents"]))
                    hmc += [h]

                hd["label"] = "data ({0:.1E})".format(np.sum(hd["contents"]))

                plt.figure(figsize=(4,4))
                a1, a2 = plot_hist_ratio(hmc, hd)
                a1.text(0.01,0.99, r"CMS internal, $\int L = {0:.1f}\ pb^{{-1}}$".format(int_lumi),
                    horizontalalignment='left', verticalalignment='top', transform=a1.transAxes
                )
                a1.set_title(" ".join([var, weight]))
                handles, labels = a1.get_legend_handles_labels()
                a1.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8)

                plt.savefig("out/{0}_{1}_{2}.pdf".format(analysis, var, weight), bbox_inches="tight")
