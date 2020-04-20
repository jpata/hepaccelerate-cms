import numpy as np
import os
from hepaccelerate.utils import Histogram, Results

from collections import OrderedDict
import uproot


import copy
import multiprocessing

from pars import catnames, varnames, analysis_names, shape_systematics, controlplots_shape, genweight_scalefactor, lhe_pdf_variations
from pars import process_groups, colors, extra_plot_kwargs,proc_grps,combined_signal_samples, remove_proc

from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cmsutils.stats import kolmogorov_smirnov

import argparse
import pickle
import glob

import cloudpickle
import json
import yaml

def get_cross_section(cross_sections, mc_samp, dataset_era):
    d = cross_sections[mc_samp]
    if isinstance(d, dict):
        return d[dataset_era]
    return d

def parse_args():
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis plotting')
    parser.add_argument('--input', action='store', type=str, help='Input directory from the previous step')
    parser.add_argument('--keep-processes', action='append', help='Keep only certain processes, defaults to all', default=None)
    parser.add_argument('--histnames', action='append', help='Process only these histograms, defaults to all', default=None)
    parser.add_argument('--nthreads', action='store', help='Number of parallel threads', default=4, type=int)
    parser.add_argument('--eras', action='append', help='Data eras to process', type=str, required=False)
    args = parser.parse_args()
    return args

def pct_barh(ax, values, colors):
    prev = 0
    norm = sum(values)
    for v, c in zip(values, colors):
        ax.barh(0, width=v/norm, height=1.0, left=prev, color=c)
        prev += v/norm
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0,prev)
    ax.axis('off')

def assign_plot_title_label(histname):
    spl = histname.split("__")
    varname_nice = "UNKNOWN"
    catname_nice = "UNKNOWN"
    if len(spl) == 3:
        catname = spl[1]
        varname = spl[2]
        catname_nice = catnames[catname]
        if varname in varnames:
            varname_nice = varnames[varname]
        else:
            varname_nice = varname
            print("WARNING: please define {0} in pars.py".format(varname))
            
    return varname_nice, catname_nice
             
def plot_hist_ratio(hists_mc, hist_data,
        total_err_stat=None,
        total_err_stat_syst=None,
        figure=None, **kwargs):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xbins = kwargs.get("xbins", None)

    if not figure:
        figure = plt.figure(figsize=(4,4), dpi=100)

    ax1 = plt.axes([0.0, 0.23, 1.0, 0.8])
       
    hmc_tot = np.zeros_like(hist_data.contents)
    hmc_tot2 = np.zeros_like(hist_data.contents)

    edges = hist_data.edges
    if xbins == "uniform":
        edges = np.arange(len(hist_data.edges))

    for h in hists_mc:
        plot_hist_step(ax1, edges, hmc_tot + h.contents,
            np.sqrt(hmc_tot2 + h.contents_w2),
            kwargs_step={"label": getattr(h, "label", None), "color": getattr(h, "color", None)}
        )
        
        b = ax1.bar(midpoints(edges), h.contents, np.diff(edges), hmc_tot, edgecolor=getattr(h, "color", None), facecolor=getattr(h, "color", None))
        hmc_tot += h.contents
        hmc_tot2 += h.contents_w2

#    plot_hist_step(h["edges"], hmc_tot, np.sqrt(hmc_tot2), kwargs_step={"color": "gray", "label": None})
    mask_data_from = kwargs.get("mask_data_from_bin", len(hist_data.contents))
    ax1.errorbar(
        midpoints(edges)[:mask_data_from], hist_data.contents[:mask_data_from],
        np.sqrt(hist_data.contents_w2)[:mask_data_from], marker="o", lw=0,
        elinewidth=1.0, color="black", ms=3, label=getattr(hist_data, "label", None))
    
    if not (total_err_stat_syst is None):
        histstep(ax1, edges, hmc_tot + total_err_stat_syst, color="blue", linewidth=0.5, linestyle="--", label="stat+syst")
        histstep(ax1, edges, hmc_tot - total_err_stat_syst, color="blue", linewidth=0.5, linestyle="--")
    
    if not (total_err_stat is None):
        histstep(ax1, edges, hmc_tot + total_err_stat, color="gray", linewidth=0.5, linestyle="--", label="stat")
        histstep(ax1, edges, hmc_tot - total_err_stat, color="gray", linewidth=0.5, linestyle="--")
    
    if kwargs.get("do_log", False):
        ax1.set_yscale("log")
        ax1.set_ylim(1, 100*np.max(hist_data.contents))
    else:
        ax1.set_ylim(0, 2*np.max(hist_data.contents))

    #ax1.get_yticklabels()[-1].remove()
    
    ax2 = plt.axes([0.0, 0.0, 1.0, 0.16], sharex=ax1)

    ratio = hist_data.contents / hmc_tot
    ratio_err = np.sqrt(hist_data.contents_w2) /hmc_tot
    ratio[np.isnan(ratio)] = 0

    plt.errorbar(midpoints(edges)[:mask_data_from], ratio[:mask_data_from], ratio_err[:mask_data_from], marker="o", lw=0, elinewidth=1, ms=3, color="black")

    if not (total_err_stat_syst is None):
        ratio_up = (hmc_tot + total_err_stat_syst) / hmc_tot
        ratio_down = (hmc_tot - total_err_stat_syst) / hmc_tot
        ratio_down[np.isnan(ratio_down)] = 1
        ratio_down[np.isnan(ratio_up)] = 1
        histstep(ax2, edges, ratio_up, color="blue", linewidth=0.5, linestyle="--")
        histstep(ax2, edges, ratio_down, color="blue", linewidth=0.5, linestyle="--")

    if not (total_err_stat is None):
        ratio_up = (hmc_tot + total_err_stat) / hmc_tot
        ratio_down = (hmc_tot - total_err_stat) / hmc_tot
        ratio_down[np.isnan(ratio_down)] = 1
        ratio_down[np.isnan(ratio_up)] = 1
        histstep(ax2, edges, ratio_up, color="gray", linewidth=0.5, linestyle="--")
        histstep(ax2, edges, ratio_down, color="gray", linewidth=0.5, linestyle="--")

                
    plt.ylim(0.5, 1.5)
    plt.axhline(1.0, color="black")

    if xbins == "uniform":
        print(hist_data.edges)
        ax1.set_xticks(edges)
        ax1.set_xticklabels(["{0:.2f}".format(x) for x in hist_data.edges])

    ax1.set_xlim(min(edges), max(edges))
    xlim = kwargs.get("xlim", None)
    if not xlim is None:
        ax1.set_xlim(*xlim)

    ylim = kwargs.get("ylim", None)
    if not ylim is None:
        ax1.set_ylim(*ylim)

    return ax1, ax2

def plot_variations(args):
    res, hd, mc_samp, analysis, var, weight, weight_xs, int_lumi, outdir, datataking_year, unc = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _outdir = outdir + "/shape_systematics/{0}/".format(var)
    try:
        os.makedirs(_outdir)
    except Exception as e:
        pass
    fig = plt.figure(figsize=(5,5), dpi=100)
    ax = plt.axes()
    hnom = res[mc_samp]["nominal"]* weight_xs[mc_samp]
    plot_hist_step(ax, hnom.edges, hnom.contents,
        np.sqrt(hnom.contents_w2),
                   kwargs_step={"label": "nominal "+"({0:.3E})".format(np.sum(hnom.contents))},
    )
    for sdir in ["__up", "__down"]:
        if (unc + sdir) in res[mc_samp]:
            hvar = res[mc_samp][unc + sdir]* weight_xs[mc_samp]
            plot_hist_step(ax, hvar.edges, hvar.contents,
                np.sqrt(hvar.contents_w2),
                           kwargs_step={"label": sdir.replace("__", "") + " ({0:.3E})".format(np.sum(hvar.contents))},
            
            )

    if('EWZ105160PS' in unc) and ('ewk_lljj_mll105_160_ptJ_herwig' in mc_samp):
        h_ps_pythia = res["ewk_lljj_mll105_160_pythia"]["nominal"]* weight_xs["ewk_lljj_mll105_160_pythia"]
        h_ps_herwig = res["ewk_lljj_mll105_160_herwig"]["nominal"]* weight_xs["ewk_lljj_mll105_160_herwig"]
        h_nom_up = copy.deepcopy(hnom)
        h_nom_down = copy.deepcopy(hnom)
        h_nom_up.contents = hnom.contents - 0.2*(h_ps_pythia.contents - h_ps_herwig.contents)
        h_nom_down.contents = hnom.contents + 0.2*(h_ps_pythia.contents - h_ps_herwig.contents)
        plot_hist_step(ax, h_nom_up.edges, h_nom_up.contents,
                np.sqrt(h_nom_up.contents_w2),
                       kwargs_step={"label": "up "+"({0:.3E})".format(np.sum(h_nom_up.contents))},
            )
        plot_hist_step(ax, h_nom_down.edges, h_nom_down.contents,
                np.sqrt(h_nom_down.contents_w2),
                       kwargs_step={"label": "down "+"({0:.3E})".format(np.sum(h_nom_down.contents))},
            )

    if((('DYLHEScaleWeight' in unc) and ('dy' in mc_samp)) or (('EWZLHEScaleWeight' in unc) and ('ewk' in mc_samp) )):
        
        h_lhe =[]
        h_nom_up = copy.deepcopy(hnom)
        h_nom_down = copy.deepcopy(hnom)
        for i in range(9):
            sname = 'LHEScaleWeight__{0}'.format(i)
            h_lhe.append(res[mc_samp][sname]* weight_xs[mc_samp])
        for k in range(len(h_lhe[0].contents)):
            for i in range(9):
                if(h_lhe[i].contents[k]>h_nom_up.contents[k]):
                    h_nom_up.contents[k]=h_lhe[i].contents[k]
                if(h_lhe[i].contents[k]<h_nom_down.contents[k]):
                    h_nom_down.contents[k]=h_lhe[i].contents[k]
        #remove the normalization aspect from QCD scale
        sum_nom_up=np.sum(h_nom_up.contents)
        sum_nom_down=np.sum(h_nom_down.contents)
        for k in range(len(h_nom_up.contents)):
            h_nom_up.contents[k]=h_nom_up.contents[k]*np.sum(hnom.contents)/sum_nom_up
            h_nom_down.contents[k]=h_nom_down.contents[k]*np.sum(hnom.contents)/sum_nom_down
        
        plot_hist_step(ax, h_nom_up.edges, h_nom_up.contents,
                np.sqrt(h_nom_up.contents_w2),
                       kwargs_step={"label": "up "+"({0:.3E})".format(np.sum(h_nom_up.contents))},
            )
        plot_hist_step(ax, h_nom_down.edges, h_nom_down.contents,
                np.sqrt(h_nom_down.contents_w2),
                       kwargs_step={"label": "down "+"({0:.3E})".format(np.sum(h_nom_down.contents))},
            )

    if(('LHEPdfWeight' in unc) and ("dy" in mc_samp or "ewk" in mc_samp or "ggh" in mc_samp or "vbf" in mc_samp or "wph" in mc_samp or "wmh" in mc_samp or "tth" in mc_samp)):
        h_pdf =[]
        h_pdf_up = copy.deepcopy(hnom)
        h_pdf_down = copy.deepcopy(hnom)
        n_pdf =0
        for i in range(lhe_pdf_variations[str(datataking_year)]):
            sname = 'LHEPdfWeight__{0}'.format(i)
            if sname in res[mc_samp].keys():
                h_pdf.append(res[mc_samp][sname]* weight_xs[mc_samp])
        for k in range(len(h_pdf[0].contents)):
            rms = 0.0
            for i in range(len(h_pdf)):
                    rms = rms + (h_pdf[i].contents[k]-hnom.contents[k])**2
            rms = np.sqrt(rms/(len(h_pdf)-1))
            h_pdf_up.contents[k] = hnom.contents[k] + rms
            h_pdf_down.contents[k] = hnom.contents[k] - rms
        #remove the normalization aspect from pdf
        sum_pdf_up=np.sum(h_pdf_up.contents)
        sum_pdf_down=np.sum(h_pdf_down.contents)
        for k in range(len(h_pdf_up.contents)):
            if(sum_pdf_up!=0.0) : h_pdf_up.contents[k]=h_pdf_up.contents[k]*np.sum(hnom.contents)/sum_pdf_up
            if(sum_pdf_down!=0.0) : h_pdf_down.contents[k]=h_pdf_down.contents[k]*np.sum(hnom.contents)/sum_pdf_down

        plot_hist_step(ax, h_pdf_up.edges, h_pdf_up.contents,
                np.sqrt(h_pdf_up.contents_w2),
                       kwargs_step={"label": "up "+"({0:.3E})".format(np.sum(h_pdf_up.contents))},
            )
        plot_hist_step(ax, h_pdf_down.edges, h_pdf_down.contents,
                np.sqrt(h_pdf_down.contents_w2),
                       kwargs_step={"label": "down "+"({0:.3E})".format(np.sum(h_pdf_down.contents))},
            )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=4, loc=1, ncol=2)
    #ax.set_yscale("log")
    plt.savefig(_outdir + "/{0}_{1}.png".format(mc_samp, unc), bbox_inches="tight")
    plt.close(fig)
    del fig

def make_pdf_plot(args):
    res, hd, mc_samples, analysis, var, baseline_weight, weight_xs, int_lumi, outdir, datataking_year, groups, extra_kwargs = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if np.sum(hd.contents) == 0:
        print("ERROR: Histogram {0} was empty, skipping".format(var))
        return

    hist_template = copy.deepcopy(hd)
    hist_template.contents[:] = 0
    hist_template.contents_w2[:] = 0

    hmc = {}

    for mc_samp in mc_samples:
        h = res[mc_samp][baseline_weight]
        h = h * weight_xs[mc_samp]
        h.label = "{0} ({1:.1E})".format(mc_samp, np.sum(h.contents))
        hmc[mc_samp] = h
    
    hmc_g = group_samples(hmc, groups)

    for k, v in hmc_g.items():
        if k in colors.keys():
            v.color = colors[k][0]/255.0, colors[k][1]/255.0, colors[k][2]/255.0
    hmc = [hmc_g[k[0]] for k in groups]
        
    htot_nominal = sum(hmc, hist_template)
    htot_variated = {}
    hdelta_quadrature = np.zeros_like(hist_template.contents)
    
    for sdir in ["__up", "__down"]:
        for unc in shape_systematics:
            if (unc + sdir) in res[mc_samp]:
                htot_variated[unc + sdir] = sum([
                    res[mc_samp][unc + sdir]* weight_xs[mc_samp] for mc_samp in mc_samples
                ], hist_template)
                hdelta_quadrature += (htot_nominal.contents - htot_variated[unc+sdir].contents)**2
            
    hdelta_quadrature_stat = np.sqrt(htot_nominal.contents_w2)
    hdelta_quadrature_stat_syst = np.sqrt(hdelta_quadrature_stat**2 + hdelta_quadrature)
    hd.label = "data ({0:.1E})".format(np.sum(hd.contents))

    figure = plt.figure(figsize=(5,5), dpi=100)
    a1, a2 = plot_hist_ratio(
        hmc, hd,
        total_err_stat=hdelta_quadrature_stat,
        total_err_stat_syst=hdelta_quadrature_stat_syst,
        figure=figure, **extra_kwargs)
    
    colorlist = [h.color for h in hmc]
    a1inset = inset_axes(a1, width=1.0, height=0.1, loc=2)
    pct_barh(a1inset, [np.sum(h.contents) for h in hmc], colorlist)
    #a2.grid(which="both", linewidth=0.5)
    
    # Ratio axis ticks
    #ts = a2.set_yticks([0.5, 1.0, 1.5], minor=False)
    #ts = a2.set_yticks(np.arange(0,2,0.2), minor=True)
    #ts = a2.set_xticklabels([])

    a1.text(0.03,0.95, "CMS internal\n" +
        r"$L = {0:.1f}\ fb^{{-1}}$".format(int_lumi/1000.0) + 
        "\nd/mc={0:.2f}".format(np.sum(hd.contents)/np.sum(htot_nominal.contents)) + 
        "\nwd={0:.2E}".format(wasserstein_distance(htot_nominal.contents/np.sum(htot_nominal.contents), hd.contents/np.sum(hd.contents))) +
        "\nks={0:.2E}".format(kolmogorov_smirnov(
            htot_nominal.contents, hd.contents,
            variances1=htot_nominal.contents_w2,
            variances2=hd.contents_w2
        )),
        horizontalalignment='left',
        verticalalignment='top',
        transform=a1.transAxes,
        fontsize=10
    )
    handles, labels = a1.get_legend_handles_labels()
    a1.legend(handles[::-1], labels[::-1], frameon=False, fontsize=10, loc=1, ncol=2)
    
    varname, catname = assign_plot_title_label(var)
    
    a1.set_title(catname + " ({0})".format(analysis_names[analysis][datataking_year]))
    a2.set_xlabel(varname)
    
    binwidth = np.diff(hd.edges)[0]
    a1.set_ylabel("events / bin [{0:.1f}]".format(binwidth))
    try:
        os.makedirs(outdir + "/png")
    except Exception as e:
        pass
    try:
        os.makedirs(outdir + "/pdf")
    except Exception as e:
        pass

    plt.savefig(outdir + "/pdf/{0}_{1}_{2}.pdf".format(analysis, var, baseline_weight), bbox_inches="tight")
    plt.savefig(outdir + "/png/{0}_{1}_{2}.png".format(analysis, var, baseline_weight), bbox_inches="tight", dpi=100)
    plt.close(figure)
    del figure
 
    return

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

def create_variated_histos(weight_xs, proc,
    hdict, hdict_ps_pythia, ps_pythia, hdict_ps_herwig, ps_herwig, era,
    baseline="nominal",
        variations=shape_systematics):
    if not baseline in hdict.keys():
        raise KeyError("baseline histogram missing")
   
    #hbase = copy.deepcopy(hdict[baseline])
    hbase = hdict[baseline]
    ret = Results(OrderedDict())
    ret["nominal"] = hbase
    for variation in variations:
        for vdir in ["up", "down"]:
            #print("create_variated_histos", variation, vdir)
            sname = "{0}__{1}".format(variation, vdir)
            if sname.endswith("__up"):
                sname2 = sname.replace("__up", "Up")
            elif sname.endswith("__down"):
                sname2 = sname.replace("__down", "Down")

            if (sname not in hdict.keys()):
                #print("systematic", sname, "not found, taking baseline") 
                hret = hbase
            else:
                hret = hdict[sname]
            ret[sname2] = hret
 
    if('EWZ105160PS' in variations) and ('ewk_lljj_mll105_160_ptJ_herwig' in proc):
        h_ps_pythia = copy.deepcopy(hdict_ps_pythia[baseline])
        h_ps_herwig = copy.deepcopy(hdict_ps_herwig[baseline])
        hnom = copy.deepcopy(hbase)
        h_nom_up = copy.deepcopy(hbase)
        h_nom_down = copy.deepcopy(hbase)
        hnom = hnom * weight_xs[proc]
        h_ps_pythia = h_ps_pythia * weight_xs[ps_pythia]
        h_ps_herwig = h_ps_herwig * weight_xs[ps_herwig]
        h_nom_up.contents = hnom.contents - 0.2*(h_ps_pythia.contents - h_ps_herwig.contents)
        h_nom_down.contents = hnom.contents + 0.2*(h_ps_pythia.contents - h_ps_herwig.contents)
        hnom = hnom * (1./weight_xs[proc])
        h_nom_up = h_nom_up * (1./weight_xs[proc])
        h_nom_down = h_nom_down * (1./ weight_xs[proc])
        h_ps_pythia = h_ps_pythia * (1./weight_xs[ps_pythia])
        h_ps_herwig = h_ps_herwig * (1./weight_xs[ps_herwig])
        ret['EWZ105160PSUp']=h_nom_up
        ret['EWZ105160PSDown']=h_nom_down

    if(('DYLHEScaleWeight' in variations) or ('EWZLHEScaleWeight' in variations)):
        h_lhe =[]
        h_nom_up = copy.deepcopy(hbase)
        h_nom_down = copy.deepcopy(hbase)
        for i in range(9):
            sname = 'LHEScaleWeight__{0}'.format(i)
            h_lhe.append(hdict[sname])
        for k in range(len(h_lhe[0].contents)):
            for i in range(9):
                if(h_lhe[i].contents[k]>h_nom_up.contents[k]):
                    h_nom_up.contents[k]=h_lhe[i].contents[k]
                if(h_lhe[i].contents[k]<h_nom_down.contents[k]):
                    h_nom_down.contents[k]=h_lhe[i].contents[k]
        #remove the normalization aspect from QCD scale
        sum_nom_up=np.sum(h_nom_up.contents)
        sum_nom_down=np.sum(h_nom_down.contents)
        for k in range(len(h_nom_up.contents)):
            h_nom_up.contents[k]=h_nom_up.contents[k]*np.sum(hbase.contents)/sum_nom_up
            h_nom_down.contents[k]=h_nom_down.contents[k]*np.sum(hbase.contents)/sum_nom_down

        if('dy' in proc and '160' in proc):
            ret['DYLHEScaleWeightUp']=h_nom_up
            ret['DYLHEScaleWeightDown']=h_nom_down
        elif('ewk' in proc and '160' in proc):
            ret['EWZLHEScaleWeightUp']=h_nom_up
            ret['EWZLHEScaleWeightDown']=h_nom_down
        elif('dy' in proc):
            ret['DYLHEScaleWeightZUp']=h_nom_up
            ret['DYLHEScaleWeightZDown']=h_nom_down
        elif('ewk' in proc):
            ret['EWZLHEScaleWeightZUp']=h_nom_up
            ret['EWZLHEScaleWeightZDown']=h_nom_down

    if('LHEPdfWeight' in variations):
        h_pdf =[]
        h_pdf_up = copy.deepcopy(hbase)
        h_pdf_down = copy.deepcopy(hbase)
        h_pdf_nom = copy.deepcopy(hbase)
        if "dy" in proc or "ewk" in proc or "ggh" in proc or "vbf" in proc or "zh_125" in proc or "wmh_125" in proc or "wph_125" in proc or "tth" in proc:
            h_nom = np.zeros_like(hbase.contents)
            n_pdf =0
            for i in range(lhe_pdf_variations[str(era)]):
                sname = 'LHEPdfWeight__{0}'.format(i)
                if sname in hdict.keys():
                    h_pdf.append(hdict[sname])
            for i in range(len(h_pdf)):
                h_nom = h_nom + h_pdf[i].contents 
            h_nom = h_nom/len(h_pdf)
            for k in range(len(h_pdf[0].contents)):
                h_pdf_nom.contents[k] = h_nom[k]
                rms = 0.0
                for i in range(len(h_pdf)):
                    rms = rms + (h_pdf[i].contents[k]-hbase.contents[k])**2
                rms = np.sqrt(rms/(len(h_pdf)-1))
                h_pdf_up.contents[k] = hbase.contents[k] + rms
                h_pdf_down.contents[k] = hbase.contents[k] - rms
            #remove the normalization aspect from pdf
            sum_pdf_up=np.sum(h_pdf_up.contents)
            sum_pdf_down=np.sum(h_pdf_down.contents)
            for k in range(len(h_pdf_up.contents)):
                if(sum_pdf_up!=0.0): h_pdf_up.contents[k]=h_pdf_up.contents[k]*np.sum(hbase.contents)/sum_pdf_up
                if(sum_pdf_down!=0.0): h_pdf_down.contents[k]=h_pdf_down.contents[k]*np.sum(hbase.contents)/sum_pdf_down
        ret['LHEPdfNom']=h_pdf_nom
        ret['LHEPdfWeightUp']=h_pdf_up
        ret['LHEPdfWeightDown']=h_pdf_down
        
    return ret

def create_datacard(dict_procs, parameter_name, all_processes, histname, baseline, variations, weight_xs, era):
    ret = Results(OrderedDict())
    event_counts = {}
    hists_mc = []
    for pid,pid_procs in proc_grps:
        event_counts[pid]=0
    for proc in all_processes:
        rr = dict_procs[proc]
        _variations = variations
        
        ps_pythia = 'ewk_lljj_mll105_160_pythia'
        ps_herwig = 'ewk_lljj_mll105_160_herwig'
        if ((ps_pythia in dict_procs.keys()) and (ps_herwig in dict_procs.keys())):
            rr_ps_pythia = dict_procs[ps_pythia]
            rr_ps_herwig = dict_procs[ps_herwig]
        else:
            rr_ps_pythia = rr
            rr_ps_herwig = rr
            
        #don't produce variated histograms for data
        if proc == "data":
            _variations = []

        variated_histos = create_variated_histos(weight_xs, proc, rr, rr_ps_pythia, ps_pythia, rr_ps_herwig, ps_herwig, era, baseline, _variations)

        for syst_name, histo in variated_histos.items():
            if proc != "data":
                histo = histo * weight_xs[proc]

            if syst_name == "nominal":
                found_proc=0
                for pid,pid_procs in proc_grps:
                    
                    if proc in pid_procs:
                        event_counts[pid]+= np.sum(histo.contents)
                        found_proc=1
                        #print(pid,proc, syst_name, np.sum(histo.contents))
                    
                if proc != "data":
                    hists_mc += [histo]
                if found_proc==0:
                    event_counts[proc] = np.sum(histo.contents)
            #create histogram name for combine datacard
            
            hist_name = "{0}__{2}".format(proc, histname, syst_name)
            if hist_name == "data__nominal":
                hist_name = "data_obs"
            hist_name = hist_name.replace("__nominal", "")
            
            ret[hist_name] = copy.deepcopy(histo)
    assert(len(hists_mc) > 0)
    hist_mc_tot = copy.deepcopy(hists_mc[0])
    for h in hists_mc[:1]:
        hist_mc_tot += h
    ret["data_fake"] = hist_mc_tot
    ret_g = group_samples_datacard(ret, proc_grps)
    return ret_g, event_counts

def save_datacard(dc, outfile):
    fi = uproot.recreate(outfile)
    for histo_name in dc.keys():
        fi[histo_name] = to_th1(dc[histo_name], histo_name)
    fi.close()

def create_datacard_combine_wrap(args):
    return create_datacard_combine(*args)

def create_datacard_combine(
    dict_procs, parameter_name,
    all_processes,
    signal_processes,
    combined_all_processes,
    combined_signal_processes,
    histname, baseline,
    weight_xs,
    variations,
    common_scale_uncertainties,
    scale_uncertainties,
    txtfile_name,
    era
    ):

    dc, event_counts = create_datacard(
        dict_procs, parameter_name, all_processes,
        histname, baseline, variations, weight_xs, era)
    rootfile_name = txtfile_name.replace(".txt", ".root")
    
    save_datacard(dc, rootfile_name)
 
    all_processes.pop(all_processes.index("data"))
    combined_all_processes.pop(combined_all_processes.index("data"))
    shape_uncertainties = {v:1.0 for v in variations}
    cat = Category(
        name=histname,
        processes=list(combined_all_processes),
        signal_processes=combined_signal_processes,
        common_shape_uncertainties=shape_uncertainties,
        common_scale_uncertainties=common_scale_uncertainties,
        scale_uncertainties=scale_uncertainties,
     )
    
    categories = [cat]

    filenames = {}
    for cat in categories:
        filenames[cat.full_name] = rootfile_name
 
    PrintDatacard(categories, dict_procs, era, event_counts, filenames, txtfile_name)

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
def group_samples_datacard(histos, groups):
    ret = {}
    grp_list=[]
    for n,h in histos.items():
        found_h=0
        for groupname, groupcontents in groups:
            if n in groupcontents:
                if groupname not in grp_list:
                    grp_list.append(groupname)
                    ret[groupname] = []
                ret[groupname] += [h]
                found_h=1
            else:
                for gc in groupcontents:
                    if( (gc == n.split('__')[0]) and (n.split('__')[-1]!=gc)):
                        ext=n.split('__')[1]
                        if (groupname+"__"+ext) not in grp_list:
                            grp_list.append(groupname+"__"+ext)
                            ret[groupname+"__"+ext]=[]
                        ret[groupname+"__"+ext] += [h]
                        found_h=1
        if found_h==0:
            if n not in grp_list:
                grp_list.append(n)
                ret[n]=[]
            ret[n]+= [h]
    for gn in grp_list: 
        ret[gn] = sum(ret[gn][1:], ret[gn][0])
    return ret

def group_samples(histos, groups):
    ret = {}
    for groupname, groupcontents in groups:
        ret[groupname] = []
        for gc in groupcontents:
            if gc in histos:
                ret[groupname] += [histos[gc]]
        assert(len(ret[groupname]) > 0)
        ret[groupname] = sum(ret[groupname][1:], ret[groupname][0])
        ret[groupname].label = "{0} ({1:.2E})".format(groupname, np.sum(ret[groupname].contents))
    return ret

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

def calculate_LHEPdf_norm(histos, era, proc):
    #Based on https://arxiv.org/pdf/1510.03865.pdf
    h_pdf = []
    n_pdf =0 
    for i in range(lhe_pdf_variations[str(era)]):
        sname = 'LHEPdfWeight__{0}'.format(i)
        if sname in histos.keys():
            h_pdf.append(histos[sname])
    h_nom = np.zeros_like(histos["nominal"].contents)
    for i in range(len(h_pdf)):
        h_nom = h_nom + h_pdf[i].contents
    h_nom = h_nom/len(h_pdf)
    var = 0.0
    for i in range(len(h_pdf)):
        var = var + (np.sum(h_pdf[i].contents) - np.sum(h_nom))**2

    if((era == "2016") and ("ewk_lljj_mll105_160_ptJ_herwig" not in proc) and ("tth_125" not in proc)):
        var = np.sqrt(var/(len(h_pdf)-1)) # MC replicas for 2016
    else: var = np.sqrt(var) #Hessian weights for 2017 and 2018
    if(np.sum(h_nom)!=0.0): var = var/np.sum(h_nom)
    return var

def calculate_LHEscale_norm(histos, era):
        h_lhe =[]
        h_nom_up = copy.deepcopy(histos["nominal"])
        h_nom_down = copy.deepcopy(histos["nominal"])
        for i in range(9):
            sname = 'LHEScaleWeight__{0}'.format(i)
            h_lhe.append(histos[sname])
        for k in range(len(h_lhe[0].contents)):
            for i in range(9):
                if(h_lhe[i].contents[k]>h_nom_up.contents[k]):
                    h_nom_up.contents[k]=h_lhe[i].contents[k]
                if(h_lhe[i].contents[k]<h_nom_down.contents[k]):
                    h_nom_down.contents[k]=h_lhe[i].contents[k]
        var = (np.sum(h_nom_up.contents) - np.sum(histos["nominal"].contents))**2 + (np.sum(h_nom_down.contents) - np.sum(histos["nominal"].contents))**2 
        var = np.sqrt(var/2.0)
        if(np.sum(histos["nominal"].contents)!=0.0): var = var/np.sum(histos["nominal"].contents)
        return var

def PrintDatacard(categories, dict_procs, era, event_counts, filenames, ofname):
    dcof = open(ofname, "w")
    
    number_of_bins = len(categories)
    number_of_backgrounds = 0
   
    backgrounds = []    
    signals = []    
    for cat in categories:
        for proc in cat.processes:
            if (proc in cat.signal_processes):
                signals += [proc]
            elif (proc in remove_proc):
                continue
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
            if (sample in remove_proc): 
                continue
            else:
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
            if proc not in remove_proc:
                all_shape_uncerts.extend(cat.shape_uncertainties[proc].keys())
                all_scale_uncerts.extend(cat.scale_uncertainties[proc].keys())
    # Uniquify
    all_shape_uncerts = sorted(list(set(all_shape_uncerts)))
    all_scale_uncerts = sorted(list(set(all_scale_uncerts)))

    #print out shape uncertainties
    for syst in all_shape_uncerts:
        if('LHEScale' in syst): continue
        dcof.write(syst + "\t shape \t")
        for cat in categories:
            for proc in cat.processes:
                if proc in remove_proc:
                    continue
                elif (proc in cat.shape_uncertainties.keys() and
                    syst in cat.shape_uncertainties[proc].keys()):
                    dcof.write(str(cat.shape_uncertainties[proc][syst]))
                else:
                    dcof.write("-")
                dcof.write("\t")
        dcof.write("\n")
    if 'z_peak' in  cat.full_name:
        dcof.write("EWZLHEScaleWeightZ" + "\t shape \t")
    else:
        dcof.write("EWZLHEScaleWeight" + "\t shape \t")
    for cat in categories:
        for proc in cat.processes:
            if proc in remove_proc:
                continue
            elif ('ewk' in proc):
                dcof.write(str(1.0))
            else:
                dcof.write("-")
            dcof.write("\t")
    dcof.write("\n")
    
    if 'z_peak'in  cat.full_name:
        dcof.write("DYLHEScaleWeightZ" + "\t shape \t")
    else:
        dcof.write("DYLHEScaleWeight" + "\t shape \t")
    for cat in categories:
        for proc in cat.processes:
            if proc in remove_proc:
                continue
            elif ('dy' in proc):
                dcof.write(str(1.0))
            else:
                dcof.write("-")
            dcof.write("\t")
    dcof.write("\n")
    #print out scale uncertainties
    for syst in all_scale_uncerts:
        dcof.write(syst + "\t lnN \t")
        for cat in categories:
            for proc in cat.processes:
                if proc in remove_proc:
                    continue
                elif (proc in cat.scale_uncertainties.keys() and
                    syst in cat.scale_uncertainties[proc].keys()):
                    dcof.write(str(cat.scale_uncertainties[proc][syst]))
                else:
                    dcof.write("-")
                dcof.write("\t")
        dcof.write("\n")
    # print out LHE PDf norm uncert
    dcof.write("LHEPdfWeight_norm" + "\t lnN \t")
    for cat in categories:
        for proc in cat.processes:
            if proc in remove_proc:
                continue
            elif "dy" in proc or "ewk" in proc or "ggh" in proc or "vbf" in proc or "vh" in proc or "tth" in proc:
                if("vh" not in proc):
                    Pdf_norm = calculate_LHEPdf_norm(dict_procs[proc], era, proc)
                    dcof.write("{0:.3f}".format(1.0 + Pdf_norm))
                else:
                    Pdf_norm_vh =[]
                    Pdf_norm_vh.append(calculate_LHEPdf_norm(dict_procs["wph_125"], era, proc))
                    Pdf_norm_vh.append(calculate_LHEPdf_norm(dict_procs["wmh_125"], era, proc))
                    Pdf_norm_vh.append(calculate_LHEPdf_norm(dict_procs["zh_125"], era, proc))
                    sorted(Pdf_norm_vh, reverse = True)
                    dcof.write("{0:.3f}".format(1.0 + Pdf_norm_vh[0]))
                    
            else:
                dcof.write("-")
            dcof.write("\t")
        dcof.write("\n")

    # print out LHE Scale norm uncert
    if 'z_peak'in  cat.full_name:
        dcof.write("DYLHEScaleWeightZ_norm" + "\t lnN \t")
    else:
        dcof.write("DYLHEScaleWeight_norm" + "\t lnN \t")
    
    for cat in categories:
        for proc in cat.processes:
            if "dy" in proc:
                QCD_norm = calculate_LHEscale_norm(dict_procs[proc], era)
                dcof.write("{0:.2f}".format(1.0 + QCD_norm))
            else:
                dcof.write("-")
            dcof.write("\t")
        dcof.write("\n")
    
    if 'z_peak'in  cat.full_name:
        dcof.write("EWKLHEScaleWeightZ_norm" + "\t lnN \t")
    else:
        dcof.write("EWKLHEScaleWeight_norm" + "\t lnN \t")

    for cat in categories:
        for proc in cat.processes:
            if "ewk" in proc:
                QCD_norm = calculate_LHEscale_norm(dict_procs[proc], era)
                dcof.write("{0:.2f}".format(1.0 + QCD_norm))
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
    if "z_peak" in cat.full_name:
        dcof.write("RZ rateParam {0} dy_0j 1 \n".format(cat.full_name))  
        dcof.write("RZ rateParam {0} dy_1j 1 \n".format(cat.full_name))  
        dcof.write("RZ rateParam {0} dy_2j 1 \n".format(cat.full_name)) 
        #dcof.write("REWZ rateParam {0} ewk_lljj_mll50_mjj120 1 \n".format(cat.full_name))
    elif ("h_peak" in cat.full_name) or ("h_sideband" in cat.full_name):
        dcof.write("R_01j rateParam {0} dy_m105_160_amc_01j 1 \n".format(cat.full_name))           
        dcof.write("R_01j rateParam {0} dy_m105_160_vbf_amc_01j 1 \n".format(cat.full_name))
        dcof.write("R_2j rateParam {0} dy_m105_160_amc_2j 1 \n".format(cat.full_name))
        dcof.write("R_2j rateParam {0} dy_m105_160_vbf_amc_2j 1 \n".format(cat.full_name))
        #dcof.write("REWZ rateParam {0} ewk_lljj_mll105_160 1 \n".format(cat.full_name))
    dcof.write("{0} autoMCStats 0 0 1 \n".format(cat.full_name))
    dcof.write("\n")
    dcof.write("# Execute with:\n")
    dcof.write("# combine -n {0} -M FitDiagnostics -t -1 {1} \n".format(cat.full_name, os.path.basename(ofname)))

if __name__ == "__main__":

    cmdline_args = parse_args()

    pool = multiprocessing.Pool(cmdline_args.nthreads)

    from pars import cross_sections, categories, combined_categories
    from pars import signal_samples, shape_systematics, common_scale_uncertainties, scale_uncertainties

    #create a list of all the processes that need to be loaded from the result files
    datasets = yaml.load(open("data/datasets_NanoAODv6_Run2_mixv1.yml"), Loader=yaml.FullLoader)["datasets"]
    mc_samples_load = set([d["name"] for d in datasets])

    data_results_glob = cmdline_args.input + "/results/data_*.pkl"
    print("looking for {0}".format(data_results_glob))
    data_results = glob.glob(data_results_glob)
    if len(data_results) == 0:
        raise Exception("Did not find any data_*.pkl files in {0}, please check that this is a valid results directory and that the merge step has been completed".format(data_results_glob))

    eras = []
    if cmdline_args.eras is None:
        for dr in data_results:
            dr_filename = os.path.basename(dr)
            dr_filename_noext = dr_filename.split(".")[0]
            name, era = dr_filename_noext.split("_")
            eras += [era]
    else:
        eras = cmdline_args.eras
 
    print("Will make datacards and control plots for eras {0}".format(eras))
    for era in eras:
        rea = {}
        genweights = {}
        weight_xs = {}
        datacard_args = []
        plot_args = []
        plot_args_shape_syst = []
        plot_args_weights_off = []

        analysis = "results"
        input_folder = cmdline_args.input
        dd = "{0}/{1}".format(input_folder, analysis)
        res = {} 
        res["data"] = pickle.load(open(dd + "/data_{0}.pkl".format(era), "rb"))
        for mc_samp in mc_samples_load:
            res_file_name = dd + "/{0}_{1}.pkl".format(mc_samp, era)
            try:
                res[mc_samp] = pickle.load(open(res_file_name, "rb"))
            except Exception as e:
                print("Could not find results file {0}, skipping process {1}".format(res_file_name, mc_samp))

        analyses = [k for k in res["data"].keys() if not k in ["cache_metadata", "num_events", "int_lumi"]]

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
            int_lumi = res["data"]["int_lumi"]
            for mc_samp in res.keys():
                if mc_samp != "data":
                    genweights[mc_samp] = res[mc_samp]["genEventSumw"]
                    weight_xs[mc_samp] = get_cross_section(cross_sections, mc_samp, era) * int_lumi / genweights[mc_samp]
           
            with open(outdir + "/normalization.json", "w") as fi:
                fi.write(json.dumps({
                    "weight_xs": {k: v*genweight_scalefactor for k, v in weight_xs.items()},
                    "genweights": {k: v/genweight_scalefactor for k, v in genweights.items()},
                    "int_lumi": int_lumi,
                    }, indent=2)
                )

            histnames = []
            if cmdline_args.histnames is None:
                histnames = [h for h in res["data"]["baseline"].keys() if h.startswith("hist__")]
                print("Will create datacards and plots for all histograms")
                print("Use commandline option --histnames hist__dimuon__leading_muon_pt --histnames hist__dimuon__subleading_muon_pt ... to change that")
            else:
                histnames = cmdline_args.histnames
            #print("Processing histnames", histnames)
            
            for var in histnames:
                if var in ["hist_puweight", "hist__dijet_inv_mass_gen", "hist__dnn_presel__dnn_pred"]:
                    print("Skipping {0}".format(var))
                    continue
                if ("h_peak" in var):
                    mc_samples = categories["h_peak"]["datacard_processes"]
                    combined_mc_samples = combined_categories["h_peak"]["datacard_processes"]
                elif ("h_sideband" in var):
                    mc_samples = categories["h_sideband"]["datacard_processes"]
                    combined_mc_samples = combined_categories["h_sideband"]["datacard_processes"]
                elif ("z_peak" in var):
                    mc_samples = categories["z_peak"]["datacard_processes"]
                    combined_mc_samples = combined_categories["z_peak"]["datacard_processes"]
                else:
                    mc_samples = categories["dimuon"]["datacard_processes"]
                    combined_mc_samples = combined_categories["dimuon"]["datacard_processes"]


                #If we specified to only use certain processes in the datacard, keep only those
                if cmdline_args.keep_processes is None:
                    pass
                else:
                    mc_samples_new = []
                    for proc in mc_samples:
                        if proc in cmdline_args.keep_processes:
                            mc_samples_new += [proc]
                    mc_samples = mc_samples_new
                if len(mc_samples) == 0:
                    raise Exception(
                        "Could not match any MC process to histogram {0}, ".format(var) + 
                        "please check the definition in pars.py -> categories as "
                        "well as --keep-processes commandline option."
                        )

                histos = {}
                for sample in mc_samples + ["data"]:
                    histos[sample] = res[sample][analysis][var]

                print(era, analysis, var)
                datacard_args += [
                    (histos,
                    analysis,
                    ["data"] + mc_samples,
                    signal_samples,
                    ["data"] + combined_mc_samples,
                    combined_signal_samples,
                    var,
                    "nominal",
                    weight_xs,
                    shape_systematics,
                    common_scale_uncertainties,
                    scale_uncertainties,
                    outdir_datacards + "/{0}.txt".format(var),
                    era
                )]

                hdata = res["data"][analysis][var]["nominal"]
                plot_args += [(
                    histos, hdata, mc_samples, analysis,
                    var, "nominal", weight_xs, int_lumi, outdir, era, process_groups, extra_plot_kwargs.get(var, {}))]
                for weight in ["trigger", "id", "iso", "puWeight", "L1PreFiringWeight"]:
                    plot_args_weights_off += [(
                        histos, hdata, mc_samples, analysis,
                        var, "{0}__off".format(weight), weight_xs, int_lumi, outdir, era, process_groups, extra_plot_kwargs.get(var, {}))]

                for var_shape in controlplots_shape:
                    if var_shape in var: 
                        for mc_samp in mc_samples:
                            for unc in shape_systematics:
                                plot_args_shape_syst += [(
                                    histos, hdata, mc_samp, analysis,
                                    var, "nominal", weight_xs, int_lumi, outdir, era, unc)]
        rets = list(pool.map(make_pdf_plot, plot_args))
        #rets = list(pool.map(make_pdf_plot, plot_args_weights_off))
        rets = list(pool.map(create_datacard_combine_wrap, datacard_args))
        rets = list(pool.map(plot_variations, plot_args_shape_syst))

        #for args, retval in zip(datacard_args, rets):
        #    res, hd, mc_samples, analysis, var, weight, weight_xs, int_lumi, outdir, datataking_year = args
        #    htot_nominal, hd, htot_variated, hdelta_quadrature = retval
        #    wd = wasserstein_distance(htot_nominal.contents/np.sum(htot_nominal.contents), hd.contents/np.sum(hd.contents))
        #    print("DataToMC", analysis, var, wd)
