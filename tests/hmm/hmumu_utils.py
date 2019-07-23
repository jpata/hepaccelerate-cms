import pickle, json
import threading
import uproot
import copy
import psutil
import glob
import time
import numpy
import numpy as np
import sys
import os

import numba
import numba.cuda as cuda

import threading
from threading import Thread
from queue import Queue
import queue
import concurrent.futures

import hepaccelerate
import hepaccelerate.utils
from hepaccelerate.utils import Results
from hepaccelerate.utils import Dataset
from hepaccelerate.utils import Histogram
import hepaccelerate.backend_cpu as backend_cpu
from cmsutils.plotting import plot_hist_step
from cmsutils.decisiontree import DecisionTreeNode, DecisionTreeLeaf, make_random_node, grow_randomly, make_random_tree, prune_randomly, generate_cut_trees
from cmsutils.stats import likelihood, sig_q0_asimov, sig_naive

#global variables need to be configured here for the hepaccelerate backend and numpy library
ha = None
NUMPY_LIB = None

#Used to scale the genweight to prevent a numerical overflow
genweight_scalefactor = 1e-5

#Use these to turn on debugging
debug = False
debug_event_ids = []

data_runs = {
    "2017": [
        (294927, 297019, "RunA"),
        (297020, 299329, "RunB"),
        (299337, 302029, "RunC"),
        (302030, 303434, "RunD"),
        (303435, 304826, "RunE"),
        (304911, 306462, "RunF")
    ],

    "2016": [
        (272007, 275376, "RunB"),  
        (275657, 276283, "RunC"),  
        (276315, 276811, "RunD"),  
        (276831, 277420, "RunE"),  
        (277772, 278808, "RunF"),  
        (278820, 280385, "RunG"),  
        (280919, 284044, "RunH"),  
    ],

    "2018": [
        (315252, 316995, "RunA"),
        (316998, 319312, "RunB"),
        (319313, 320393, "RunC"),
        (320394, 325273, "RunD"),
        (325274, 325765, "RunE"),
    ]
}

runmap_numerical = {
    "RunA": 0,
    "RunB": 1,
    "RunC": 2,
    "RunD": 3,
    "RunE": 4,
    "RunF": 5,
    "RunG": 6,
    "RunH": 7,
}

runmap_numerical_r = {v: k for k, v in runmap_numerical.items()}

def analyze_data(
    data,
    use_cuda=False,
    is_mc=True,
    pu_corrections=None,
    lumimask=None,
    lumidata=None,
    rochester_corrections=None,
    lepsf_iso=None,
    lepsf_id=None,
    lepsf_trig=None,
    dnn_model=None,
    jetmet_corrections=None,
    parameters={},
    parameter_set_name="",
    doverify=False,
    do_sync = False,
    dataset_era = "",
    dataset_name = "",
    dataset_num_chunk = "",
    ):

    muons = data["Muon"]
    jets = data["Jet"]
    electrons = data["Electron"]
    trigobj = data["TrigObj"]
    scalars = data["eventvars"]


    check_and_fix_qgl(jets)

    #output histograms 
    hists = {}

    #temporary hack
    muons.hepaccelerate_backend = ha
    jets.hepaccelerate_backend = ha

    #associate the muon genpt to reco muons based on the NanoAOD index
    genJet = None
    genpart = None
    if is_mc:
        genJet = data["GenJet"]
        genpart = data["GenPart"]
        muons_genpt = NUMPY_LIB.zeros(muons.numobjects(), dtype=NUMPY_LIB.float32)
        jets_genpt = NUMPY_LIB.zeros(jets.numobjects(), dtype=NUMPY_LIB.float32)
        jets_genmass = NUMPY_LIB.zeros(jets.numobjects(), dtype=NUMPY_LIB.float32)
        if not use_cuda:
            get_genpt_cpu(muons.offsets, muons.genPartIdx, genpart.offsets, genpart.pt, muons_genpt)
            get_genpt_cpu(jets.offsets, jets.genJetIdx, genJet.offsets, genJet.pt, jets_genpt)
            get_genpt_cpu(jets.offsets, jets.genJetIdx, genJet.offsets, genJet.mass, jets_genmass)
        else:
            get_genpt_cuda[32,1024](muons.offsets, muons.genPartIdx, genpart.offsets, genpart.pt, muons_genpt)
            get_genpt_cuda[32,1024](jets.offsets, jets.genJetIdx, genJet.offsets, genJet.pt, jets_genpt)
            get_genpt_cuda[32,1024](jets.offsets, jets.genJetIdx, genJet.offsets, genJet.mass, jets_genmass)
        muons.attrs_data["genpt"] = muons_genpt
        jets.attrs_data["genpt"] = jets_genpt
        jets.attrs_data["genmass"] = jets_genmass

    #assign a numerical flag to each data event that corresponds to the data era
    if not is_mc:
        scalars["run_index"] = NUMPY_LIB.zeros_like(scalars["run"])
        scalars["run_index"][:] = -1
        runranges_list = data_runs[dataset_era]
        for run_start, run_end, run_name in runranges_list:
            msk = (scalars["run"] >= run_start) & (scalars["run"] <= run_end)
            scalars["run_index"][msk] = runmap_numerical[run_name]
        assert(NUMPY_LIB.sum(scalars["run_index"]==-1)==0)

    # scalars["run"] = NUMPY_LIB.array(scalars["run"], dtype=NUMPY_LIB.uint32)
    # scalars["luminosityBlock"] = NUMPY_LIB.array(scalars["run"], dtype=NUMPY_LIB.uint32)

    #Get the mask of events that pass trigger selection
    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    mask_events = select_events_trigger(scalars, parameters, mask_events, parameters["hlt_bits"][dataset_era])


    #Compute integrated luminosity on data sample and apply golden JSON
    int_lumi = 0
    if not is_mc:
        runs = NUMPY_LIB.asnumpy(scalars["run"])
        lumis = NUMPY_LIB.asnumpy(scalars["luminosityBlock"])
        mask_lumi_golden_json = NUMPY_LIB.array(lumimask[dataset_era](runs, lumis))
        #print("Number of events pre-json: {0}".format(mask_events.sum()))
        mask_events = mask_events & mask_lumi_golden_json
        #print("Number of events post-json: {0}".format(mask_events.sum()))
        if not is_mc and not (lumimask is None):
            if parameter_set_name == "baseline":
                mask_events = mask_events & mask_lumi_golden_json
                #get integrated luminosity in this file
                if not (lumidata is None):
                    int_lumi = get_int_lumi(runs, lumis, NUMPY_LIB.asnumpy(mask_lumi_golden_json), lumidata[dataset_era])

    #Compute event weights
    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"] * genweight_scalefactor
        if debug:
            print("mean genWeight=", scalars["genWeight"].mean())
            print("sum genWeight=", scalars["genWeight"].sum())
        pu_weights, pu_weights_up, pu_weights_down = compute_pu_weights(
            pu_corrections[dataset_era],
            weights["nominal"],
            scalars["Pileup_nTrueInt"],
            scalars["PV_npvsGood"])

        if debug:
            print("pu_weights", pu_weights.mean(), pu_weights.std())
            print("pu_weights_up", pu_weights_up.mean(), pu_weights_up.std())
            print("pu_weights_down", pu_weights_down.mean(), pu_weights_down.std())

        weights["puWeight_off"] = weights["nominal"] 
        weights["puWeight__up"] = weights["nominal"] * pu_weights_up
        weights["puWeight__down"] = weights["nominal"] * pu_weights_down
        weights["nominal"] = weights["nominal"] * pu_weights

    #Apply Rochester corrections to leading and subleading muon momenta
    if parameters["do_rochester_corrections"]:
        if debug:
            print("Before applying Rochester corrections: muons.pt={0:.2f} +- {1:.2f}".format(muons.pt.mean(), muons.pt.std()))
        do_rochester_corrections(
            is_mc,
            rochester_corrections[dataset_era],
            muons)
        if debug:
            print("After applying Rochester corrections muons.pt={0:.2f} +- {1:.2f}".format(muons.pt.mean(), muons.pt.std()))


    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(
        scalars,
        muons, trigobj, mask_events,
        parameters["muon_pt_leading"][dataset_era], parameters["muon_pt"],
        parameters["muon_eta"], parameters["muon_iso"],
        parameters["muon_id"][dataset_era], parameters["muon_trigger_match_dr"]
    )
    
    # Create arrays with just the leading and subleading particle contents for easier management
    mu_attrs = ["pt", "eta", "phi", "mass", "pdgId", "nTrackerLayers"]
    if is_mc:
        mu_attrs += ["genpt"]
    leading_muon = muons.select_nth(0, ret_mu["selected_events"], ret_mu["selected_muons"], attributes=mu_attrs)
    subleading_muon = muons.select_nth(1, ret_mu["selected_events"], ret_mu["selected_muons"], attributes=mu_attrs)
    if doverify:
        assert(NUMPY_LIB.all(leading_muon["pt"][leading_muon["pt"]>0] > parameters["muon_pt_leading"]))
        assert(NUMPY_LIB.all(subleading_muon["pt"][subleading_muon["pt"]>0] > parameters["muon_pt"]))

    #Compute lepton scale factors
    if parameters["do_lepton_sf"] and is_mc:
        sf_tot = compute_lepton_sf(leading_muon, subleading_muon,
            lepsf_iso[dataset_era], lepsf_id[dataset_era], lepsf_trig[dataset_era],
            use_cuda, dataset_era, NUMPY_LIB, debug)
        weights["leptonsf_off"] = weights["nominal"]
        weights["nominal"] = weights["nominal"] * sf_tot

    hists["hist__dimuon__npvs"] = fill_with_weights(
        scalars["PV_npvsGood"], weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0,100,101))
    
    #Just a check to verify that there are exactly 2 muons per event
    if doverify:
        z = ha.sum_in_offsets(
            muons,
            ret_mu["selected_muons"],
            ret_mu["selected_events"],
            ret_mu["selected_muons"],
            dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))

    # Get the selected electrons
    ret_el = get_selected_electrons(electrons, parameters["extra_electrons_pt"], parameters["extra_electrons_eta"], parameters["extra_electrons_id"])
    
    # Get the invariant mass of the dimuon system and compute mass windows
    inv_mass = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"])
    #FIXME: check this
    inv_mass[NUMPY_LIB.isnan(inv_mass)] = 0
    inv_mass[NUMPY_LIB.isinf(inv_mass)] = 0

    masswindow_70_110 = ((inv_mass >= 70) & (inv_mass < 110))
    masswindow_110_150 = ((inv_mass >= 110) & (inv_mass < 150))
    masswindow_120_130 = ((inv_mass >= 120) & (inv_mass < 130))
    masswindow_exclude_120_130 = masswindow_110_150 & (NUMPY_LIB.invert((inv_mass >= 120) & (inv_mass <= 130)))

    #get the number of additional muons (not OS) that pass ID and iso cuts
    n_additional_muons = ha.sum_in_offsets(muons, ret_mu["additional_muon_sel"], ret_mu["selected_events"], ret_mu["additional_muon_sel"], dtype=NUMPY_LIB.int8)
    n_additional_electrons = ha.sum_in_offsets(electrons, ret_el["additional_electron_sel"], ret_mu["selected_events"], ret_el["additional_electron_sel"], dtype=NUMPY_LIB.int8)
    n_additional_leptons = n_additional_muons + n_additional_electrons

    fill_muon_hists(
        hists, scalars, weights, ret_mu, inv_mass,
        leading_muon, subleading_muon, parameters,
        masswindow_110_150, masswindow_120_130, NUMPY_LIB)

    #Apply JEC, create a dictionary of variated jet momenta
    jet_pt_syst = apply_jec(jets, scalars, parameters, jetmet_corrections[dataset_era][parameters["jec_tag"][dataset_era]], NUMPY_LIB, use_cuda, is_mc)

    # Loop over all jet momentum variations and do analysis that depends on jets
    for jet_syst_name, jet_pt_vec in jet_pt_syst.items():
       
        #For the moment, skip other jet systematics 
        #if jet_syst_name[0] != "nominal":
        #    continue
        
        # For events where the JEC/JER was variated, fill only the nominal weight
        weights_selected = select_weights(weights, jet_syst_name)

        # In case the pt vector is a function, we evaluate it
        # This is used for GPUs to transfer the pt vector only when needed
        if callable(jet_pt_vec):
            jet_pt_vec = jet_pt_vec()

        # Configure the jet pt vector to the variated one
        # Would need to also do the mass here
        jets.pt = jet_pt_vec

        #get the passing jets for events that pass muon selection
        ret_jet = get_selected_jets(
            scalars,
            jets, muons, genJet, genpart,
            ret_mu["selected_muons"], mask_events,
            parameters["jet_pt_subleading"][dataset_era],
            parameters["jet_eta"],
            parameters["jet_mu_dr"],
            parameters["jet_id"],
            parameters["jet_puid"],
            parameters["jet_btag"][dataset_era],
            is_mc, use_cuda
        )

        # Set this default value as in Nan and Irene's code
        ret_jet["dijet_inv_mass"][ret_jet["num_jets"] < 2] = -1000.0

        # Get the data for the leading and subleading jets as contiguous vectors
        leading_jet = jets.select_nth(0, ret_mu["selected_events"], ret_jet["selected_jets"], attributes=["pt", "eta", "phi", "mass", "qgl"])
        subleading_jet = jets.select_nth(1, ret_mu["selected_events"], ret_jet["selected_jets"], attributes=["pt", "eta", "phi", "mass", "qgl"])

        category =  assign_category_nan_irene(
            ret_jet["num_jets"], ret_jet["num_jets_btag"],
            n_additional_muons, n_additional_electrons,
            ret_jet["dijet_inv_mass"], leading_jet, subleading_jet,
            parameters["cat5_dijet_inv_mass"]
        )
        scalars["category"] = category

        if do_sync and jet_syst_name[0] == "nominal":
            sync_printout(ret_mu, muons, scalars,
                leading_muon, subleading_muon, inv_mass,
                n_additional_muons, n_additional_electrons,
                ret_jet, leading_jet, subleading_jet)

        #compute DNN input variables in 2 muon, >=2jet region
        dnn_presel = (ret_mu["selected_events"]) & (ret_jet["num_jets"] >= 2) & (leading_jet["pt"] > parameters["jet_pt_leading"][dataset_era])

        #apply VBF filter cut
        if is_mc and dataset_name in parameters["vbf_filter"]:
            mask_dijet_genmass = (ret_jet["dijet_inv_mass_gen"] > parameters["vbf_filter_mjj_cut"])
            mask_2gj = ret_jet["num_good_genjets"]>=2
            invert_mask = parameters["vbf_filter"][dataset_name]
            if invert_mask:
                mask_dijet_genmass = NUMPY_LIB.invert(mask_dijet_genmass)

            mask_out = NUMPY_LIB.ones_like(mask_dijet_genmass)
            mask_out[mask_2gj & NUMPY_LIB.invert(mask_dijet_genmass)] = False
            if debug:
                print("sample", dataset_name, "numev", len(mask_out), "2gj", mask_2gj.sum(), "2gj&&mjj", (mask_2gj&mask_dijet_genmass).sum(), "out", mask_out.sum()) 
            dnn_presel = dnn_presel & mask_out

        #Compute the DNN inputs, the DNN output, fill the DNN input and output variable histograms
        hists_dnn = {}
        dnn_prediction = None
        #dnn_vars, dnn_prediction, weights_dnn = compute_fill_dnn(parameters, use_cuda, dnn_presel, dnn_model,
        #   scalars, leading_muon, subleading_muon, leading_jet, subleading_jet,
        #   weights_selected, hists_dnn)

        #Assign the final analysis discriminator based on category
        scalars["final_discriminator"] = NUMPY_LIB.zeros_like(inv_mass)
        if not (dnn_prediction is None):
            inds_nonzero = NUMPY_LIB.nonzero(dnn_presel)[0]
            if len(inds_nonzero) > 0:
                ha.copyto_dst_indices(scalars["final_discriminator"], dnn_prediction, inds_nonzero)
            scalars["final_discriminator"][category != 5] = 0

            #Add some additional debugging info to the DNN training ntuples
            dnn_vars["cat_index"] = category[dnn_presel]
            dnn_vars["run"] = scalars["run"][dnn_presel]
            dnn_vars["lumi"] = scalars["luminosityBlock"][dnn_presel]
            dnn_vars["event"] = scalars["event"][dnn_presel]
            dnn_vars["Higgs_mass"] = inv_mass[dnn_presel]
            dnn_vars["dnn_pred"] = dnn_prediction

            #Save the DNN training ntuples as npy files
            if parameters["save_dnn_vars"] and jet_syst_name[0] == "nominal" and parameter_set_name == "baseline":
                dnn_vars_np = {k: NUMPY_LIB.asnumpy(v) for k, v in dnn_vars.items()}
                if is_mc:
                    dnn_vars_np["genweight"] = NUMPY_LIB.asnumpy(scalars["genWeight"][dnn_presel])
                arrs = []
                names = []
                for k, v in dnn_vars_np.items():
                    arrs += [v]
                    names += [k]
                arrdata = np.core.records.fromarrays(arrs, names=names)
                outpath = "{0}/{1}".format(parameters["dnn_vars_path"], dataset_era) 
                if not os.path.isdir(outpath):
                    os.makedirs(outpath)
                np.save("{0}/{1}_{2}.npy".format(outpath, dataset_name, dataset_num_chunk), arrdata, allow_pickle=False)

        #Put the DNN histograms into the result dictionary
        for k, v in hists_dnn.items():
            if k not in hists:
                hists[k] = {}
            if jet_syst_name[0] == "nominal":
                hists[k].update(hists_dnn[k])
            else:
                hists[k][jet_syst_name[0] + "__" + jet_syst_name[1]] = hists_dnn[k]["nominal"]

        #Auto-categorization (currently not used)
        #Split the events into categories based on the categorization cut tree
        #cut_pre = ret_mu["selected_events"] & masswindow_120_130
        #for cat_tree_name, cat_tree in parameters["categorization_trees"].items():
        #    hists_cat = Results({})
        #    categories = cat_tree.predict(len(inv_mass), {
        #        "dimuon_inv_mass": inv_mass,
        #        "dijet_inv_mass": ret_jet["dijet_inv_mass"],
        #        "num_jets": ret_jet["num_jets"],
        #        "num_jets_btag": ret_jet["num_jets_btag"],
        #        "leading_mu_abs_eta": NUMPY_LIB.abs(leading_muon["eta"]),
        #        "additional_leptons": n_additional_leptons,
        #    })
        #    
        #    #Make histograms for each category
        #    for cat in [l.value for l in cat_tree.get_all_leaves()]:
        #        #Choose events that pass this category
        #        cut = cut_pre & (categories == cat)
        #        #Fill the invariant mass distribution for each category
        #        hists_cat["hist__cat{0}__inv_mass".format(int(cat))] = Results(fill_with_weights(
        #            inv_mass, weights_selected,
        #            cut,
        #            NUMPY_LIB.linspace(120, 130, parameters["inv_mass_bins"])
        #        ))
        #    hists[cat_tree_name] = hists_cat

        #Save some histograms
        #update_histograms_systematic(
        #    hists,
        #    "hist__dimuon_jge1__leading_jet_pt",
        #    jet_syst_name, leading_jet["pt"], weights_selected,
        #   ret_mu["selected_events"] & (ret_jet["num_jets"]>=1), NUMPY_LIB.linspace(30, 200.0, 101))

        #update_histograms_systematic(
        #    hists,
        #    "hist__dimuon_invmass_110_150_exclude_120_130_jge1__leading_jet_pt",
        #    jet_syst_name, leading_jet["pt"], weights_selected,
        #   ret_mu["selected_events"] & (ret_jet["num_jets"]>=1) & masswindow_110_150 & masswindow_exclude_120_130, NUMPY_LIB.linspace(30, 200.0, 101))

        #update_histograms_systematic(
        #    hists,
        #    "hist__dimuon_invmass_110_150_exclude_120_130_jge2__subleading_jet_pt",
        #    jet_syst_name, subleading_jet["pt"], weights_selected,
        #   ret_mu["selected_events"] & (ret_jet["num_jets"]>=2) & masswindow_110_150 & masswindow_exclude_120_130, NUMPY_LIB.linspace(30, 100.0, 101))
        #
        #update_histograms_systematic(
        #    hists,
        #    "hist__dimuon_invmass_110_150_exclude_120_130_jge1__leading_jet_eta",
        #    jet_syst_name, leading_jet["eta"], weights_selected,
        #   ret_mu["selected_events"] & (ret_jet["num_jets"]>=1) & masswindow_110_150 & masswindow_exclude_120_130, NUMPY_LIB.linspace(-5.0, 5.0, 30))

        #update_histograms_systematic(
        #    hists,
        #    "hist__dimuon_invmass_110_150_exclude_120_130_jge2__subleading_jet_eta",
        #    jet_syst_name, subleading_jet["eta"], weights_selected,
        #   ret_mu["selected_events"] & (ret_jet["num_jets"]>=2) & masswindow_110_150 & masswindow_exclude_120_130, NUMPY_LIB.linspace(-5.0, 5.0, 30))
        
        update_histograms_systematic(
            hists,
            "hist__dnn_presel__inv_mass",
            jet_syst_name, inv_mass, weights_selected,
           dnn_presel, NUMPY_LIB.linspace(70, 150, 41))

        #Save histograms for numerical categories (cat5 only right now) and all mass bins
        for massbin_name, massbin_msk, mass_edges in [
                #("all", ret_mu["selected_events"]), 
                #("110_150", masswindow_110_150, (110, 150)),
                #("120_130", masswindow_120_130, (120, 130)),
                ("70_110", masswindow_70_110, (70, 110))]:

            update_histograms_systematic(
                hists,
                "hist__dimuon_invmass_{0}__inv_mass".format(massbin_name),
                jet_syst_name, inv_mass, weights_selected,
                dnn_presel & massbin_msk, NUMPY_LIB.linspace(mass_edges[0], mass_edges[1], 41))

            update_histograms_systematic(
                hists,
                "hist__dimuon_invmass_{0}__numjet".format(massbin_name),
                jet_syst_name, ret_jet["num_jets"], weights_selected,
                dnn_presel & massbin_msk, NUMPY_LIB.linspace(0, 10, 11))

            update_histograms_systematic(
                hists,
                "hist__dimuon_invmass_{0}__dijet_inv_mass".format(massbin_name),
                jet_syst_name, ret_jet["dijet_inv_mass"], weights_selected,
                dnn_presel & massbin_msk, NUMPY_LIB.linspace(0, 1000, 41))

            #if jet_syst_name[0] == "nominal":
            #    update_histograms_systematic(
            #        hists,
            #        "hist__dimuon_invmass_{0}__numjet".format(massbin_name),
            #        jet_syst_name, ret_jet["num_jets"], weights_selected,
            #       massbin_msk, NUMPY_LIB.linspace(0, 10, 11))
            #    
            #    update_histograms_systematic(
            #        hists,
            #        "hist__dimuon_invmass_{0}__leading_mu_pt".format(massbin_name),
            #        jet_syst_name, leading_muon["pt"], weights_selected,
            #       massbin_msk, NUMPY_LIB.linspace(0, 200, 100))
            #    
            #    update_histograms_systematic(
            #        hists,
            #        "hist__dimuon_invmass_{0}__subleading_mu_pt".format(massbin_name),
            #        jet_syst_name, subleading_muon["pt"], weights_selected,
            #       massbin_msk, NUMPY_LIB.linspace(0, 200, 100))
            #    
            #    update_histograms_systematic(
            #        hists,
            #        "hist__dimuon_invmass_{0}__leading_mu_eta".format(massbin_name),
            #        jet_syst_name, leading_muon["eta"], weights_selected,
            #       massbin_msk, NUMPY_LIB.linspace(-3, 3, 100))
            #    
            #    update_histograms_systematic(
            #        hists,
            #        "hist__dimuon_invmass_{0}__subleading_mu_eta".format(massbin_name),
            #        jet_syst_name, subleading_muon["eta"], weights_selected,
            #       massbin_msk, NUMPY_LIB.linspace(-3, 3, 100))

            for icat in [5, ]:
                msk_cat = category == icat

                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__inv_mass".format(massbin_name, icat),
                    jet_syst_name, inv_mass, weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(mass_edges[0], mass_edges[1], 41))

                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__leading_jet_pt".format(massbin_name, icat),
                    jet_syst_name, leading_jet["pt"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(30, 200.0, 41))
                
                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__subleading_jet_pt".format(massbin_name, icat),
                    jet_syst_name, subleading_jet["pt"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(30, 200.0, 41))

                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__leading_jet_eta".format(massbin_name, icat),
                    jet_syst_name, leading_jet["eta"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(-4.7, 4.7, 41))
                
                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__subleading_jet_eta".format(massbin_name, icat),
                    jet_syst_name, subleading_jet["eta"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(-4.7, 4.7, 41))

                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__dijet_inv_mass".format(massbin_name, icat),
                    jet_syst_name, ret_jet["dijet_inv_mass"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(400, 1000.0, 41))

                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__num_soft_jets".format(massbin_name, icat),
                    jet_syst_name, scalars["SoftActivityJetNjets5"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(0, 10, 11))
                
                update_histograms_systematic(
                    hists,
                    "hist__dimuon_invmass_{0}_cat{1}__num_jets".format(massbin_name, icat),
                    jet_syst_name, ret_jet["num_jets"], weights_selected,
                    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(0, 10, 11))

                #update_histograms_systematic(
                #    hists,
                #    "hist__dimuon_invmass_{0}_cat{1}__subleading_jet_pt".format(massbin_name, icat),
                #    jet_syst_name, subleading_jet["pt"], weights_selected,
                #    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(30, 200.0, 101))
                #
                #update_histograms_systematic(
                #    hists,
                #    "hist__dimuon_invmass_{0}_cat{1}__leading_jet_eta".format(massbin_name, icat),
                #    jet_syst_name, leading_jet["eta"], weights_selected,
                #    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(30, 200.0, 101))
                #
                #update_histograms_systematic(
                #    hists,
                #    "hist__dimuon_invmass_{0}_cat{1}__subleading_jet_eta".format(massbin_name, icat),
                #    jet_syst_name, subleading_jet["eta"], weights_selected,
                #    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(30, 200.0, 101))

                #update_histograms_systematic(
                #    hists,
                #    "hist__dimuon_invmass_{0}_cat{1}__final_discriminator".format(massbin_name, icat),
                #    jet_syst_name, scalars["final_discriminator"], weights_selected,
                #    dnn_presel & massbin_msk & msk_cat, NUMPY_LIB.linspace(0, 1, 101))

                ##save all DNN input variables
                #for varname in dnn_vars.keys():
                #    if varname in parameters["dnn_input_histogram_bins"].keys():
                #        hb = parameters["dnn_input_histogram_bins"][varname]
                #        update_histograms_systematic(
                #            hists,
                #            "hist__dimuon_invmass_{0}_cat{1}__{2}".format(massbin_name, icat, varname),
                #            jet_syst_name, dnn_vars[varname], weights_dnn,
                #            (ret_mu["selected_events"] & massbin_msk & msk_cat)[dnn_presel], NUMPY_LIB.linspace(*hb))

    #end of jet systematic loop

    # Collect results
    ret = Results({
        "int_lumi": int_lumi,
    })

    for histname, r in hists.items():
        ret[histname] = Results(r)

    ret["numev_passed"] = get_numev_passed(
        muons.numevents(), {
        "trigger": mask_events,
        "muon": ret_mu["selected_events"]
    })
 
#save raw data arrays
#    ret["dimuon_inv_mass"] = [inv_mass]
#    ret["num_jets"] = [ret_jet["num_jets"]]
#    ret["num_jets_btag"] = [ret_jet["num_jets_btag"]]
#    ret["dijet_inv_mass"] = [ret_jet["dijet_inv_mass"]]
#    ret["selected_events"] = [ret_mu["selected_events"]]
#    ret["additional_leptons"] = [additional_leptons]

    return ret

#Main analysis entry point
def run_analysis(
    args, outpath, datasets, parameters,
    chunksize, maxfiles,
    lumidata, lumimask, pu_corrections, rochester_corrections,
    lepsf_iso, lepsf_id, lepsf_trig, dnn_model, jetmet_corrections):

    #Keep track of number of events
    nev_total = 0
    nev_loaded = 0
    t0 = time.time()
    
    if "cache" in args.action:
       print("Running the 'cache' step of the analysis, ROOT files will be opened and branches will be uncompressed")
       print("Will retrieve dataset filenames based on existing ROOT files on filesystem in datapath={0}".format(args.datapath)) 
       try:
           os.makedirs(args.cache_location)
       except Exception as e:
           pass
       filenames_cache = {}
       for datasetname, dataset_era, globpattern, is_mc in datasets:
           filenames_all = glob.glob(args.datapath + globpattern, recursive=True)
           filenames_all = [fn for fn in filenames_all if not "Friend" in fn]
           filenames_cache[datasetname + "_" + dataset_era] = [fn.replace(args.datapath, "") for fn in filenames_all]
   
       #save all dataset filenames to a json file 
       print("Creating a json dump of all the dataset filenames")
       with open(args.cache_location + "/datasets.json", "w") as fi:
           fi.write(json.dumps(filenames_cache, indent=2))
    else:
       print("Running the 'analyze' step of the analysis, loading cached branch data and using it in physics code via analyze_data()")
       print("Loading list of filenames loaded from {0}/datasets.json".format(args.cache_location))
       filenames_cache = json.load(open(args.cache_location + "/datasets.json", "r"))

    for dataset, filenames in filenames_cache.items():
        print("dataset {0} consists of {1} ROOT files".format(dataset, len(filenames)))
        if len(filenames) == 0:
            raise Exception("No files found for dataset {0}".format(dataset))
            
    processed_size_mb = 0
    for datasetname, dataset_era, globpattern, is_mc in datasets:
        filenames_all = filenames_cache[datasetname + "_" + dataset_era]
        filenames_all = [args.datapath + "/" + fn for fn in filenames_all]
 
        print("Processing dataset {0}_{1}".format(datasetname, dataset_era))
        if maxfiles[dataset_era] > 0:
            mf = maxfiles[dataset_era]
            if datasetname == "data":
                mf = 10*mf
            filenames_all = filenames_all[:mf]

        datastructure = create_datastructure(is_mc, dataset_era)

        if "cache" in args.action:

            #Used for preselection in the cache
            hlt_bits = parameters["baseline"]["hlt_bits"][dataset_era]
                
            _nev_total, _processed_size_mb = cache_data(
                filenames_all, datasetname, datastructure,
                args.cache_location, args.datapath, is_mc,
                hlt_bits,
                nworkers=args.nthreads)
            nev_total += _nev_total
            processed_size_mb += _processed_size_mb

        elif "analyze" in args.action:

            #Create a thread that will load data in the background
            training_set_generator = InputGen(
                datasetname, dataset_era, list(filenames_all), datastructure,
                args.nthreads, chunksize[dataset_era], args.cache_location, args.datapath)
            threadk = thread_killer()
            threadk.set_tokill(False)
            train_batches_queue = Queue(maxsize=10)
            
            #Start the thread if using a multithreaded approach
            if args.async_data:
                for _ in range(1):
                    t = Thread(target=threaded_batches_feeder, args=(threadk, train_batches_queue, training_set_generator))
                    t.start()

            rets = []
            num_processed = 0
           
            cache_metadata = []
            #loop over all data, call the analyze function
            while num_processed < len(training_set_generator.paths_chunks):

                # In case we are processing data synchronously, just load the dataset here
                # and put to queue.
                if not args.async_data:
                    ds = training_set_generator.nextone()
                    if ds is None:
                        break
                    train_batches_queue.put(ds)

                #Progress indicator for each chunk of files
                sys.stdout.write(".");sys.stdout.flush()

                #Process the dataset
                ret, nev, memsize = event_loop(
                    train_batches_queue,
                    args.use_cuda,
                    verbose=False, is_mc=is_mc, lumimask=lumimask,
                    lumidata=lumidata,
                    pu_corrections=pu_corrections,
                    rochester_corrections=rochester_corrections,
                    lepsf_iso=lepsf_iso,
                    lepsf_id=lepsf_id,
                    lepsf_trig=lepsf_trig,
                    parameters=parameters,
                    dnn_model=dnn_model,
                    jetmet_corrections=jetmet_corrections,
                    do_sync = args.do_sync) 

                rets += [ret]
                processed_size_mb += memsize
                nev_total += sum([md["numevents"] for md in ret["cache_metadata"]])
                nev_loaded += nev
                num_processed += 1
            print()

            #clean up threads
            threadk.set_tokill(True)

            #save output
            ret = sum(rets, Results({}))
            if is_mc:
                ret["genEventSumw"] = genweight_scalefactor * sum([md["precomputed_results"]["genEventSumw"] for md in ret["cache_metadata"]])
                ret["genEventSumw2"] = genweight_scalefactor * sum([md["precomputed_results"]["genEventSumw2"] for md in ret["cache_metadata"]])
            ret.save_json("{0}/{1}_{2}.json".format(outpath, datasetname, dataset_era))
    
    t1 = time.time()
    dt = t1 - t0
    print("Overall processed {nev:.2E} ({nev_loaded:.2E} loaded) events in total {size:.2f} GB, {dt:.1f} seconds, {evspeed:.2E} Hz, {sizespeed:.2f} MB/s".format(
        nev=nev_total, nev_loaded=nev_loaded, dt=dt, size=processed_size_mb/1024.0, evspeed=nev_total/dt, sizespeed=processed_size_mb/dt)
    )

    bench_ret = {}
    bench_ret.update(args.__dict__)
    bench_ret["hostname"] = os.uname()[1]
    bench_ret["nev_total"] = nev_total
    bench_ret["total_time"] = dt
    bench_ret["evspeed"] = nev_total/dt/1000/1000
    with open("analysis_benchmarks.txt", "a") as of:
        of.write(json.dumps(bench_ret) + '\n')

def event_loop(train_batches_queue, use_cuda, **kwargs):
    ds = train_batches_queue.get(block=True)
    #print("event_loop nev={0}, queued={1}".format(len(ds), train_batches_queue.qsize()))

    #copy dataset to GPU and make sure future operations are done on it
    if use_cuda:
        import cupy
        ds.numpy_lib = cupy
        ds.move_to_device(cupy)

    parameters = kwargs.pop("parameters")

    ret = {}
    for parameter_set_name, parameter_set in parameters.items():
        ret[parameter_set_name] = ds.analyze(
            analyze_data,
            use_cuda = use_cuda,
            parameter_set_name = parameter_set_name,
            parameters = parameter_set,
            dataset_era = ds.era,
            dataset_name = ds.name,
            dataset_num_chunk = ds.num_chunk,
            **kwargs)
    ret["num_events"] = len(ds)

    train_batches_queue.task_done()

    #clean up CUDA memory
    if use_cuda:
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
     
    ret["cache_metadata"] = ds.cache_metadata 
    return ret, len(ds), ds.memsize()/1024.0/1024.0

def get_histogram(data, weights, bins):
    """Given N-unit vectors of data and weights, returns the histogram in bins
    """
    return Histogram(*ha.histogram_from_vector(data, weights, bins))

def get_selected_muons(
    scalars,
    muons, trigobj, mask_events,
    mu_pt_cut_leading, mu_pt_cut_subleading,
    mu_aeta_cut, mu_iso_cut, muon_id_type,
    muon_trig_match_dr):
    """
    Given a list of muons in events, selects the muons that pass quality, momentum and charge criteria.
    Selects events that have at least 2 such muons. Selections are made by producing boolean masks.

    muons (JaggedStruct) - The muon content of a given file
    trigobj (JaggedStruct) - The trigger objects
    mask_events (array of bool) - a mask of events that are used for muon processing
    mu_pt_cut_leading (float) - the pt cut on the leading muon
    mu_pt_cut_subleading (float) - the pt cut on all other muons
    mu_aeta_cut (float) - upper abs eta cut signal on muon
    mu_iso_cut (float) - cut to choose isolated muons
    muon_id_type (string) - "medium" or "tight" for the muon ID
    muon_trig_match_dr (float) - dR matching criterion between trigger object and leading muon

    """
    passes_iso = muons.pfRelIso04_all < mu_iso_cut

    if muon_id_type == "medium":
        passes_id = muons.mediumId == 1
    elif muon_id_type == "tight":
        passes_id = muons.tightId == 1
    else:
        raise Exception("unknown muon id: {0}".format(muon_id_type))

    #find muons that pass ID
    passes_global = (muons.isGlobal == 1)
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(muons.eta) < mu_aeta_cut
    muons_passing_id =  (
        passes_global & passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )

    #Get muons that are high-pt and are matched to trigger object
    mask_trigger_objects_mu = (trigobj.id == 13)
    muons_matched_to_trigobj = NUMPY_LIB.invert(ha.mask_deltar_first(
        muons, muons_passing_id & passes_leading_pt, trigobj,
        mask_trigger_objects_mu, muon_trig_match_dr
    ))
    muons.attrs_data["triggermatch"] = muons_matched_to_trigobj
    muons.attrs_data["pass_id"] = muons_passing_id
    muons.attrs_data["passes_leading_pt"] = passes_leading_pt

    #At least one muon must be matched to trigger object, find such events
    events_passes_triggermatch = ha.sum_in_offsets(
        muons, muons_matched_to_trigobj, mask_events,
        muons.masks["all"], NUMPY_LIB.int8
    ) >= 1

    #select events that have muons passing cuts: 2 passing ID, 1 passing leading pt, 2 passing subleading pt
    events_passes_muid = ha.sum_in_offsets(
        muons, muons_passing_id, mask_events, muons.masks["all"],
        NUMPY_LIB.int8) >= 2
    events_passes_leading_pt = ha.sum_in_offsets(
        muons, muons_passing_id & passes_leading_pt, mask_events,
        muons.masks["all"], NUMPY_LIB.int8) >= 1
    events_passes_subleading_pt = ha.sum_in_offsets(
        muons, muons_passing_id & passes_subleading_pt,
        mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2

    #Get the mask of selected events
    base_event_sel = (
        mask_events &
        events_passes_triggermatch &
        events_passes_muid &
        events_passes_leading_pt &
        events_passes_subleading_pt
    )

    #Find two opposite sign muons among the muons passing ID and subleading pt
    muons_passing_os = ha.select_muons_opposite_sign(
        muons, muons_passing_id & passes_subleading_pt)
    events_passes_os = ha.sum_in_offsets(
        muons, muons_passing_os, mask_events,
        muons.masks["all"], NUMPY_LIB.int32) == 2

    muons.attrs_data["pass_os"] = muons_passing_os
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    additional_muon_sel = muons_passing_id & NUMPY_LIB.invert(muons_passing_os)
    muons.masks["iso_id_aeta"] = passes_iso & passes_id & passes_aeta

    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("muons")
            jaggedstruct_print(muons, idx,
                ["pt", "eta", "phi", "charge", "pfRelIso04_all", "mediumId",
                "isGlobal", "isTracker", 
                "triggermatch", "pass_id", "passes_leading_pt"])

    return {
        "selected_events": final_event_sel,
        "muons_passing_id_pt": muons_passing_id & passes_subleading_pt,
        "selected_muons": final_muon_sel,
        "muons_passing_os": muons_passing_os,
        "additional_muon_sel": additional_muon_sel,
    }

def get_bit_values(array, bit_index):
    """
    Given an array of N binary values (e.g. jet IDs), return the bit value at bit_index in [0, N-1].
    """
    return (array & 2**(bit_index)) >> 1

def get_selected_jets(
    scalars,
    jets, muons, genJet, genParticle,
    mask_muons,
    mask_events,
    jet_pt_cut_subleading,
    jet_eta_cut,
    jet_dr_cut,
    jet_id,
    jet_puid,
    jet_btag,
    is_mc,
    use_cuda
    ):
    """
    Given jets and selected muons in events, choose jets that pass quality
    criteria and that are not dR-matched to muons.

    jets (JaggedStruct)
    muons (JaggedStruct)
    mask_muons (array of bool)
    mask_events (array of bool)
    jet_pt_cut (float)
    jet_eta_cut(float)
    jet_dr_cut (float)
    jet_id (string)
    jet_puid (string)
    jet_btag (string)
    """

    #Jet ID flags bit0 is loose (always false in 2017 since it does not exist), bit1 is tight, bit2 is tightLepVeto
    if jet_id == "tight":
        pass_jetid = jets.jetId >= 2
    elif jet_id == "loose":
        pass_jetid = jets.jetId >= 1

    #The value is a bit representation of the fulfilled working points: tight (1), medium (2), and loose (4).
    #As tight is also medium and medium is also loose, there are only 4 different settings: 0 (no WP, 0b000), 4 (loose, 0b100), 6 (medium, 0b110), and 7 (tight, 0b111).
    if jet_puid == "loose":
        pass_jet_puid = jets.puId >= 4
    elif jet_puid == "medium":
        pass_jet_puid = jets.puId >= 6
    elif jet_puid == "tight":
        pass_jet_puid = jets.puId >= 7
    elif jet_puid == "none":
        pass_jet_puid = NUMPY_LIB.ones(jets.numobjects(), dtype=NUMPY_LIB.bool)

    pass_qgl = jets.qgl > -1 
    selected_jets = (
        (jets.pt > jet_pt_cut_subleading) &
        (NUMPY_LIB.abs(jets.eta) < jet_eta_cut) &
        pass_jetid & pass_jet_puid & pass_qgl
    )
    jets_pass_dr = ha.mask_deltar_first(
        jets, selected_jets, muons,
        muons.masks["iso_id_aeta"], jet_dr_cut)

    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = selected_jets & jets_pass_dr
   
    #produce a mask that selects the first two selected jets 
    first_two_jets = NUMPY_LIB.zeros_like(selected_jets)
   
    inds = NUMPY_LIB.zeros_like(mask_events, dtype=NUMPY_LIB.int32) 
    targets = NUMPY_LIB.ones_like(mask_events, dtype=NUMPY_LIB.int32) 
    inds[:] = 0
    ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, mask_events, selected_jets)
    inds[:] = 1
    ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, mask_events, selected_jets)
    jets.attrs_data["pass_dr"] = jets_pass_dr
    jets.attrs_data["selected"] = selected_jets
    jets.attrs_data["first_two"] = first_two_jets

    dijet_inv_mass = compute_inv_mass(jets, mask_events, selected_jets & first_two_jets)

    #Find the first two genjets in the event that are not matched to gen-leptons
    if is_mc:
        #find genleptons
        genpart_pdgid = NUMPY_LIB.abs(genParticle.pdgId)
        genpart_mask = (genpart_pdgid == 11)
        genpart_mask = NUMPY_LIB.logical_or(genpart_mask, (genpart_pdgid == 13))
        genpart_mask = NUMPY_LIB.logical_or(genpart_mask, (genpart_pdgid == 15))

        genjets_not_matched_genlepton = ha.mask_deltar_first(
            genJet, genJet.masks["all"], genParticle, genpart_mask, 0.3
        )
        out_genjet_mask = NUMPY_LIB.zeros(genJet.numobjects(), dtype=NUMPY_LIB.bool)
        inds = NUMPY_LIB.zeros_like(mask_events)
        targets = NUMPY_LIB.ones_like(mask_events)
        inds[:] = 0
        ha.set_in_offsets(out_genjet_mask, genJet.offsets, inds, targets, mask_events, genjets_not_matched_genlepton)
        inds[:] = 1
        ha.set_in_offsets(out_genjet_mask, genJet.offsets, inds, targets, mask_events, genjets_not_matched_genlepton)

        num_good_genjets = ha.sum_in_offsets(genJet, out_genjet_mask, mask_events, genJet.masks["all"], NUMPY_LIB.int8)

        #for iev in range(10):
        #    print("iev", iev)
        #    for igp in range(genParticle.offsets[iev], genParticle.offsets[iev+1]):
        #        if (genpart_mask[igp]):
        #            print("igp", igp, genParticle.pt[igp], genParticle.eta[igp], genParticle.phi[igp], genParticle.pdgId[igp])
        #    for igj in range(genJet.offsets[iev], genJet.offsets[iev+1]):
        #        print("igj", igj, genJet.pt[igj], genJet.eta[igj], genJet.phi[igj], out_genjet_mask[igj])

        genjet_inv_mass = compute_inv_mass(genJet, mask_events, out_genjet_mask)
    
    selected_jets_btag = selected_jets & (jets.btagDeepB >= jet_btag)

    num_jets = ha.sum_in_offsets(jets, selected_jets, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)

    num_jets_btag = ha.sum_in_offsets(jets, selected_jets_btag, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)

    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets")
            jaggedstruct_print(jets, idx,
                ["pt", "eta", "phi", "mass", "jetId", "puId",
                "pass_dr", "selected", 
                "first_two"])

    ret = {
        "selected_jets": selected_jets,
        "num_jets": num_jets,
        "num_jets_btag": num_jets_btag,
        "dijet_inv_mass": dijet_inv_mass
    }
    if is_mc:
        ret["dijet_inv_mass_gen"] = genjet_inv_mass
        ret["num_good_genjets"] = num_good_genjets

    return ret

def get_selected_electrons(electrons, pt_cut, eta_cut, id_type):
    if id_type == "mvaFall17V1Iso_WP90":
        passes_id = electrons.mvaFall17V1Iso_WP90 == 1
    elif id_type == "none":
        passes_id = NUMPY_LIB.ones(electrons.num_objects, dtype=NUMPY_LIB.bool)
    else:
        raise Exception("Unknown id_type {0}".format(id_type))
        
    passes_pt = electrons.pt > pt_cut
    passes_eta = NUMPY_LIB.abs(electrons.eta) < eta_cut
    final_electron_sel = passes_id & passes_pt & passes_eta

    return {
        "additional_electron_sel": final_electron_sel,
    }

def compute_inv_mass(objects, mask_events, mask_objects):
    """
    Computes the invariant mass in the selected objects.
    
    objects (JaggedStruct)
    mask_events (array of bool) 
    mask_objects (array of bool)
    
    """
    if objects.numobjects() != len(mask_objects):
        raise Exception(
            "Object mask size {0} did not match number of objects {1}".format(
                len(mask_objects), objects.numobjects()))
    if objects.numevents() != len(mask_events):
        raise Exception(
            "Event mask size {0} did not match number of events {1}".format(
                len(mask_events), objects.numevents()))

    pt = objects.pt
    eta = objects.eta
    phi = objects.phi
    mass = objects.mass

    px = pt * NUMPY_LIB.cos(phi)
    py = pt * NUMPY_LIB.sin(phi)
    pz = pt * NUMPY_LIB.sinh(eta)
    e = NUMPY_LIB.sqrt(px**2 + py**2 + pz**2 + mass**2)

    px_total = ha.sum_in_offsets(objects, px, mask_events, mask_objects)
    py_total = ha.sum_in_offsets(objects, py, mask_events, mask_objects)
    pz_total = ha.sum_in_offsets(objects, pz, mask_events, mask_objects)
    e_total = ha.sum_in_offsets(objects, e, mask_events, mask_objects)
    inv_mass = NUMPY_LIB.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
    return inv_mass

def fill_with_weights(values, weight_dict, mask, bins):
    ret = {
        wn: get_histogram(values[mask], weight_dict[wn][mask], bins)
        for wn in weight_dict.keys()
    }
    return ret

def update_histograms_systematic(hists, hist_name, systematic_name, values, weight_dict, mask, bins):
    if hist_name not in hists:
        hists[hist_name] = {}
    ret = fill_with_weights(values, weight_dict, mask, bins)
    if systematic_name[0] == "nominal":
        hists[hist_name].update(ret)
    else:
        if systematic_name[1] == "":
            syst_string = systematic_name[0]
        else:
            syst_string = systematic_name[0] + "__" + systematic_name[1]
        ret = {syst_string: ret["nominal"]}
        hists[hist_name].update(ret)

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0

def fix_large_weights(weights, maxw=10.0):
    weights[weights > maxw] = maxw
    weights[:] = weights[:] / NUMPY_LIB.mean(weights)

def compute_pu_weights(pu_corrections_target, weights, mc_nvtx, reco_nvtx):
    pu_edges, (values_nom, values_up, values_down) = pu_corrections_target

    src_pu_hist = get_histogram(mc_nvtx, weights, pu_edges)
    norm = sum(src_pu_hist.contents)
    values_target = src_pu_hist.contents/norm

    ratio = values_nom / values_target
    remove_inf_nan(ratio)
    pu_weights = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges),
        NUMPY_LIB.array(ratio), pu_weights)
    fix_large_weights(pu_weights) 
     
    ratio_up = values_up / values_target
    remove_inf_nan(ratio_up)
    pu_weights_up = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges),
        NUMPY_LIB.array(ratio_up), pu_weights_up)
    fix_large_weights(pu_weights_up) 
    
    ratio_down = values_down / values_target
    remove_inf_nan(ratio_down)
    pu_weights_down = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges),
        NUMPY_LIB.array(ratio_down), pu_weights_down)
    fix_large_weights(pu_weights_down) 
    
    return pu_weights, pu_weights_up, pu_weights_down

def select_events_trigger(scalars, parameters, mask_events, hlt_bits):
    flags = [
        "Flag_BadPFMuonFilter",
        "Flag_EcalDeadCellTriggerPrimitiveFilter",
        "Flag_HBHENoiseFilter",
        "Flag_HBHENoiseIsoFilter",
        "Flag_globalSuperTightHalo2016Filter",
        "Flag_goodVertices",
        "Flag_BadChargedCandidateFilter"
    ]
    for flag in flags:
        mask_events = mask_events & scalars[flag]
    
    pvsel = scalars["PV_npvsGood"] > parameters["nPV"]
    pvsel = pvsel & (scalars["PV_ndof"] > parameters["NdfPV"])
    pvsel = pvsel & (scalars["PV_z"] < parameters["zPV"])

    trig_decision = scalars[hlt_bits[0]]
    for hlt_bit in hlt_bits[1:]:
        trig_decision += scalars[hlt_bit]
    trig_decision = trig_decision >= 1
    mask_events = mask_events & trig_decision & pvsel
    return mask_events

def get_int_lumi(runs, lumis, mask_events, lumidata):
    processed_runs = NUMPY_LIB.asnumpy(runs[mask_events])
    processed_lumis = NUMPY_LIB.asnumpy(lumis[mask_events])
    runs_lumis = np.zeros((processed_runs.shape[0], 2), dtype=np.uint32)
    runs_lumis[:, 0] = processed_runs[:]
    runs_lumis[:, 1] = processed_lumis[:]
    lumi_proc = lumidata.get_lumi(runs_lumis)
    return lumi_proc

def get_gen_sumweights(filenames):
    sumw = 0
    sumw2 = 0
    for fi in filenames:
        ff = uproot.open(fi)
        bl = ff.get("Runs")
        arr = bl.array("genEventSumw")
        arr2 = bl.array("genEventSumw2")
        arr = arr
        sumw += arr.sum()
        sumw2 += arr2.sum()
    return sumw, sumw2

"""
Applies Rochester corrections on leading and subleading muons, returns the corrected pt

    is_mc: bool
    rochester_corrections: RochesterCorrections object that contains calibration data
    muons: JaggedStruct of all muon data

    returns: nothing

"""
def do_rochester_corrections(
    is_mc,
    rochester_corrections,
    muons):

    qterm = rochester_correction_muon_qterm(
        is_mc, rochester_corrections, muons)
    
    muon_pt_corr = muons.pt * qterm
    muons.pt[:] = muon_pt_corr[:]

    return

"""
Computes the Rochester correction q-term for an array of muons.

    is_mc: bool
    rochester_corrections: RochesterCorrections object that contains calibration data
    muons: JaggedStruct of all muon data

    returns: array of the computed q-term values
"""
def rochester_correction_muon_qterm(
    is_mc, rochester_corrections,
    muons):
    if is_mc:
        rnd = NUMPY_LIB.random.rand(len(muons.pt)).astype(NUMPY_LIB.float32)
        qterm = rochester_corrections.compute_kSpreadMC_or_kSmearMC(
            NUMPY_LIB.asnumpy(muons.pt),
            NUMPY_LIB.asnumpy(muons.eta),
            NUMPY_LIB.asnumpy(muons.phi),
            NUMPY_LIB.asnumpy(muons.charge),
            NUMPY_LIB.asnumpy(muons.genpt),
            NUMPY_LIB.asnumpy(muons.nTrackerLayers),
            NUMPY_LIB.asnumpy(rnd)
        )
    else:
        qterm = rochester_corrections.compute_kScaleDT(
            NUMPY_LIB.asnumpy(muons.pt),
            NUMPY_LIB.asnumpy(muons.eta),
            NUMPY_LIB.asnumpy(muons.phi),
            NUMPY_LIB.asnumpy(muons.charge),
        )

    return NUMPY_LIB.array(qterm)

# Custom kernels to get the pt of the muon based on the matched genPartIdx of the reco muon
# Implement them here as they are too specific to NanoAOD for the hepaccelerate library
@numba.njit(parallel=True, fastmath=True)
def get_genpt_cpu(reco_offsets, reco_genPartIdx, genparts_offsets, genparts_pt, out_reco_genpt):
    #loop over events
    for iev in numba.prange(len(reco_offsets) - 1):
        #loop over muons
        for imu in range(reco_offsets[iev], reco_offsets[iev + 1]):
            #get index of genparticle that reco particle was matched to
            idx_gp = reco_genPartIdx[imu]
            if idx_gp >= 0:
                genpt = genparts_pt[genparts_offsets[iev] + idx_gp]
                out_reco_genpt[imu] = genpt

@cuda.jit
def get_genpt_cuda(reco_offsets, reco_genPartIdx, genparts_offsets, genparts_pt, out_reco_genpt):
    #loop over events
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, len(reco_offsets) - 1, xstride):
        #loop over muons
        for imu in range(reco_offsets[iev], reco_offsets[iev + 1]):
            #get index of genparticle that reco particle was matched to
            idx_gp = reco_genPartIdx[imu]
            if idx_gp >= 0:
                genpt = genparts_pt[genparts_offsets[iev] + idx_gp]
                out_reco_genpt[imu] = genpt

@numba.njit(parallel=True, fastmath=True)
def get_matched_genparticles(reco_offsets, reco_genPartIdx, mask_reco, genparts_offsets, out_genparts_mask):
    #loop over events
    for iev in numba.prange(len(reco_offsets) - 1):
        #loop over reco objects
        for iobj in range(reco_offsets[iev], reco_offsets[iev + 1]):
            if not mask_reco[iobj]:
                continue
            #get index of genparticle that muon was matched to
            idx_gp_ev = reco_genPartIdx[iobj]
            if idx_gp_ev >= 0:
                idx_gp = int(genparts_offsets[iev]) + int(idx_gp_ev)
                out_genparts_mask[idx_gp] = True

@cuda.jit
def get_matched_genparticles_kernel(reco_offsets, reco_genPartIdx, mask_reco, genparts_offsets, out_genparts_mask):
    #loop over events
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, len(reco_offsets) - 1, xstride):
        #loop over reco objects
        for iobj in range(reco_offsets[iev], reco_offsets[iev + 1]):
            if not mask_reco[iobj]:
                continue
            #get index of genparticle that muon was matched to
            idx_gp_ev = reco_genPartIdx[iobj]
            if idx_gp_ev >= 0:
                idx_gp = int(genparts_offsets[iev]) + int(idx_gp_ev)
                out_genparts_mask[idx_gp] = True


def to_cartesian(arrs):
    pt = arrs["pt"]
    eta = arrs["eta"]
    phi = arrs["phi"]
    mass = arrs["mass"]
    px = pt * NUMPY_LIB.cos(phi)
    py = pt * NUMPY_LIB.sin(phi)
    pz = pt * NUMPY_LIB.sinh(eta)
    e = NUMPY_LIB.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return {"px": px, "py": py, "pz": pz, "e": e}

"""
Given a a dictionary of arrays of cartesian coordinates (px, py, pz, e),
computes the array of spherical coordinates (pt, eta, phi, m)

    arrs: dict of str -> array
    returns: dict of str -> array
"""
def to_spherical(arrs):
    px = arrs["px"]
    py = arrs["py"]
    pz = arrs["pz"]
    e = arrs["e"]
    pt = NUMPY_LIB.sqrt(px**2 + py**2)
    eta = NUMPY_LIB.arcsinh(pz / pt)
    phi = NUMPY_LIB.arccos(px / pt)
    mass = NUMPY_LIB.sqrt(e**2 - (px**2 + py**2 + pz**2))
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass}

"""
Given two objects, computes the dr = sqrt(deta^2+dphi^2) between them.
    obj1: array of spherical coordinates (pt, eta, phi, m) for the first object
    obj2: array of spherical coordinates for the second object

    returns: arrays of deta, dphi, dr
"""
def deltar(obj1, obj2):
    deta = obj1["eta"] - obj2["eta"] 
    dphi = obj1["phi"] - obj2["phi"]

    dr = NUMPY_LIB.sqrt(deta**2 + dphi**2)
    
    return deta, dphi, dr 

"""
Fills the DNN input variables based on two muons and two jets.
    leading_muon: spherical coordinate data of the leading muon
    subleading_muon: spherical coordinate data of the subleading muon
    leading_jet: spherical coordinate data + QGL of the leading jet
    subleading_jet: spherical coordinate data + QGL of the subleading jet
    nsoft: number of soft jets
    
    'softJet5' - # of soft EWK jets with pt > 5 GeV
    'dRmm' - Delta R between two muons
    'dEtamm' - Delta eta between two muons
    'dPhimm' - Delta Phi between two muons
    'M_jj' - dijet mass
    'pt_jj' - dijet pt
    'eta_jj' - dijet eta
    'phi_jj' - dijet phi
    'M_mmjj' - mass of dimuon + dijet system
    'eta_mmjj' - eta of dimuon + dijet system
    'phi_mmjj' - phi of dimuon + dijet system
    'dEta_jj' - delta eta between two jets
    'Zep' - zeppenfeld variable
    'dRmin_mj' - Min delta R between a muon and jet
    'dRmax_mj' - Max delta R between a muon and jet
    'dRmin_mmj' - Min delta R between dimuon and jet
    'dRmax_mmj' - Max delta R between dimuon and jet
    'leadingJet_pt' - Leading jet pt
    'subleadingJet_pt' - sub-leading jet pt 
    'leadingJet_eta' - leading jet eta
    'subleadingJet_eta' - sub-leading jet eta
    'leadingJet_qgl' - leading jet qgl
    'subleadingJet_qgl' - sub - leading jet qgl
    'cthetaCS' - cosine of collins Soper frame angle
    'Higgs_pt' - dimuon pt
    'Higgs_eta' - dimuon eta
"""
def dnn_variables(leading_muon, subleading_muon, leading_jet, subleading_jet, nsoft):
    #delta eta, phi and R between two muons
    mm_deta, mm_dphi, mm_dr = deltar(leading_muon, subleading_muon)
    
    #delta eta between jets 
    jj_deta = leading_jet["eta"] - subleading_jet["eta"]

    #muons in cartesian, create dimuon system 
    m1 = to_cartesian(leading_muon)    
    m2 = to_cartesian(subleading_muon)    
    mm = {k: m1[k] + m2[k] for k in ["px", "py", "pz", "e"]}
    mm_sph = to_spherical(mm)

    #jets in cartesian, create dimuon system 
    j1 = to_cartesian(leading_jet)
    j2 = to_cartesian(subleading_jet)
    jj = {k: j1[k] + j2[k] for k in ["px", "py", "pz", "e"]}
    jj_sph = to_spherical(jj)
  
    #create dimuon-dijet system 
    mmjj = {k: j1[k] + j2[k] + m1[k] + m2[k] for k in ["px", "py", "pz", "e"]} 
    mmjj_sph = to_spherical(mmjj)

    #compute deltaR between all muons and jets
    dr_mjs = []
    for mu in [leading_muon, subleading_muon]:
        for jet in [leading_jet, subleading_jet]:
            _, _, dr_mj = deltar(mu, jet)
            dr_mjs += [dr_mj]
    dr_mj = NUMPY_LIB.vstack(dr_mjs)
    dRmin_mj = NUMPY_LIB.min(dr_mj, axis=0) 
    dRmax_mj = NUMPY_LIB.max(dr_mj, axis=0) 
    
    #compute deltaR between dimuon system and both jets 
    dr_mmjs = []
    for jet in [leading_jet, subleading_jet]:
        _, _, dr_mmj = deltar(mm_sph, jet)
        dr_mmjs += [dr_mmj]
    dr_mmj = NUMPY_LIB.vstack(dr_mmjs)
    dRmin_mmj = NUMPY_LIB.min(dr_mmj, axis=0) 
    dRmax_mmj = NUMPY_LIB.max(dr_mmj, axis=0)

    #Zeppenfeld variable
    Zep = (mm_sph["eta"] - 0.5*(leading_jet["eta"] + subleading_jet["eta"]))

    #Collin-Soper frame variable
    cthetaCS = 2*(m1["pz"] * m2["e"] - m1["e"]*m2["pz"]) / (mm_sph["mass"] * NUMPY_LIB.sqrt(NUMPY_LIB.power(mm_sph["mass"], 2) + NUMPY_LIB.power(mm_sph["pt"], 2)))

    ret = {
        "dEtamm": mm_deta, "dPhimm": mm_dphi, "dRmm": mm_dr,
        "M_jj": jj_sph["mass"], "pt_jj": jj_sph["pt"], "eta_jj": jj_sph["eta"], "phi_jj": jj_sph["phi"],
        "M_mmjj": mmjj_sph["mass"], "eta_mmjj": mmjj_sph["eta"], "phi_mmjj": mmjj_sph["phi"],
        "dEta_jj": jj_deta,
        "leadingJet_pt": leading_jet["pt"],
        "subleadingJet_pt": subleading_jet["pt"],
        "leadingJet_eta": leading_jet["eta"],
        "subleadingJet_eta": subleading_jet["eta"],
        "dRmin_mj": dRmin_mj,
        "dRmax_mj": dRmax_mj,
        "dRmin_mmj": dRmin_mmj,
        "dRmax_mmj": dRmax_mmj,
        "Zep": Zep,
        "leadingJet_qgl": leading_jet["qgl"],
        "subleadingJet_qgl": subleading_jet["qgl"], 
        "cthetaCS": cthetaCS,
        "softJet5": nsoft,
        "Higgs_pt": mm_sph["pt"],
        "Higgs_eta": mm_sph["eta"],
    }

    if debug:
        for k in ret.keys():
            msk = NUMPY_LIB.isnan(ret[k])
            if NUMPY_LIB.sum(msk) > 0:
                print("dnn vars nan", k, np.sum(msk))

    return ret

"""
Given an dictionary with arrays and a mask, applies the mask on all arrays
    arr_dict: dict of key -> array for input data
    mask: mask with the same length as the arrays
"""
def apply_mask(arr_dict, mask):
    return {k: v[mask] for k, v in arr_dict.items()}

def select_weights(weights, jet_systematic_scenario):
    if jet_systematic_scenario[0] == "nominal":
        return weights
    else:
        return {"nominal": weights["nominal"]}

# 1. Compute the DNN input variables in a given preselection
# 2. Evaluate the DNN model
# 3. Fill histograms with DNN inputs and output
def compute_fill_dnn(parameters, use_cuda, dnn_presel, dnn_model, scalars, leading_muon, subleading_muon, leading_jet, subleading_jet, weights, hists):
    nev_dnn_presel = NUMPY_LIB.sum(dnn_presel)

    #for some reason, on the cuda backend, the sum does not return a simple number
    if use_cuda:
        nev_dnn_presel = int(NUMPY_LIB.asnumpy(nev_dnn_presel).flatten()[0])

    leading_muon_s = apply_mask(leading_muon, dnn_presel)
    subleading_muon_s = apply_mask(subleading_muon, dnn_presel)
    leading_jet_s = apply_mask(leading_jet, dnn_presel)
    subleading_jet_s = apply_mask(subleading_jet, dnn_presel)
    nsoft = scalars["SoftActivityJetNjets5"][dnn_presel]
    dnn_vars = dnn_variables(leading_muon_s, subleading_muon_s, leading_jet_s, subleading_jet_s, nsoft)
    if (not (dnn_model is None)) and nev_dnn_presel > 0:
        dnn_vars_arr = NUMPY_LIB.vstack([dnn_vars[k] for k in parameters["dnn_varlist_order"]]).T
        #for TF, need to convert library to numpy, as it doesn't accept cupy arrays
        dnn_pred = NUMPY_LIB.array(dnn_model.predict(NUMPY_LIB.asnumpy(dnn_vars_arr), batch_size=10000)[:, 0])
    else:
        dnn_pred = NUMPY_LIB.zeros(nev_dnn_presel)

    #Fill the histograms with DNN inputs
    dnn_mask = NUMPY_LIB.ones(nev_dnn_presel, dtype=NUMPY_LIB.bool)
    weights_dnn = {k: w[dnn_presel] for k, w in weights.items()}
    hists["hist__dnn_presel__dnn_pred"] = fill_with_weights(
       dnn_pred, weights_dnn, dnn_mask,
       NUMPY_LIB.linspace(0.0, 1.0, 30)
    )

    for vn in parameters["dnn_varlist_order"]:
        subarr = dnn_vars[vn]
        hb = parameters["dnn_input_histogram_bins"][vn]
        hists["hist__dnn_presel__{0}".format(vn)] = fill_with_weights(
           subarr, weights_dnn, dnn_mask,
           NUMPY_LIB.linspace(*hb)
        )
    return dnn_vars, dnn_pred, weights_dnn

def get_jer_smearfactors(pt_or_m, ratio_jet_genjet, msk_no_genjet, msk_poor_reso, resos, resosfs):
    
    #smearing for matched jets
    smear_matched_n = 1.0 + (resosfs[:, 0] - 1.0) * ratio_jet_genjet
    smear_matched_u = 1.0 + (resosfs[:, 1] - 1.0) * ratio_jet_genjet
    smear_matched_d = 1.0 + (resosfs[:, 2] - 1.0) * ratio_jet_genjet

    #compute random smearing for unmatched jets
    sigma_unmatched_n = resos * NUMPY_LIB.sqrt(NUMPY_LIB.clip(resosfs[:, 0]**2 - 1.0, 0, 100))
    sigma_unmatched_u = resos * NUMPY_LIB.sqrt(NUMPY_LIB.clip(resosfs[:, 1]**2 - 1.0, 0, 100))
    sigma_unmatched_d = resos * NUMPY_LIB.sqrt(NUMPY_LIB.clip(resosfs[:, 2]**2 - 1.0, 0, 100))

    zeros = NUMPY_LIB.ones_like(sigma_unmatched_n)
    rand = NUMPY_LIB.random.normal(loc=zeros, scale=resos, size=len(zeros))
    
    smear_rnd_n = 1. + rand * NUMPY_LIB.sqrt(resosfs[:, 0]**2 - 1.)
    smear_rnd_u = 1. + rand * NUMPY_LIB.sqrt(resosfs[:, 1]**2 - 1.)
    smear_rnd_d = 1. + rand * NUMPY_LIB.sqrt(resosfs[:, 2]**2 - 1.)

    inds_no_genjet = NUMPY_LIB.nonzero(msk_no_genjet)[0]

    smear_n = NUMPY_LIB.array(smear_matched_n)
    smear_u = NUMPY_LIB.array(smear_matched_u)
    smear_d = NUMPY_LIB.array(smear_matched_d)

    #for jets that have no matched genjet, use random smearing
    ha.copyto_dst_indices(smear_n, smear_rnd_n[msk_no_genjet], inds_no_genjet)
    ha.copyto_dst_indices(smear_u, smear_rnd_u[msk_no_genjet], inds_no_genjet)
    ha.copyto_dst_indices(smear_d, smear_rnd_d[msk_no_genjet], inds_no_genjet)

    smear_n[msk_no_genjet & (resosfs[:, 0]<1.0)] = 1
    smear_u[msk_no_genjet & (resosfs[:, 1]<1.0)] = 1
    smear_d[msk_no_genjet & (resosfs[:, 2]<1.0)] = 1

    smear_n[(smear_n * pt_or_m) < 0.01] = 0.01
    smear_u[(smear_u * pt_or_m) < 0.01] = 0.01
    smear_d[(smear_d * pt_or_m) < 0.01] = 0.01

    return smear_n, smear_u, smear_d, sigma_unmatched_n

def apply_jec(jets, scalars, parameters, jetmet_corrections, NUMPY_LIB, use_cuda, is_mc):
    # Change rho from a per-event variable to per-jet by broadcasting the
    # value of each event to all the jets in the event
    jets_rho = NUMPY_LIB.zeros_like(jets.pt)
    ha.broadcast(scalars["fixedGridRhoFastjetAll"], jets.offsets, jets_rho)
    
    # Get the uncorrected jet pt and mass
    raw_pt = jets.pt * (1.0 - jets.rawFactor)
    raw_mass = jets.mass * (1.0 - jets.rawFactor) 

    # Need to use the CPU for JEC/JER currently
    if use_cuda:
        raw_pt = NUMPY_LIB.asnumpy(raw_pt)
        eta = NUMPY_LIB.asnumpy(jets.eta)
        rho = NUMPY_LIB.asnumpy(jets_rho)
        area = NUMPY_LIB.asnumpy(jets.area)
    else:
        raw_pt = raw_pt
        eta = jets.eta
        rho = jets_rho
        area = jets.area

    #dictionary of jet systematic scenario name -> jet pt vector
    #the data vectors can be large, so it is possible to use a lambda function instead
    #which upon calling will return a jet pt vector
    jet_pt_syst = {
        ("raw_jets", ""): NUMPY_LIB.array(raw_pt),
        ("nominal", ""): jets.pt,
    }

    #Re-apply jet energy corrections and compute JEC & JER uncertainties
    if parameters["do_jec"]:
        #compute and apply jet corrections
        if is_mc:
            corr = jetmet_corrections.jec_mc.getCorrection(JetPt=raw_pt.copy(), Rho=rho, JetEta=eta, JetA=area)
        else:
            final_corr = NUMPY_LIB.zeros_like(jets.pt)

            #final correction is run-dependent, compute that for each run separately
            for run_idx in NUMPY_LIB.unique(scalars["run_index"]):
                
                if use_cuda:
                    run_idx = int(run_idx)
                msk = scalars["run_index"] == run_idx
                
                #find the jets in the events that pass this run index cut
                jets_msk = NUMPY_LIB.zeros(jets.numobjects(), dtype=NUMPY_LIB.bool)
                ha.broadcast(msk, jets.offsets, jets_msk)
                inds_nonzero = NUMPY_LIB.nonzero(jets_msk)[0]

                #Evaluate jet correction (on CPU only currently)
                if use_cuda:
                    jets_msk = NUMPY_LIB.asnumpy(jets_msk)
                run_name = runmap_numerical_r[run_idx]
                corr = jetmet_corrections.jec_data[run_name].getCorrection(
                    JetPt=raw_pt[jets_msk].copy(), Rho=rho[jets_msk], JetEta=eta[jets_msk], JetA=area[jets_msk])
                if debug:
                    print("run_idx=", run_idx, corr.mean(), corr.std())

                #update the final jet correction for the jets in the events in this run
                if len(inds_nonzero) > 0:
                    ha.copyto_dst_indices(final_corr, corr, inds_nonzero)
            corr = final_corr
	
        corr = NUMPY_LIB.array(corr)
        pt_jec = NUMPY_LIB.array(raw_pt) * corr 
        mass_jec = raw_mass * corr
        jet_pt_syst[("nominal", "")] = pt_jec
        
        #JER and JEC uncertainty
        if is_mc:
            if jetmet_corrections.jer:

                #This is done only on CPU
                resos = jetmet_corrections.jer.getResolution(
                    JetEta=eta, Rho=rho, JetPt=NUMPY_LIB.asnumpy(pt_jec))
                resosfs = jetmet_corrections.jersf.getScaleFactor(JetEta=eta)

                #The following is done either on CPU or GPU
                resos = NUMPY_LIB.array(resos)
                resosfs = NUMPY_LIB.array(resosfs)

                dpt_jet_genjet = jets.pt - jets.genpt
                dpt_jet_genjet[jets.genpt == 0] = 0
                ratio_jet_genjet_pt = dpt_jet_genjet / jets.pt

                msk_no_genjet = ratio_jet_genjet_pt == 0
                msk_poor_reso = resosfs[:, 0] < 1

                dm_jet_genjet = jets.mass - jets.genmass
                dm_jet_genjet[jets.genmass == 0] = 0
                ratio_jet_genjet_mass = dm_jet_genjet / jets.mass
               
                smear_n, smear_u, smear_d, sigma_unmatched_n = get_jer_smearfactors(
                    jets.pt, ratio_jet_genjet_pt, msk_no_genjet, msk_poor_reso, resos, resosfs)
                # mass_smear_n, mass_smear_u, mass_smear_d = get_jer_smearfactors(
                #     jets.mass, ratio_jet_genjet_mass, msk_no_genjet, msk_poor_reso, resosfs)

                pt_jec_jer = pt_jec * smear_n
                pt_jec_jer_up = pt_jec * smear_u
                pt_jec_jer_down = pt_jec * smear_d

                # mass_jec_jer_central = mass_jec * mass_smear_n
                # mass_jec_jer_up = mass_jec * mass_smear_u
                # mass_jec_jer_down = mass_jec * mass_smear_d

            else:
                pt_jec_jer = pt_jec
                pt_jec_jer_up = pt_jec
                pt_jec_jer_down = pt_jec

                # mass_jec_jer = mass_jec
                # mass_jec_jer_up = mass_jec
                # mass_jec_jer_down = mass_jec

            jet_pt_syst[("nominal", "")] = pt_jec_jer
            jet_pt_syst[("nanoaod", "")] = NUMPY_LIB.array(jets.pt)
            jet_pt_syst[("jec", "")] = pt_jec

            jesunc = dict(list(jetmet_corrections.jesunc.getUncertainty(JetPt=NUMPY_LIB.array(pt_jec_jer), JetEta=jets.eta))) 

            jet_pt_syst[("jer", "up")] = pt_jec_jer_up
            jet_pt_syst[("jer", "down")] = pt_jec_jer_down

            #add variated jet momenta
            for unc_name, arr in jesunc.items():
                jet_pt_syst[(unc_name, "up")] = pt_jec_jer * NUMPY_LIB.array(jesunc[unc_name][:, 0])
                jet_pt_syst[(unc_name, "down")] = pt_jec_jer * NUMPY_LIB.array(jesunc[ unc_name][:, 1]) 

    if debug and is_mc:
        print("variated jet pt")
        for unc_name in jet_pt_syst.keys():
            arr = jet_pt_syst[unc_name]
            diff = arr - jets.pt
            print("wrt NanoAOD", unc_name, diff.mean(), diff.std())
        
    #dictionary of variated jet momenta
    return jet_pt_syst

def fill_muon_hists(hists, scalars, weights, ret_mu, inv_mass,
    leading_muon, subleading_muon, parameters,
    masswindow_110_150, masswindow_120_130, NUMPY_LIB):

    hists["hist__dimuon__inv_mass"] = fill_with_weights(
        inv_mass, weights,
        ret_mu["selected_events"], NUMPY_LIB.linspace(50, 200, 101)
    )

    #get histograms of leading and subleading muon momenta
    hists["hist__dimuon__leading_muon_pt"] = fill_with_weights(leading_muon["pt"], weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 101))
    hists["hist__dimuon__subleading_muon_pt"] = fill_with_weights(subleading_muon["pt"], weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 101))

    hists["hist__dimuon__leading_muon_eta"] = fill_with_weights(
        leading_muon["eta"], weights, ret_mu["selected_events"],
        NUMPY_LIB.linspace(-4.0, 4.0, 101)
    )
    hists["hist__dimuon__subleading_muon_eta"] = fill_with_weights(
        subleading_muon["eta"], weights, ret_mu["selected_events"],
        NUMPY_LIB.linspace(-4.0, 4.0, 101)
    )

    hists["hist__dimuon_invmass_110_150__inv_mass"] = fill_with_weights(
       inv_mass, weights, ret_mu["selected_events"] & masswindow_110_150,
       NUMPY_LIB.linspace(110, 150, parameters["inv_mass_bins"])
    )

    hists["hist__dimuon_invmass_120_130__inv_mass"] = fill_with_weights(
       inv_mass, weights, ret_mu["selected_events"] & masswindow_120_130,
       NUMPY_LIB.linspace(120, 130, parameters["inv_mass_bins"])
    )

def compute_lepton_sf(leading_muon, subleading_muon, lepsf_iso, lepsf_id, lepsf_trig, use_cuda, dataset_era, NUMPY_LIB, debug):
    sfs = []

    for mu in [leading_muon, subleading_muon]: 
        if use_cuda:
            mu = {k: NUMPY_LIB.asnumpy(v) for k, v in mu.items()}
        pdgid = numpy.array(mu["pdgId"])
        
        #In 2016, the histograms are flipped
        if dataset_era == "2016":
            pdgid[:] = 11

        sf_iso = lepsf_iso.compute(pdgid, mu["pt"], mu["eta"])
        sf_id = lepsf_id.compute(pdgid, mu["pt"], mu["eta"])
        sf_trig = lepsf_trig.compute(pdgid, mu["pt"], mu["eta"])
        if debug:
            print("sf_iso: ", sf_iso.mean(), "+-", sf_iso.std())
            print("sf_id: ", sf_id.mean(), "+-", sf_id.std())
            print("sf_trig: ", sf_id.mean(), "+-", sf_trig.std())
        sfs += [sf_iso, sf_id, sf_trig]

    #multiply all weights
    sf_tot = sfs[0]
    for sf in sfs[1:]:
        sf_tot = sf_tot * sf
    
    if debug:
        print("sf_tot: ", sf_tot.mean(), "+-", sf_tot.std())

    #move to GPU
    if use_cuda:
        sf_tot = NUMPY_LIB.array(sf_tot)

    return sf_tot

def jaggedstruct_print(struct, idx, attrs):
    of1 = struct.offsets[idx]
    of2 = struct.offsets[idx+1]
    print("nstruct", of2-of1)
    for i in range(of1, of2):
        print("s", [getattr(struct, a)[i] for a in attrs])

def deepdive_event(scalars, mask_events, ret_mu, jets, muons, id):
    print("deepdive")
    idx = np.where(scalars["event"]==id)[0][0]
    print("scalars:", {k: v[idx] for k, v in scalars.items()})
    print("trigger:", mask_events[idx])
    print("muon:", ret_mu["selected_events"][idx])
    jaggedstruct_print(jets, idx, ["pt", "eta"])
    jaggedstruct_print(muons, idx, ["pt", "eta", "mediumId", "pfRelIso04_all", "charge", "triggermatch", "passes_leading_pt", "pass_id", "pass_os"])

def sync_printout(
    ret_mu, muons, scalars,
    leading_muon, subleading_muon, inv_mass,
    n_additional_muons, n_additional_electrons,
    ret_jet, leading_jet, subleading_jet):
    with open("log_sync.txt", "w") as fi:
        msk = ret_mu["selected_events"] & (
            NUMPY_LIB.logical_or(
                (inv_mass > 110.0) & (inv_mass < 150.0),
                (inv_mass > 76.0) & (inv_mass < 106.0)
            )
        )
        for iev in range(muons.numevents()):
            if msk[iev]:
                s = ""
                s += "{0} ".format(scalars["run"][iev])
                s += "{0} ".format(scalars["luminosityBlock"][iev])
                s += "{0} ".format(scalars["event"][iev])
                s += "{0} ".format(ret_jet["num_jets"][iev])
                s += "{0} ".format(ret_jet["num_jets_btag"][iev])
                s += "{0} ".format(n_additional_muons[iev])
                s += "{0} ".format(n_additional_electrons[iev])
                s += "{0:.2f} ".format(leading_muon["pt"][iev])
                s += "{0:.2f} ".format(leading_muon["eta"][iev])
                s += "{0:.2f} ".format(subleading_muon["pt"][iev])
                s += "{0:.2f} ".format(subleading_muon["eta"][iev])
                s += "{0:.2f} ".format(inv_mass[iev])

                s += "{0:.2f} ".format(leading_jet["pt"][iev])
                s += "{0:.2f} ".format(leading_jet["eta"][iev])
                s += "{0:.2f} ".format(subleading_jet["pt"][iev])
                s += "{0:.2f} ".format(subleading_jet["eta"][iev])

                s += "{0:.2f} ".format(ret_jet["dijet_inv_mass"][iev])

                #category index
                s += "{0} ".format(scalars["category"][iev])
                print(s, file=fi)

def assign_category_nan_irene(
    njet, nbjet, n_additional_muons, n_additional_electrons,
    dijet_inv_mass, leading_jet, subleading_jet, cat5_dijet_inv_mass):
    cats = NUMPY_LIB.zeros_like(njet)
    cats[:] = -9999

    msk_prev = NUMPY_LIB.zeros_like(cats, dtype=NUMPY_LIB.bool)

    #cat 1, ttH
    msk_1 = (nbjet > 0) & NUMPY_LIB.logical_or(n_additional_muons > 0, n_additional_electrons > 0)
    cats[NUMPY_LIB.invert(msk_prev) & msk_1] = 1
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_1)

    #cat 2
    msk_2 = (nbjet > 0) & (njet > 1)
    cats[NUMPY_LIB.invert(msk_prev) & msk_2] = 2
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_2)

    #cat 3
    msk_3 = (n_additional_muons > 0)
    cats[NUMPY_LIB.invert(msk_prev) & msk_3] = 3
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_3)

    #cat 4
    msk_4 = (n_additional_electrons > 0)
    cats[NUMPY_LIB.invert(msk_prev) & msk_4] = 4
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_4)

    #cat 5
    msk_5 = (dijet_inv_mass > cat5_dijet_inv_mass)
    cats[NUMPY_LIB.invert(msk_prev) & msk_5] = 5
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_5)

    #cat 6
    msk_6 = (dijet_inv_mass > 70) & (dijet_inv_mass < 100)
    cats[NUMPY_LIB.invert(msk_prev) & msk_6] = 6
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_6)

    #cat 7
    msk_7 = (njet > 1)
    cats[NUMPY_LIB.invert(msk_prev) & msk_7] = 7
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_7)

    cats[NUMPY_LIB.invert(msk_prev)] = 8

    return cats

def check_and_fix_qgl(jets):
    msk = NUMPY_LIB.isnan(jets["qgl"])
    jets["qgl"][msk] = -1
    if debug:
        if NUMPY_LIB.sum(msk) > 0:
            print("jets with qgl = NaN")
            print("pt", jets["pt"][msk])
            print("eta", jets["eta"][msk])
            print("puId", jets["puId"][msk])

def get_numev_passed(nev, masks):
    out = Results({})
    out["all"] = nev
    for name, mask in masks.items():
        out[name] = float(NUMPY_LIB.sum(mask))
    return out
 
def load_puhist_target(filename):
    fi = uproot.open(filename)
    
    h = fi["pileup"]
    edges = np.array(h.edges)
    values_nominal = np.array(h.values)
    values_nominal = values_nominal / np.sum(values_nominal)
    
    h = fi["pileup_plus"]
    values_up = np.array(h.values)
    values_up = values_up / np.sum(values_up)
    
    h = fi["pileup_minus"]
    values_down = np.array(h.values)
    values_down = values_down / np.sum(values_down)
    return edges, (values_nominal, values_up, values_down)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cache_data(filenames, name, datastructures, cache_location, datapath, is_mc, hlt_bits, nworkers=16):
    if nworkers == 1:
        tot_ev = 0
        tot_mb = 0
        for result in map(cache_data_multiproc_worker, [(name, fn, datastructures, cache_location, datapath, is_mc, hlt_bits) for fn in filenames]):
            tot_ev += result[0]
            tot_mb += result[1]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
            tot_ev = 0
            tot_mb = 0
            for result in executor.map(cache_data_multiproc_worker, [(name, fn, datastructures, cache_location, datapath, is_mc, hlt_bits) for fn in filenames]):
                tot_ev += result[0]
                tot_mb += result[1]
    return tot_ev, tot_mb

"""Given a ROOT file, run any checks that can only be done
on the original file. In our case, we need to access the number
of generated events.
"""
def func_filename_precompute_mc(filename):
    sumw, sumw2 = get_gen_sumweights([filename])
    ret = {"genEventSumw": sumw, "genEventSumw2": sumw2}
    return ret
 
def create_dataset(name, filenames, datastructures, cache_location, datapath, is_mc):
    ds = Dataset(name, filenames, datastructures, cache_location=cache_location, datapath=datapath, treename="Events")
    if is_mc:
        ds.func_filename_precompute = func_filename_precompute_mc
    return ds

def cache_preselection(ds, hlt_bits):
    for ifile in range(len(ds.filenames)):

        #OR of the trigger bits by summing
        hlt_res = [ds.eventvars[ifile][hlt_bit]==1 for hlt_bit in hlt_bits]
        sel = NUMPY_LIB.stack(hlt_res).sum(axis=0) >= 1

        #If we didn't have >=2 muons in NanoAOD, no need to keep this event 
        sel = sel & (ds.eventvars[ifile]["nMuon"] >= 2)

        for structname in ds.structs.keys():
            struct_compact = ds.structs[structname][ifile].compact_struct(sel)
            ds.structs[structname][ifile] = struct_compact
        for evvar_name in ds.eventvars[ifile].keys():
            ds.eventvars[ifile][evvar_name] = ds.eventvars[ifile][evvar_name][sel]

def cache_data_multiproc_worker(args):
    name, filename, datastructure, cache_location, datapath, is_mc, hlt_bits = args
    t0 = time.time()
    ds = create_dataset(name, [filename], datastructure, cache_location, datapath, is_mc)
    ds.numpy_lib = np

    #Skip loading this file if cache already done
    if ds.check_cache():
        return 0, 0

    ds.load_root()

    #put any preselection here
    processed_size_mb = ds.memsize()/1024.0/1024.0
    cache_preselection(ds, hlt_bits)
    processed_size_mb_post = ds.memsize()/1024.0/1024.0

    ds.to_cache()
    t1 = time.time()
    dt = t1 - t0
    print("built cache for {0}, loaded {1:.2f} MB, cached {2:.2f} MB, {3:.2E} Hz, {4:.2f} MB/s".format(
        filename, processed_size_mb, processed_size_mb_post, len(ds)/dt, processed_size_mb/dt))
    return len(ds), processed_size_mb

#Branches to load from the ROOT files
def create_datastructure(is_mc, dataset_era):
    datastructures = {
        "Muon": [
            ("Muon_pt", "float32"), ("Muon_eta", "float32"),
            ("Muon_phi", "float32"), ("Muon_mass", "float32"),
            ("Muon_pdgId", "int32"),
            ("Muon_pfRelIso04_all", "float32"), ("Muon_mediumId", "bool"),
            ("Muon_tightId", "bool"), ("Muon_charge", "int32"),
            ("Muon_isGlobal", "bool"), ("Muon_isTracker", "bool"),
            ("Muon_nTrackerLayers", "int32"),
        ],
        "Electron": [
            ("Electron_pt", "float32"), ("Electron_eta", "float32"),
            ("Electron_phi", "float32"), ("Electron_mass", "float32"),
            ("Electron_pfRelIso03_all", "float32"),
            ("Electron_mvaFall17V1Iso_WP90", "bool"),
        ],
        "Jet": [
            ("Jet_pt", "float32"),
            ("Jet_eta", "float32"),
            ("Jet_phi", "float32"),
            ("Jet_mass", "float32"),
            ("Jet_btagDeepB", "float32"),
            ("Jet_qgl", "float32"),
            ("Jet_jetId", "int32"),
            ("Jet_puId", "int32"),
            ("Jet_area", "float32"),
            ("Jet_rawFactor", "float32")
        ],
        "TrigObj": [
            ("TrigObj_pt", "float32"),
            ("TrigObj_eta", "float32"),
            ("TrigObj_phi", "float32"),
            ("TrigObj_id", "int32")
        ],
        "EventVariables": [
            ("nMuon", "int32"),
            ("PV_npvsGood", "float32"), 
            ("PV_ndof", "float32"),
            ("PV_z", "float32"),
            ("Flag_BadChargedCandidateFilter", "bool"),
            ("Flag_HBHENoiseFilter", "bool"),
            ("Flag_HBHENoiseIsoFilter", "bool"),
            ("Flag_EcalDeadCellTriggerPrimitiveFilter", "bool"),
            ("Flag_goodVertices", "bool"),
            ("Flag_globalSuperTightHalo2016Filter", "bool"),
            ("Flag_BadPFMuonFilter", "bool"),
            ("Flag_BadChargedCandidateFilter", "bool"),
            ("run", "uint32"),
            ("luminosityBlock", "uint32"),
            ("event", "uint64"),
            ("SoftActivityJetNjets5", "int32"),
            ("fixedGridRhoFastjetAll", "float32")
        ],
    }

    if is_mc:
        datastructures["EventVariables"] += [
            ("Pileup_nTrueInt", "uint32"),
            ("Generator_weight", "float32"),
            ("genWeight", "float32")
        ]
        datastructures["Muon"] += [
            ("Muon_genPartIdx", "int32"),
        ]
        datastructures["GenPart"] = [
            ("GenPart_pt", "float32"),
            ("GenPart_eta", "float32"),
            ("GenPart_phi", "float32"),
            ("GenPart_pdgId", "int32"),
        ]
        datastructures["Jet"] += [
            ("Jet_genJetIdx", "int32")
        ]
        datastructures["GenJet"] = [
            ("GenJet_pt", "float32"), 
            ("GenJet_eta", "float32"), 
            ("GenJet_phi", "float32"), 
            ("GenJet_mass", "float32"), 
        ]

    if dataset_era == "2016":
        datastructures["EventVariables"] += [
            ("HLT_IsoMu24", "bool"),
            ("HLT_IsoTkMu24", "bool"),
        ]
    elif dataset_era == "2017":
        datastructures["EventVariables"] += [
            ("HLT_IsoMu27", "bool"),
        ]
    elif dataset_era == "2018":
        datastructures["EventVariables"] += [
            ("HLT_IsoMu24", "bool"),
        ]

    return datastructures

###
### Threading stuff
###

def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    while not tokill():
        ds = dataset_generator.nextone()
        if ds is None:
            break 
        batches_queue.put(ds, block=True)
    #print("Cleaning up threaded_batches_feeder worker", threading.get_ident())
    return

class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.to_kill = False
    
    def __call__(self):
        return self.to_kill
    
    def set_tokill(self,tokill):
        with self.lock:
            self.to_kill = tokill

class InputGen:
    def __init__(self, name, era, paths, is_mc, nthreads, chunksize, cache_location, datapath):
        self.name = name
        self.era = era
        self.paths_chunks = list(chunks(paths, chunksize))
        self.chunk_lock = threading.Lock()
        self.loaded_lock = threading.Lock()
        self.num_chunk = 0
        self.num_loaded = 0
        self.is_mc = is_mc
        self.nthreads = nthreads
        self.cache_location = cache_location
        self.datapath = datapath
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=nthreads)

    def is_done(self):
        return (self.num_chunk == len(self.paths_chunks)) and (self.num_loaded == len(self.paths_chunks))
 
    def __iter__(self):
        return self.generator()

    #did not make this a generator to simplify handling the thread locks
    def nextone(self):
        self.chunk_lock.acquire()

        if self.num_chunk > 0 and self.num_chunk == len(self.paths_chunks):
            self.chunk_lock.release()
            #print("Generator is done: num_chunk={0}, len(self.paths_chunks)={1}".format(self.num_chunk, len(self.paths_chunks)))
            return None

        ds = create_dataset(
            self.name, self.paths_chunks[self.num_chunk],
            self.is_mc, self.cache_location, self.datapath, self.is_mc)

        ds.era = self.era
        ds.numpy_lib = numpy
        ds.num_chunk = self.num_chunk
        self.num_chunk += 1
        self.chunk_lock.release()

        # Load caches on multiple threads
        ds.from_cache(executor=self.executor, verbose=False)

        # Merge data arrays from multiple files into one big array
        ds.merge_inplace()

        # Increment the counter for number of loaded datasets
        with self.loaded_lock:
            self.num_loaded += 1

        return ds

    def __call__(self):
        return self.__iter__()


###
### Functions not currently used
###

def significance_templates(sig_samples, bkg_samples, rets, analysis, histogram_names, do_plots=False, ntoys=1):
     
    Zs = []
    Zs_naive = []
    if do_plots:
        for k in histogram_names:
            plt.figure(figsize=(4,4))
            ax = plt.axes()
            plt.title(k)
            for samp in sig_samples:
                h = rets[samp][analysis][k]["puWeight"]
                plot_hist_step(ax, h.edges, 100*h.contents, 100*np.sqrt(h.contents_w2), kwargs_step={"label":samp})
            for samp in bkg_samples:
                h = rets[samp][analysis][k]["puWeight"]
                plot_hist_step(ax, h.edges, h.contents, np.sqrt(h.contents_w2), kwargs_step={"label":samp})

    #         for name in ["ggh", "tth", "vbf", "wmh", "wph", "zh"]:
    #             plot_hist(100*rets[name][analysis][k]["puWeight"], label="{0} ({1:.2E})".format(name, np.sum(rets[name][k]["puWeight"].contents)))
    #         plot_hist(rets["dy"][k]["puWeight"], color="black", marker="o",
    #             label="DY ({0:.2E})".format(np.sum(rets["dy"][k]["puWeight"].contents)), linewidth=0, elinewidth=1
    #         )
            plt.legend(frameon=False, ncol=2)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, 5*ymax)
    
    for i in range(ntoys):
        arrs_sig = []
        arrs_bkg = []
        for hn in histogram_names:
            arrs = {}
            for samp in sig_samples + bkg_samples:
                if ntoys == 1:
                    arrs[samp] = rets[samp][analysis][hn]["puWeight"].contents,
                else:
                    arrs[samp] = np.random.normal(
                        rets[samp][analysis][hn]["puWeight"].contents,
                        np.sqrt(rets[samp][analysis][hn]["puWeight"].contents_w2)
                    )
        
            arr_sig = np.sum([arrs[s] for s in sig_samples])
            arr_bkg = np.sum([arrs[s] for s in bkg_samples])
            arrs_sig += [arr_sig]
            arrs_bkg += [arr_bkg]
        
        arr_sig = np.hstack(arrs_sig)
        arr_bkg = np.hstack(arrs_bkg)

        Z = sig_q0_asimov(arr_sig, arr_bkg)
        Zs += [Z]

        Znaive = sig_naive(arr_sig, arr_bkg)
        Zs_naive += [Znaive]
    return (np.mean(Zs), np.std(Zs)), (np.mean(Zs_naive), np.std(Zs_naive))

def compute_significances(sig_samples, bkg_samples, r, analyses):
    Zs = []
    for an in analyses:
        templates = [c for c in r["ggh"][an].keys() if "__cat" in c and c.endswith("__inv_mass")]
        (Z, eZ), (Zc, eZc) = significance_templates(
            sig_samples, bkg_samples, r, an, templates, ntoys=1
        )
        Zs += [(an, Z)]
    return sorted(Zs, key=lambda x: x[1], reverse=True)

def load_analysis(mc_samples, outpath, cross_sections, cat_trees):
    res = {}
    #res["data"] = json.load(open("../out/data_2017.json"))
    #lumi = res["data"]["baseline"]["int_lumi"]
    lumi = 41000.0

    rets = {
        k: json.load(open("{0}/{1}.json".format(outpath, k))) for k in mc_samples
    }

    histograms = {}
    for name in mc_samples:
        ret = rets[name]
        histograms[name] = {}
        for analysis in cat_trees:
            ret_an = rets[name]["baseline"][analysis]
            histograms[name][analysis] = {}
            for kn in ret_an.keys():
                if kn.startswith("hist_"):
                    histograms[name][analysis][kn] = {}
                    for w in ret_an[kn].keys():
                        h = (1.0 / ret["gen_sumweights"]) * lumi * cross_sections[name] * Histogram.from_dict(ret_an[kn][w])
                        h.label = "{0} ({1:.1E})".format(name, np.sum(h.contents))
                        histograms[name][analysis][kn][w] = h

    return histograms

def optimize_categories(sig_samples, bkg_samples, varlist, datasets, lumidata, lumimask, pu_corrections_2017, cross_sections, args, analysis_parameters, best_tree):
    Zprev = 0
    #Run optimization
    for num_iter in range(args.niter):
        outpath = "{0}/iter_{1}".format(args.out, num_iter)

        try:
            os.makedirs(outpath)
        except FileExistsError as e:
            pass

        analysis_parameters["baseline"]["categorization_trees"] = {}
        #analysis_parameters["baseline"]["categorization_trees"] = {"varA": copy.deepcopy(varA), "varB": copy.deepcopy(varB)}
        analysis_parameters["baseline"]["categorization_trees"]["previous_best"] = copy.deepcopy(best_tree)

        cut_trees = generate_cut_trees(100, varlist, best_tree)
        for icut, dt in enumerate(cut_trees):
            an_name = "an_cuts_{0}".format(icut)
            analysis_parameters["baseline"]["categorization_trees"][an_name] = dt

        with open('{0}/parameters.pickle'.format(outpath), 'wb') as handle:
            pickle.dump(analysis_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        run_analysis(args, outpath, datasets, analysis_parameters, lumidata, lumimask, pu_corrections)
        cut_trees = sorted(list(analysis_parameters["baseline"]["categorization_trees"].keys()), reverse=True)
        r = load_analysis(sig_samples + bkg_samples, outpath, cross_sections, cut_trees)
        print("computing expected significances")
        Zs = compute_significances(sig_samples, bkg_samples, r, cut_trees)

        with open('{0}/sigs.pickle'.format(outpath), 'wb') as handle:
            pickle.dump(Zs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("optimization", num_iter, Zs[:10], Zprev)
        best_tree = copy.deepcopy(analysis_parameters["baseline"]["categorization_trees"][Zs[0][0]])
        Zprev = Zs[0][1]

    return best_tree

def parse_nvidia_smi():
    """Returns the GPU symmetric multiprocessor and memory usage in %
    """
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle) 
    return {"gpu": res.gpu, "mem": res.memory}

def threaded_metrics(tokill, train_batches_queue):
    c = psutil.disk_io_counters()
    bytes_read_start = c.read_bytes

    while not tokill(): 
        dt = 0.5
        
        c = psutil.disk_io_counters()

        bytes_read_speed = (c.read_bytes - bytes_read_start)/dt/1024.0/1024.0
        bytes_read_start = c.read_bytes

        d = parse_nvidia_smi()
        print("metrics", time.time(), "IO speed", bytes_read_speed, "MB/s", "CPU", psutil.cpu_percent(), "GPU", d["gpu"], "GPUmem", d["mem"], "qsize", train_batches_queue.qsize())
        sys.stdout.flush()
        time.sleep(dt)

    return
