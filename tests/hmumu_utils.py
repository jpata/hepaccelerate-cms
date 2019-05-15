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

import threading
from threading import Thread
from queue import Queue
import queue
import concurrent.futures

import hepaccelerate
import hepaccelerate.utils
from hepaccelerate.utils import Results
from hepaccelerate.utils import NanoAODDataset
from hepaccelerate.utils import Histogram
import hepaccelerate.backend_cpu as backend_cpu
from hepaccelerate.plotting import plot_hist_step
from hepaccelerate.decisiontree import DecisionTreeNode, DecisionTreeLeaf, make_random_node, grow_randomly, make_random_tree, prune_randomly, generate_cut_trees
from hepaccelerate.stats import likelihood, sig_q0_asimov, sig_naive

ha = None
NUMPY_LIB = None
genweight_scalefactor = 0.00001

try:
    import nvidia_smi
except Exception as e:
    print("Could not import nvidia_smi", file=sys.stderr)
    pass

def parse_nvidia_smi():
    """Returns the GPU symmetric multiprocessor and memory usage in %
    """
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle) 
    return {"gpu": res.gpu, "mem": res.memory}

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

def get_histogram(data, weights, bins):
    """Given N-unit vectors of data and weights, returns the histogram in bins
    """
    return Histogram(*ha.histogram_from_vector(data, weights, bins))

def get_selected_muons(
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
    passes_global_tracker = (muons.isGlobal == 1) & (muons.isTracker == 1)
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(muons.eta) < mu_aeta_cut
    muons_passing_id =  (
        passes_global_tracker & passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )

    #Get muons that are high-pt and are matched to trigger object
    mask_trigger_objects_mu = (trigobj.id == 13)
    muons_matched_to_trigobj = NUMPY_LIB.invert(ha.mask_deltar_first(
        muons, muons_passing_id & passes_leading_pt, trigobj,
        mask_trigger_objects_mu, muon_trig_match_dr
    ))

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

    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    additional_muon_sel = muons_passing_id & passes_subleading_pt & NUMPY_LIB.invert(muons_passing_os)

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
    jets, muons,
    mask_muons,
    mask_events,
    jet_pt_cut, jet_eta_cut,
    jet_dr_cut, jet_id, jet_puid, jet_btag
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

    selected_jets = ((jets.pt > jet_pt_cut) &
        (NUMPY_LIB.abs(jets.eta) < jet_eta_cut) & pass_jetid & pass_jet_puid)
    jets_pass_dr = ha.mask_deltar_first(
        jets, selected_jets, muons,
        muons.masks["all"], jet_dr_cut)

    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = selected_jets & jets_pass_dr
   
    #produce a mask that selects the first two selected jets 
    first_two_jets = NUMPY_LIB.zeros_like(selected_jets)
    targets = NUMPY_LIB.ones_like(mask_events)
    inds = NUMPY_LIB.zeros_like(mask_events)
    ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, mask_events, selected_jets)
    inds[:] = 1
    ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, mask_events, selected_jets)

    dijet_inv_mass = compute_inv_mass(jets, mask_events, selected_jets & first_two_jets)

    
    selected_jets_btag = selected_jets & (jets.btagDeepB >= jet_btag)

    num_jets = ha.sum_in_offsets(jets, selected_jets, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)

    num_jets_btag = ha.sum_in_offsets(jets, selected_jets_btag, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)
        
    return {
        "selected_jets": selected_jets,
        "num_jets": num_jets,
        "num_jets_btag": num_jets_btag,
        "dijet_inv_mass": dijet_inv_mass
    }

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
    src_pu_hist.contents = src_pu_hist.contents/norm
    src_pu_hist.contents_w2 = src_pu_hist.contents_w2/norm

    ratio = values_nom / src_pu_hist.contents
    remove_inf_nan(ratio)
    pu_weights = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges),
        NUMPY_LIB.array(ratio), pu_weights)
    fix_large_weights(pu_weights) 
     
    ratio_up = values_up / src_pu_hist.contents
    remove_inf_nan(ratio_up)
    pu_weights_up = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges),
        NUMPY_LIB.array(ratio_up), pu_weights_up)
    fix_large_weights(pu_weights_up) 
    
    ratio_down = values_down / src_pu_hist.contents
    remove_inf_nan(ratio_down)
    pu_weights_down = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges),
        NUMPY_LIB.array(ratio_down), pu_weights_down)
    fix_large_weights(pu_weights_down) 
    
    return pu_weights, pu_weights_up, pu_weights_down

def select_events_trigger(scalars, mask_events, parameters):

    flags = [
        #"Flag_BadChargedCandidateFilter",
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

    mask_events = mask_events & scalars["HLT_IsoMu27"] & pvsel

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
    for fi in filenames:
        ff = uproot.open(fi)
        bl = ff.get("Runs")
        arr = bl.array("genEventSumw")
        arr = arr * genweight_scalefactor
        sumw += arr.sum()
    return sumw


def analyze_data(
    data,
    is_mc=True,
    pu_corrections=None,
    lumimask=None,
    lumidata=None,
    parameters={},
    parameter_set_name="",
    doverify=False,
    debug=False
    ):

    muons = data["Muon"]
    jets = data["Jet"]
    electrons = data["Electron"]
    trigobj = data["TrigObj"]
    scalars = data["eventvars"]

    # scalars["run"] = NUMPY_LIB.array(scalars["run"], dtype=NUMPY_LIB.uint32)
    # scalars["luminosityBlock"] = NUMPY_LIB.array(scalars["run"], dtype=NUMPY_LIB.uint32)

    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    select_events_trigger(scalars, mask_events, parameters)

    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"] * genweight_scalefactor
        pu_weights, pu_weights_up, pu_weights_down = compute_pu_weights(
            pu_corrections, weights["nominal"],
            scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        weights["puWeight"] = weights["nominal"] * pu_weights
        #weights["puWeight_up"] = weights["nominal"] * pu_weights_up
        #weights["puWeight_down"] = weights["nominal"] * pu_weights_down

        scalars["genWeight"] = scalars["genWeight"]


    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(
        muons, trigobj, mask_events,
        parameters["muon_pt_leading"], parameters["muon_pt"],
        parameters["muon_eta"], parameters["muon_iso"],
        parameters["muon_id"], parameters["muon_trigger_match_dr"]
    )
    
    #currently does not work
    #mu_compact = muons.compact_struct(ret_mu["selected_events"])

    if doverify:
        z = ha.sum_in_offsets(muons, ret_mu["selected_muons"],
            ret_mu["selected_events"],
            ret_mu["selected_muons"], dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))

    ret_el = get_selected_electrons(electrons, parameters["extra_electrons_pt"], parameters["extra_electrons_eta"], parameters["extra_electrons_id"])
    
    #get the passing jets for events that pass muon selection
    ret_jet = get_selected_jets(
        jets, muons,
        ret_mu["selected_muons"], mask_events,
        parameters["jet_pt"],
        parameters["jet_eta"],
        parameters["jet_mu_dr"],
        parameters["jet_id"],
        parameters["jet_puid"],
        parameters["jet_btag"]
    )
    
    if doverify:
        z = ha.min_in_offsets(jets, jets.pt, ret_mu["selected_events"], ret_jet["selected_jets"])
        assert(NUMPY_LIB.all(z[z>0] > parameters["jet_pt"]))


    #get the invariant mass
    inv_mass = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"])

    # #choose first muon
    inds = NUMPY_LIB.zeros(muons.numevents(), dtype=NUMPY_LIB.int32)
    leading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_muon_eta = ha.get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    leading_jet_eta = ha.get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])

    #choose second muon
    inds[:] = 1
    subleading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_muon_eta = ha.get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    subleading_jet_eta = ha.get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])

    #get the number of additional muons (not OS) that pass ID and iso cuts
    additional_muons = ha.sum_in_offsets(muons, ret_mu["additional_muon_sel"], ret_mu["selected_events"], ret_mu["additional_muon_sel"], dtype=NUMPY_LIB.int8)
    additional_electrons = ha.sum_in_offsets(electrons, ret_el["additional_electron_sel"], ret_mu["selected_events"], ret_el["additional_electron_sel"], dtype=NUMPY_LIB.int8)
    additional_leptons = additional_muons + additional_electrons
    
    ret_el
    if doverify:
        assert(NUMPY_LIB.all(leading_muon_pt[leading_muon_pt>0] > parameters["muon_pt_leading"]))
        assert(NUMPY_LIB.all(subleading_muon_pt[subleading_muon_pt>0] > parameters["muon_pt"]))

    hists = {}
    #hists["hist__dimuon__npvs"] = fill_with_weights(scalars["PV_npvsGood"], weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0,100,101))
    #hists["hist__dimuon__inv_mass"] = fill_with_weights(inv_mass, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(50,200,101))

    ##get histograms of leading and subleading muon momenta
    #hists["hist__dimuon__leading_muon_pt"] = fill_with_weights(leading_muon_pt, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 101))
    #hists["hist__dimuon__subleading_muon_pt"] = fill_with_weights(subleading_muon_pt, weights, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 101))

    #hists["hist__dimuon__leading_muon_eta"] = fill_with_weights(
    #    leading_muon_eta, weights, ret_mu["selected_events"],
    #    NUMPY_LIB.linspace(-4.0, 4.0, 101)
    #)
    #hists["hist__dimuon__subleading_muon_eta"] = fill_with_weights(
    #    subleading_muon_eta, weights, ret_mu["selected_events"],
    #    NUMPY_LIB.linspace(-4.0, 4.0, 101)
    #)

    masswindow_110_150 = ((inv_mass >= 110) & (inv_mass < 150))
    masswindow_120_130 = ((inv_mass >= 120) & (inv_mass < 130))

    hists["hist__dimuon_invmass_110_150__inv_mass"] = fill_with_weights(
       inv_mass, weights, ret_mu["selected_events"] & masswindow_110_150,
       NUMPY_LIB.linspace(110, 150, parameters["inv_mass_bins"])
    )

    hists["hist__dimuon_invmass_120_130__inv_mass"] = fill_with_weights(
       inv_mass, weights, ret_mu["selected_events"] & masswindow_110_150,
       NUMPY_LIB.linspace(120, 130, parameters["inv_mass_bins"])
    )

    cut_pre = ret_mu["selected_events"] & masswindow_120_130
    for cat_tree_name, cat_tree in parameters["categorization_trees"].items():
        hists_cat = Results({})
        categories = cat_tree.predict(len(inv_mass), {
            "dimuon_inv_mass": inv_mass,
            "dijet_inv_mass": ret_jet["dijet_inv_mass"],
            "num_jets": ret_jet["num_jets"],
            "num_jets_btag": ret_jet["num_jets_btag"],
            "leading_mu_abs_eta": NUMPY_LIB.abs(leading_muon_eta),
            "additional_leptons": additional_leptons,
        })
        
        #Make histograms for each category
        for cat in [l.value for l in cat_tree.get_all_leaves()]:
            cut = cut_pre & (categories == cat)
            hists_cat["hist__cat{0}__inv_mass".format(int(cat))] = Results(fill_with_weights(
                inv_mass, weights,
                cut,
                NUMPY_LIB.linspace(120, 130, parameters["inv_mass_bins"])
            ))
        hists[cat_tree_name] = hists_cat

    masswindow_exclude_120_130 = masswindow_110_150 & (NUMPY_LIB.invert((inv_mass >= 120) & (inv_mass <= 130)))
    hists["hist__dimuon_invmass_110_150_exclude_120_130_jge1__leading_jet_pt"] = fill_with_weights(leading_jet_pt, weights,
       ret_mu["selected_events"] & (ret_jet["num_jets"]>=1) & masswindow_exclude_120_130, NUMPY_LIB.linspace(30, 200.0, 101))
    hists["hist__dimuon_invmass_110_150_exclude_120_130_jge2__subleading_jet_pt"] = fill_with_weights(subleading_jet_pt, weights,
       ret_mu["selected_events"] & (ret_jet["num_jets"]>=2) & masswindow_exclude_120_130, NUMPY_LIB.linspace(30, 100.0, 101))
    hists["hist__dimuon_invmass_110_150_exclude_120_130__numjet"] = fill_with_weights(ret_jet["num_jets"], weights, ret_mu["selected_events"] & masswindow_exclude_120_130, NUMPY_LIB.linspace(0, 10, 11))

    int_lumi = 0
    if not is_mc and not (lumimask is None):
        runs = NUMPY_LIB.asnumpy(scalars["run"])
        lumis = NUMPY_LIB.asnumpy(scalars["luminosityBlock"])
        mask_lumi_golden_json = NUMPY_LIB.array(lumimask(runs, lumis))
        if parameter_set_name == "baseline":
            mask_events = mask_events & mask_lumi_golden_json
            #get integrated luminosity in this file
            if not (lumidata is None):
                int_lumi = get_int_lumi(runs, lumis, NUMPY_LIB.asnumpy(mask_lumi_golden_json), lumidata)

    ret = Results({
        "int_lumi": int_lumi,
    })
    for histname, hist in hists.items():
        ret[histname] = Results(hist)

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

def cache_data(filenames, datastructures, cache_location, nworkers=16):
    if nworkers == 1:
        tot_ev = 0
        tot_mb = 0
        for result in map(cache_data_multiproc_worker, [(fn, datastructures, cache_location) for fn in filenames]):
            tot_ev += result[0]
            tot_mb += result[1]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
            tot_ev = 0
            tot_mb = 0
            for result in executor.map(cache_data_multiproc_worker, [(fn, datastructures, cache_location) for fn in filenames]):
                tot_ev += result[0]
                tot_mb += result[1]
    return tot_ev, tot_mb

def create_dataset(filenames, datastructures, cache_location):
    ds = NanoAODDataset(filenames, datastructures, cache_location)
    return ds

def cache_data_multiproc_worker(args):
    filename, datastructure, cache_location = args
    t0 = time.time()
    ds = create_dataset([filename], datastructure, cache_location)
    ds.numpy_lib = np
    ds.preload()
    ds.make_objects()
    ds.to_cache()
    t1 = time.time()
    dt = t1 - t0
    processed_size_mb = ds.memsize()/1024.0/1024.0
    print("built cache for {0}, {1:.2f} MB, {2:.2E} Hz, {3:.2f} MB/s".format(filename, processed_size_mb, len(ds)/dt, processed_size_mb/dt))
    return len(ds), processed_size_mb

class InputGen:
    def __init__(self, paths, is_mc, nthreads, chunksize, cache_location):
        self.paths_chunks = list(chunks(paths, chunksize))
        self.chunk_lock = threading.Lock()
        self.loaded_lock = threading.Lock()
        self.num_chunk = 0
        self.num_loaded = 0
        self.is_mc = is_mc
        self.nthreads = nthreads
        self.cache_location = cache_location
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=nthreads)

    def is_done(self):
        return (self.num_chunk == len(self.paths_chunks)) and (self.num_loaded == len(self.paths_chunks))
 
    def __iter__(self):
        return self.generator()

    #did not make this a generator to simplify handling the thread locks
    def nextone(self):

        self.chunk_lock.acquire()
        if self.num_chunk == len(self.paths_chunks):
            self.chunk_lock.release()
            return None

        ds = create_dataset(self.paths_chunks[self.num_chunk], self.is_mc, self.cache_location)
        self.num_chunk += 1
        ds.numpy_lib = numpy
        self.chunk_lock.release()

        #load caches on multiple threads
        ds.from_cache(executor=self.executor, verbose=True)

        with self.loaded_lock:
            self.num_loaded += 1

        return ds

    def __call__(self):
        return self.__iter__()

def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    while not tokill():
        ds = dataset_generator.nextone()
        if not ds:
            break 
        batches_queue.put(ds, block=True)
    #print("Cleaning up threaded_batches_feeder worker", threading.get_ident())
    return

def event_loop(train_batches_queue, use_cuda, **kwargs):
    ds = train_batches_queue.get(block=True)
    #print("event_loop nev={0}, queued={1}".format(len(ds), train_batches_queue.qsize()))

    if use_cuda:
        #copy dataset to GPU and make sure future operations are done on it
        import cupy
        ds.numpy_lib = cupy
        ds.move_to_device(cupy)


    parameters = kwargs.pop("parameters")

    ret = {}
    for parameter_set_name, parameter_set in parameters.items():
        ret[parameter_set_name] = ds.analyze(analyze_data, parameter_set_name=parameter_set_name, parameters=parameter_set, **kwargs)
    ret["num_events"] = len(ds)

    train_batches_queue.task_done()

    #clean up CUDA memory
    if use_cuda:
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    return ret, len(ds), ds.memsize()/1024.0/1024.0



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

def run_analysis(args, outpath, datasets, parameters, lumidata, lumimask, pu_corrections):
    nev_total = 0
    t0 = time.time()
    
    print(args)

    if args.use_cuda:
        import cupy
    else:
        os.environ["NUMBA_NUM_THREADS"] = str(args.nthreads)

    processed_size_mb = 0
    for datasetname, globpattern, is_mc in datasets:

        filenames_all = glob.glob(args.datapath + globpattern, recursive=True)
        filenames_all = [fn for fn in filenames_all if not "Friend" in fn]
        print("Dataset {0} has {1} files".format(datasetname, len(filenames_all)))
        if args.maxfiles > 0:
            filenames_all = filenames_all[:args.maxfiles]

        datastructure = create_datastructure(is_mc)

        if "cache" in args.action:
            print("Preparing caches from ROOT files")
            _nev_total, _processed_size_mb = cache_data(filenames_all, datastructure, args.cache_location, nworkers=args.nthreads)
            nev_total += _nev_total
            processed_size_mb += _processed_size_mb

        if "analyze" in args.action:        

            training_set_generator = InputGen(list(filenames_all), datastructure, args.nthreads, args.chunksize, args.cache_location)
            threadk = thread_killer()
            threadk.set_tokill(False)
            train_batches_queue = Queue(maxsize=20)
            
            if args.async_data:
                for _ in range(1):
                    t = Thread(target=threaded_batches_feeder, args=(threadk, train_batches_queue, training_set_generator))
                    t.start()

            t = Thread(target=threaded_metrics, args=(threadk, train_batches_queue))
            t.start()

            rets = []
            num_processed = 0

            #loop over all data, call the analyze function
            while num_processed < len(training_set_generator.paths_chunks):
                if not args.async_data:
                    ds = training_set_generator.nextone()
                    if not ds:
                        break
                    train_batches_queue.put(ds)
                sys.stdout.write(".");sys.stdout.flush()
                ret, nev, memsize = event_loop(
                    train_batches_queue, args.use_cuda, debug=False,
                    verbose=True, is_mc=is_mc, lumimask=lumimask,
                    lumidata=lumidata,
                    pu_corrections=pu_corrections,
                    parameters=parameters) 
                rets += [ret]
                processed_size_mb += memsize
                nev_total += nev
                num_processed += 1
            print()

            #clean up threads
            threadk.set_tokill(True)

            #save output
            ret = sum(rets, Results({}))
            if is_mc:
                ret["gen_sumweights"] = get_gen_sumweights(filenames_all)
            ret.save_json("{0}/{1}.json".format(outpath, datasetname))

    t1 = time.time()
    dt = t1 - t0
    print("Overall processed {nev:.2E} ({nev}) events in total {size:.2f} GB, {dt:.1f} seconds, {evspeed:.2E} Hz, {sizespeed:.2f} MB/s".format(
        nev=nev_total, dt=dt, size=processed_size_mb/1024.0, evspeed=nev_total/dt, sizespeed=processed_size_mb/dt)
    )

def create_datastructure(is_mc):
    datastructures = {
        "Muon": [
            ("Muon_pt", "float32"), ("Muon_eta", "float32"),
            ("Muon_phi", "float32"), ("Muon_mass", "float32"),
            ("Muon_pfRelIso04_all", "float32"), ("Muon_mediumId", "bool"),
            ("Muon_tightId", "bool"), ("Muon_charge", "int32"),
            ("Muon_isGlobal", "bool"), ("Muon_isTracker", "bool")
        ],
        "Electron": [
            ("Electron_pt", "float32"), ("Electron_eta", "float32"),
            ("Electron_phi", "float32"), ("Electron_mass", "float32"),
            ("Electron_mvaFall17V1Iso_WP90", "bool"),
        ],
        "Jet": [
            ("Jet_pt", "float32"), ("Jet_eta", "float32"),
            ("Jet_phi", "float32"), ("Jet_mass", "float32"),
            ("Jet_btagDeepB", "float32"),
            ("Jet_jetId", "int32"), ("Jet_puId", "int32"),
        ],
        "TrigObj": [
            ("TrigObj_pt", "float32"),
            ("TrigObj_eta", "float32"),
            ("TrigObj_phi", "float32"),
            ("TrigObj_id", "int32")
        ],
        "EventVariables": [
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
            ("HLT_IsoMu27", "bool"),
            ("run", "uint32"),
            ("luminosityBlock", "uint32"),
            ("event", "uint64")
        ]
    }
    if is_mc:
        datastructures["EventVariables"] += [
            ("Pileup_nTrueInt", "uint32"),
            ("Generator_weight", "float32"),
            ("genWeight", "float32")
        ]
    return datastructures

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
        # print("{0}, Z={1:.3f} +- {2:.3f}, Zc={3:.3f} +- {4:.3f}".format(
        #     an, Z, eZ, Zc, eZc)
        # )
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

        run_analysis(args, outpath, datasets, analysis_parameters, lumidata, lumimask, pu_corrections_2017)
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
