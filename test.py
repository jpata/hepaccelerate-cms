import glob
import time
import numpy
import os
import sys
import cupy

import uproot
import hepaccelerate
from hepaccelerate import Results


#Choose the backend
use_cuda = bool(int(os.environ.get("HEPACCELERATE_CUDA", 0)))
if use_cuda:
    print("Using the GPU CUDA backend")
    import cupy
    NUMPY_LIB = cupy
    from analysisgpu import *
else:
    print("Using the numpy CPU backend")
    NUMPY_LIB = numpy
    from analysiscpu import *

def get_histogram(data, weights, bins):
    return hepaccelerate.Histogram(*histogram_from_vector(data, weights, bins))

def get_selected_muons(muons, mask_events, mu_pt_cut_leading, mu_pt_cut_subleading, mu_iso_cut):
    """
    Given a list of muons in events, selects the muons that pass quality, momentum and charge criteria.
    Selects events that have at least 2 such muons. Selections are made by producing boolean masks.
 
    muons (list of JaggedArray) - The muon content of a given file
    mask_events (array of bool) - a mask of events that are used for muon processing
    mu_pt_cut_leading (float) - the pt cut on the leading muon
    mu_pt_cut_subleading (float) - the pt cut on all other muons
    mu_iso_cut (float) - cut to choose isolated muons

    """ 
    passes_iso = muons.pfRelIso03_all < mu_iso_cut
    passes_id = muons.mediumId == 1
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    
    #select muons that pass these cuts
    muons_passing_id = passes_iso & passes_id & passes_subleading_pt
   
    #select events that have muons passing cuts 
    events_passes_muid = sum_in_offsets(muons, muons_passing_id, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2
    events_passes_leading_pt = sum_in_offsets(muons, muons_passing_id & passes_leading_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 1
    events_passes_subleading_pt = sum_in_offsets(muons, muons_passing_id & passes_subleading_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2

    base_event_sel = mask_events & events_passes_muid & events_passes_leading_pt & events_passes_subleading_pt
    
    muons_passing_os = select_muons_opposite_sign(muons, muons_passing_id & passes_subleading_pt)
    events_passes_os = sum_in_offsets(muons, muons_passing_os, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 2
    
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    
    return {
         "selected_events": final_event_sel,
         "selected_muons": final_muon_sel,
    }


def get_selected_jets(jets, muons, mask_muons, mask_events, jet_pt_cut, jet_eta_cut, dr_cut):
    """
    Given jets and selected muons in events, choose jets that pass quality criteria and that are not dR-matched
    to muons.
 
    """
    jets_pass_dr = mask_deltar_first(jets, jets.masks["all"], muons, mask_muons, dr_cut)
    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = (jets.pt > jet_pt_cut) & (NUMPY_LIB.abs(jets.eta) < jet_eta_cut) & (((jets.jetId & 2)>>1)==1) & jets_pass_dr

    num_jets = sum_in_offsets(jets, selected_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)

    return {
        "selected_jets": selected_jets,
        "num_jets": num_jets
    }

def compute_inv_mass(objects, mask_events, mask_objects):
    pt = objects.pt
    eta = objects.eta
    phi = objects.phi
    mass = objects.mass

    px = pt * NUMPY_LIB.cos(phi)
    py = pt * NUMPY_LIB.sin(phi)
    pz = pt * NUMPY_LIB.sinh(eta)

    px_total = sum_in_offsets(objects, px, mask_events, mask_objects)
    py_total = sum_in_offsets(objects, py, mask_events, mask_objects)
    pz_total = sum_in_offsets(objects, pz, mask_events, mask_objects)
    mass_total = sum_in_offsets(objects, mass, mask_events, mask_objects)

    inv_mass = NUMPY_LIB.sqrt(px_total**2 + py_total**2 + pz_total**2 - mass_total**2)
    return inv_mass

def fill_with_weights(values, weights, weight_dict, mask, bins):
    ret = {
        wn: get_histogram(values[mask], (weights*weight_dict[wn])[mask], bins)
        for wn in weight_dict.keys()
    }
    return ret

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0

def compute_pu_weights(pu_corrections_target, weights, mc_nvtx, reco_nvtx):
    pu_edges, (values_nom, values_up, values_down) = pu_corrections_target

    src_pu_hist = get_histogram(mc_nvtx, weights, pu_edges)
    norm = sum(src_pu_hist.contents)
    src_pu_hist.contents = src_pu_hist.contents/norm
    src_pu_hist.contents_w2 = src_pu_hist.contents_w2/norm


    ratio = values_nom / src_pu_hist.contents
    remove_inf_nan(ratio)
    pu_weights = NUMPY_LIB.zeros_like(weights)
    get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio), pu_weights)
    
    ratio_up = values_up / src_pu_hist.contents
    remove_inf_nan(ratio_up)
    pu_weights_up = NUMPY_LIB.zeros_like(weights)
    get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio_up), pu_weights_up)
    
    ratio_down = values_down / src_pu_hist.contents
    remove_inf_nan(ratio_down)
    pu_weights_down = NUMPY_LIB.zeros_like(weights)
    get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio_down), pu_weights_down)
    
    return pu_weights, pu_weights_up, pu_weights_down

def select_events_trigger(scalars, mask_events):
    mask_events = mask_events & scalars["HLT_IsoMu24"] & scalars["Flag_METFilters"] & scalars["Flag_goodVertices"]

def analyze_data(
    muons, jets, scalars,
    is_mc=True,
    pu_corrections_target=None,
    mu_pt_cut_leading=26, mu_pt_cut_subleading=10,
    mu_iso_cut=0.3, jet_pt_cut=30,
    jet_eta_cut=4.7, jet_mu_drcut=0.4,
    doverify=True,
    debug=True
    ):
    
    weights = NUMPY_LIB.ones(len(muons), dtype=NUMPY_LIB.float32)
    weights = weights# * scalars["genWeight"]
    if debug:
        print("Starting analysis with {0} events".format(len(weights)))

    mask_events = NUMPY_LIB.ones(len(muons), dtype=NUMPY_LIB.bool)
    select_events_trigger(scalars, mask_events)
    if debug:
        print("{0} events passed trigger".format(NUMPY_LIB.sum(mask_events)))

    if is_mc:
        pu_weights, pu_weights_up, pu_weights_down = compute_pu_weights(pu_corrections_target, weights, scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        scalars["puWeight"] = pu_weights
        scalars["puWeight_up"] = pu_weights_up
        scalars["puWeight_down"] = pu_weights_down

    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(muons, mask_events, mu_pt_cut_leading, mu_pt_cut_subleading, mu_iso_cut)
    
    if doverify:
        z = sum_in_offsets(muons, ret_mu["selected_muons"], ret_mu["selected_events"], ret_mu["selected_muons"], dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))
    if debug:
        print("{0} events passed muon".format(NUMPY_LIB.sum(ret_mu["selected_events"])))

    #get the passing jets for events that pass muon selection
    ret_jet = get_selected_jets(jets, muons, ret_mu["selected_muons"], mask_events, jet_pt_cut, jet_eta_cut, jet_mu_drcut)    
    if doverify:
        z = min_in_offsets(jets, jets.pt, ret_mu["selected_events"], ret_jet["selected_jets"])
        assert(NUMPY_LIB.all(z[z>0] > jet_pt_cut))

        
    inv_mass = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"])
    
    inds = NUMPY_LIB.zeros(len(muons), dtype=NUMPY_LIB.int32)
    leading_muon_pt = get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_muon_eta = get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    leading_jet_pt = get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    leading_jet_eta = get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    
    inds[:] = 1
    subleading_muon_pt = get_in_offsets(muons.pt, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_muon_eta = get_in_offsets(muons.eta, muons.offsets, inds, ret_mu["selected_events"], ret_mu["selected_muons"])
    subleading_jet_pt = get_in_offsets(jets.pt, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    subleading_jet_eta = get_in_offsets(jets.eta, jets.offsets, inds, ret_mu["selected_events"], ret_jet["selected_jets"])
    
    if doverify:
        assert(NUMPY_LIB.all(leading_muon_pt[leading_muon_pt>0] > mu_pt_cut_leading))
        assert(NUMPY_LIB.all(subleading_muon_pt[subleading_muon_pt>0] > mu_pt_cut_subleading))

    
    hist_inv_mass_d = fill_with_weights(inv_mass, weights, scalars, ret_mu["selected_events"], NUMPY_LIB.linspace(0,500,201))

    #get histograms of leading and subleading muon momenta
    hist_leading_muon_pt_d = fill_with_weights(leading_muon_pt, weights, scalars, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 401))
    hist_subleading_muon_pt_d = fill_with_weights(subleading_muon_pt, weights, scalars, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 401))
    
    #create lots of histograms 
    hist_leading_muon_pt_d_weights = fill_with_weights(leading_muon_pt, weights, scalars, ret_mu["selected_events"], NUMPY_LIB.linspace(0.0, 200.0, 401))  

    #get histograms of leading and subleading muon eta
    hist_leading_muon_eta_d = fill_with_weights(leading_muon_eta, weights, scalars, ret_mu["selected_events"], NUMPY_LIB.linspace(-4.0, 4.0, 401))
    hist_subleading_muon_eta_d = fill_with_weights(subleading_muon_eta, weights, scalars, ret_mu["selected_events"], NUMPY_LIB.linspace(-4.0, 4.0, 401))

    hist_leading_jet_pt_d = fill_with_weights(leading_jet_pt, weights, scalars, ret_mu["selected_events"] & (ret_jet["num_jets"]>=1), NUMPY_LIB.linspace(0, 300.0, 401))
    hist_subleading_jet_pt_d = fill_with_weights(subleading_jet_pt, weights, scalars, ret_mu["selected_events"] & (ret_jet["num_jets"]>=2), NUMPY_LIB.linspace(0, 300.0, 401))

    return Results({
        "hist_inv_mass_d": Results(hist_inv_mass_d),
        "hist_mu0_pt": Results(hist_leading_muon_pt_d),
        
        "hist_mu0_pt_weights": Results(hist_leading_muon_pt_d_weights),
        "hist_mu1_pt": Results(hist_subleading_muon_pt_d),
        
        "hist_mu0_eta": Results(hist_leading_muon_eta_d),
        "hist_mu1_eta": Results(hist_subleading_muon_eta_d),
        
        "hist_leading_jet_pt": Results(hist_leading_jet_pt_d),
        "hist_subleading_jet_pt": Results(hist_subleading_jet_pt_d),        
    })

def load_puhist_target(filename):
    fi = uproot.open("RunII_2016_data.root")
    
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

if __name__ == "__main__":
    #

    nev_total = 0
    t0 = time.time()
    for datasetname, globpattern, is_mc in [
        ("data_2017", "/nvmedata/store/data/Run2017*/SingleMuon/NANOAOD/Nano14Dec2018-v1/**/*.root", False),
        ("dy", "/nvmedata/store/mc/RunIIFall17NanoAOD/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", True),
        ("ggh", "/nvmedata/store/mc/RunIIFall17NanoAOD/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/**/*.root", True),
        ]:
        filenames_all = glob.glob(globpattern, recursive=True)
        filenames_all = [fn for fn in filenames_all if not "Friend" in fn][:100]
        for filenames in chunks(filenames_all, 20):
            arrays_ev = [
                "PV_npvsGood", "Flag_METFilters", "Flag_goodVertices", "HLT_IsoMu24"
            ]
            if is_mc:
                arrays_ev += ["Pileup_nTrueInt", "Generator_weight", "genWeight"]
            arrays_jet = [
                "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId"
            ]
            
            arrays_muon = [
                "nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso03_all", "Muon_mediumId", "Muon_charge"
            ]
            
            pu_corrections_2016 = load_puhist_target("RunII_2016_data.root") 
 
            arrays_to_load = arrays_ev + arrays_jet + arrays_muon
            ds = hepaccelerate.NanoAODDataset(filenames, arrays_to_load, "Events", NUMPY_LIB)
            prepare_cache = "--prepare-cache" in sys.argv
            
            if prepare_cache:
                ds.preload(nthreads=16, do_progress=True, event_vars=[bytes(x, encoding='ascii') for x in arrays_ev])
                ds.to_cache(do_progress=True)
            else:
                ds.from_cache(do_progress=True, nthreads=16)
            nev_total += len(ds)
            #ds.make_random_weights()

            ret = ds.analyze(analyze_data, is_mc=is_mc, pu_corrections_target=pu_corrections_2016, mu_pt_cut_leading=30, debug=False)
            #ret_35 = ds.analyze(analyze_data, mu_pt_cut_leading=35)
            #ret_40 = ds.analyze(analyze_data, mu_pt_cut_leading=40)
            ret.save_json("{0}.json".format(datasetname))
             
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
    t1 = time.time()
    dt = t1 - t0
    print("Processed {0:.2E} events in total, {1:.1f} seconds, {2:.2E} Hz".format(nev_total, dt, nev_total/dt))
