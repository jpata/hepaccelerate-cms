import hepaccelerate
from hepaccelerate import Results
import glob
import time
import numpy
import os

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

def get_selected_muons(muons, mu_pt_cut_leading, mu_pt_cut_subleading, mu_iso_cut):
    
    passes_iso = muons.pfRelIso03_all < mu_iso_cut
    passes_id = muons.mediumId == 1
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    
    #select events with at least 2 muons passing cuts
    muons_passing_id = passes_iso & passes_id & passes_subleading_pt
    
    events_all = NUMPY_LIB.ones(len(muons), dtype=NUMPY_LIB.bool)
    events_passes_muid = sum_in_offsets(muons, muons_passing_id, events_all, muons.masks["all"], NUMPY_LIB.int8) >= 2
    events_passes_leading_pt = sum_in_offsets(muons, muons_passing_id & passes_leading_pt, events_all, muons.masks["all"], NUMPY_LIB.int8) >= 1
    events_passes_subleading_pt = sum_in_offsets(muons, muons_passing_id & passes_subleading_pt, events_all, muons.masks["all"], NUMPY_LIB.int8) >= 2

    base_event_sel = events_passes_muid & events_passes_leading_pt & events_passes_subleading_pt
    
    muons_passing_os = select_muons_opposite_sign(muons, muons_passing_id & passes_subleading_pt)
    events_passes_os = sum_in_offsets(muons, muons_passing_os, events_all, muons.masks["all"], NUMPY_LIB.int8) == 2
    
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    
    return {
         "selected_events": final_event_sel,
         "selected_muons": final_muon_sel,
    }


def get_selected_jets(jets, muons, mask_muons, jet_pt_cut, jet_eta_cut, dr_cut):
    events_all = NUMPY_LIB.ones(len(jets), dtype=NUMPY_LIB.bool)

    jets_pass_dr = mask_deltar_first(jets, jets.masks["all"], muons, mask_muons, dr_cut)
    jets.masks["pass_dr"] = jets_pass_dr
    selected_jets = (jets.pt > jet_pt_cut) & (NUMPY_LIB.abs(jets.eta) < jet_eta_cut) & (((jets.jetId & 2)>>1)==1) & jets_pass_dr

    num_jets = sum_in_offsets(jets, selected_jets, events_all, jets.masks["all"], NUMPY_LIB.int8)

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

def analyze_data(
    muons, jets, scalars,
    mu_pt_cut_leading=26, mu_pt_cut_subleading=10,
    mu_iso_cut=0.3, jet_pt_cut=30,
    jet_eta_cut=4.1, jet_mu_drcut=0.2, doverify=True
    ):

    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(muons, mu_pt_cut_leading, mu_pt_cut_subleading, mu_iso_cut)
    
    if doverify:
        z = sum_in_offsets(muons, ret_mu["selected_muons"], ret_mu["selected_events"], ret_mu["selected_muons"], dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))

    #get the passing jets for events that pass muon selection
    ret_jet = get_selected_jets(jets, muons, ret_mu["selected_muons"], jet_pt_cut, jet_eta_cut, jet_mu_drcut)    
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

    
    weights = NUMPY_LIB.ones(len(muons), dtype=NUMPY_LIB.float32)
    weights = weights * scalars["genWeight"]
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
        
#         #"hist_dimuon_mass_gen": hist_inv_mass_gen_d,        
#         "hist_dimuon_mass": hist_inv_mass_d,        
        
        "hist_leading_jet_pt": Results(hist_leading_jet_pt_d),
        "hist_subleading_jet_pt": Results(hist_subleading_jet_pt_d),        
    })

if __name__ == "__main__":
    filenames = glob.glob("/nvmedata/store/mc/RunIIFall17NanoAOD/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/**/*.root", recursive=True)[:40]
    
    arrays_ev = [
        "PV_npvsGood", "Flag_METFilters", "Flag_goodVertices", "Generator_weight", "genWeight", "HLT_IsoMu24"
    ]
    arrays_jet = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId"
    ]
    
    arrays_muon = [
        "nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso03_all", "Muon_mediumId", "Muon_charge"
    ]
    
    arrays_to_load = arrays_ev + arrays_jet + arrays_muon
    ds = hepaccelerate.NanoAODDataset(filenames, arrays_to_load, "Events", NUMPY_LIB)
    #ds.preload(nthreads=16, do_progress=True)
    #ds.to_cache(do_progress=True)
    ds.from_cache(do_progress=True)

    ds.make_random_weights()

    ret_30 = ds.analyze(analyze_data, mu_pt_cut_leading=30)
    ret_35 = ds.analyze(analyze_data, mu_pt_cut_leading=35)
    ret_40 = ds.analyze(analyze_data, mu_pt_cut_leading=40)
    ret_30.save_json("test.json")
