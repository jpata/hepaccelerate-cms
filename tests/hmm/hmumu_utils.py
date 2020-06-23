import pickle
import json
import threading
import uproot
import copy
import glob
import time
import numpy
import numpy as np
import sys
import os
import math
import warnings

import numba
import numba.cuda as cuda

from hepaccelerate.utils import Results, Dataset, Histogram
import hepaccelerate.backend_cpu as backend_cpu

from coffea.lookup_tools import extractor

from pars import runmap_numerical, runmap_numerical_r, data_runs, genweight_scalefactor, jer_unc, VBF_STXS_unc, HSTXS_rel, mass_point, btag_unc

#global variables need to be configured here for the hepaccelerate backend and numpy library
#they will be overwritten later
ha = None
NUMPY_LIB = None

#Use this to turn on debugging
debug = False
#debug = True
#event IDs for which to print out detailed information
debug_event_ids = [735225363]
#Run additional checks on the analyzed data to ensure consistency - for debugging
doverify = False

#raise an error if there is any inf or nan
def check_inf_nan(data):
    m = NUMPY_LIB.isinf(data)|NUMPY_LIB.isnan(data)
    assert(np.sum(m)==0) 

#fix inf or nan and raise a warning
def fix_inf_nan(data, default=0):
    m = NUMPY_LIB.isinf(data)|NUMPY_LIB.isnan(data)
    if NUMPY_LIB.sum(m) != 0:
        warnings.warn("array had {0} inf/nan entries!".format(NUMPY_LIB.sum(m)))
    data[m] = default


def analyze_data(
    data, analysis_corrections,
    parameters, parameter_set_name,
    random_seed,
    do_fsr=False, use_cuda=False):
    """Analyzes the dataset with a parameter set
    
        Args:
            data (hepaccelerate.Dataset): The dataset to analyze
            analysis_corrections (analysis_hmumu.AnalysisCorrections): The calibration data for the analysis
            parameters (dict): dictionary with all the cuts and other numerical parameters of the analysis
            parameter_set_name (str): the name of the parameter set that is being analyzed
            random_seed (int): the random seed used in smearing algorithms
        Returns:
            dict: a hepaccelerate.Results dictionary with all the results, primarily name-histogram pairs 
    """

    #old arguments
    dataset_name = data.name
    dataset_era = data.era
    is_mc = data.is_mc

    if use_cuda:
        import hepaccelerate.backend_cuda as backend_cuda

    # Collect results into this dictionary
    ret = Results({})

    #set the random seed to the predefined value
    NUMPY_LIB.random.seed(random_seed)

    if data.numfiles != 1:
        raise Exception("Currently support only 1 file per job descripton")

    #create variables for muons, jets etc
    muons = data.structs["Muon"][0] 
    fsrphotons = None
    if do_fsr:
        fsrphotons = data.structs["FsrPhoton"][0]
    jets = data.structs["Jet"][0]
    softjets = data.structs["SoftActivityJet"][0]
    electrons = data.structs["Electron"][0]
    trigobj = data.structs["TrigObj"][0]
    scalars = data.eventvars[0]

    LHEScalew = None
    if "dy" in dataset_name or "ewk" in dataset_name:
        LHEScalew = data.structs["LHEScaleWeight"][0]
    
    LHEPdfw = None
    if "dy" in dataset_name or "ewk" in dataset_name or "ggh" in dataset_name or "vbf" in dataset_name or "wmh" in dataset_name or "wph" in dataset_name or "zh" in dataset_name or "tth" in dataset_name:
        LHEPdfw = data.structs["LHEPdfWeight"][0]
    histo_bins = parameters["histo_bins"]

    #first mask of all events enabled
    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)

    #Golden JSON filtering (CPU-only)
    if not is_mc:
        mask_events = mask_events & NUMPY_LIB.array(analysis_corrections.lumimask[dataset_era](
            NUMPY_LIB.asnumpy(scalars["run"]),
            NUMPY_LIB.asnumpy(scalars["luminosityBlock"])))
 
    check_and_fix_qgl(jets)

    #output histograms 
    hists = {}

    #temporary hack for JaggedStruct.select_objects (relies on backend)
    muons.hepaccelerate_backend = ha
    if do_fsr:
        fsrphotons.hepaccelerate_backend = ha
    jets.hepaccelerate_backend = ha
    softjets.hepaccelerate_backend = ha

    #associate the muon genpt to reco muons based on the NanoAOD index
    genJet, genpart = get_genparticles(data, muons, jets, is_mc, use_cuda)
    
    #NNLOPS reweighting for ggH signal
    gghnnlopsw = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)

    #find genHiggs
    if is_mc and (dataset_name in parameters["ggh_nnlops_reweight"]):
        genHiggs_mask = NUMPY_LIB.logical_and((genpart.pdgId == 25), (genpart.status == 62))
        genHiggs_pt = genhpt(genpart, genHiggs_mask, use_cuda)
        selected_genJet_mask = genJet.pt>30
        genNjets = ha.sum_in_offsets(genJet.offsets, selected_genJet_mask, mask_events,genJet.masks["all"], NUMPY_LIB.int8)
        gghnnlopsw = analysis_corrections.nnlopsreweighting.compute(NUMPY_LIB.asnumpy(genNjets), NUMPY_LIB.asnumpy(genHiggs_pt), parameters["ggh_nnlops_reweight"][dataset_name])
        if use_cuda:
            gghnnlopsw = NUMPY_LIB.array(gghnnlopsw)

    #Find the first two genjets in the event that are not matched to gen-leptons
    mask_vbf_filter = None
    if is_mc and (dataset_name in parameters["vbf_filter"]):
        #find genleptons
        genpart_pdgid = NUMPY_LIB.abs(genpart.pdgId)
        genpart_mask = (genpart_pdgid == 11)
        genpart_mask = NUMPY_LIB.logical_or(genpart_mask, (genpart_pdgid == 13))
        genpart_mask = NUMPY_LIB.logical_or(genpart_mask, (genpart_pdgid == 15))

        genjets_not_matched_genlepton = ha.mask_deltar_first(
            {"eta": genJet.eta, "phi": genJet.phi, "offsets": genJet.offsets},
            genJet.masks["all"],
            {"eta": genpart.eta, "phi": genpart.phi, "offsets": genpart.offsets},
            genpart_mask, 0.3
        )
        out_genjet_mask = NUMPY_LIB.zeros(genJet.numobjects(), dtype=NUMPY_LIB.bool)
        inds = NUMPY_LIB.zeros_like(mask_events)
        targets = NUMPY_LIB.ones_like(mask_events)
        inds[:] = 0
        ha.set_in_offsets(genJet.offsets, out_genjet_mask, inds, targets, mask_events, genjets_not_matched_genlepton)
        inds[:] = 1
        ha.set_in_offsets(genJet.offsets, out_genjet_mask, inds, targets, mask_events, genjets_not_matched_genlepton)

        num_good_genjets = ha.sum_in_offsets(genJet.offsets, out_genjet_mask, mask_events, genJet.masks["all"], NUMPY_LIB.int8)

        genjet_inv_mass, _ = compute_inv_mass(genJet, mask_events, out_genjet_mask, use_cuda)
        genjet_inv_mass[num_good_genjets<2] = 0
        
        mask_vbf_filter = vbf_genfilter(genjet_inv_mass, num_good_genjets, parameters, dataset_name)

    #assign a numerical flag to each data event that corresponds to the data era
    assign_data_run_id(scalars, data_runs, dataset_era, is_mc, runmap_numerical)

    #Get the mask of events that pass trigger selection
    mask_events = select_events_trigger(scalars, parameters, mask_events, parameters["hlt_bits"][dataset_era])
    if not (mask_vbf_filter is None):
        mask_events = mask_events & mask_vbf_filter

    #Event weight dictionary, 2 levels.
    #systematic name -> syst_dir -> individual weight value (not multiplied up) 
    weights_individual = {}
    weights_individual["nominal"] = {"nominal": NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)}

    #Apply Rochester corrections to leading and subleading muon momenta
    if parameters["do_rochester_corrections"]:
        if debug:
            print("Before applying Rochester corrections: muons.pt={0:.2f} +- {1:.2f}".format(muons.pt.mean(), muons.pt.std()))
        do_rochester_corrections(
            is_mc,
            analysis_corrections.rochester_corrections[dataset_era],
            muons)
        if debug:
            print("After applying Rochester corrections muons.pt={0:.2f} +- {1:.2f}".format(muons.pt.mean(), muons.pt.std()))

    #get the two leading muons after applying all muon selection
    ret_mu = get_selected_muons(
        scalars,
        muons, fsrphotons, trigobj, mask_events,
        parameters["muon_pt_leading"][dataset_era], parameters["muon_pt"],
        parameters["muon_eta"], parameters["muon_iso"],
        parameters["muon_id"][dataset_era], parameters["muon_trigger_match_dr"],
        parameters["muon_iso_trigger_matched"], parameters["muon_id_trigger_matched"][dataset_era],
        parameters["fsr_dROverEt2"], parameters["fsr_relIso03"], parameters["pt_fsr_over_mu_e"],
        use_cuda
    )
    if debug:
        print("muon selection eff", ret_mu["selected_muons"].sum() / float(muons.numobjects()))

    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets before re-apply JEC")
            
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi", "mass", "jetId", "puId","qgl"])

    #re-apply JEC
    JetTransformer(
        jets, scalars,
        parameters,
        analysis_corrections.jetmet_corrections[dataset_era][parameters["jec_tag"][dataset_era]],
        NUMPY_LIB, ha, use_cuda, is_mc, dataset_era, parameters["do_jec"][dataset_era])

    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets after re-apply JEC")
            
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi", "mass", "jetId", "puId","qgl"])
    
    #jet selection
    selected_jets_id = get_selected_jets_id(
        scalars,
        jets, muons,
        parameters["jet_eta"],
        parameters["jet_mu_dr"],
        parameters["jet_id"],
        parameters["jet_puid"],
        parameters["jet_veto_eta"][0],
        parameters["jet_veto_eta"][1],
        parameters["jet_veto_raw_pt"],
        dataset_era)
    if debug:
        print("jet selection eff based on id", selected_jets_id.sum() / float(len(selected_jets_id)))

    #Now we throw away all the jets that didn't pass the ID to save time on computing JECs on them
    jets_passing_id = jets.select_objects(selected_jets_id)
    
    #Just a check to verify that there are exactly 2 muons per event
    if doverify:
        z = ha.sum_in_offsets(
            muons.offsets,
            ret_mu["selected_muons"],
            ret_mu["selected_events"],
            ret_mu["selected_muons"],
            dtype=NUMPY_LIB.int8)
        assert(NUMPY_LIB.all(z[z!=0] == 2))
        
    # Create arrays with just the leading and subleading particle contents for easier management
    mu_attrs = ["miniPFRelIso_chg", "pfRelIso03_chg", "pt", "eta", "phi", "mass", "pdgId", "nTrackerLayers", "charge", "ptErr"]

    #do geofit after the selection of jets
    if parameters["do_geofit"]:
        if debug:
            print("Before applying GeoFit corrections: muons.pt={0:.2f} +- {1:.2f}".format(muons.pt.mean(), muons.pt.std()))

        if debug:
            for evtid in debug_event_ids:
                idx = np.where(scalars["event"] == evtid)[0][0]
                print("muons")
                jaggedstruct_print(muons, idx,
                    ["pt", "eta", "phi", "charge", "dxybs"])

        do_geofit_corrections(analysis_corrections.miscvariables, muons, dataset_era)
        if debug:
            print("After applying Geofit corrections muons.pt={0:.2f} +- {1:.2f}".format(muons.pt.mean(), muons.pt.std()))
            
    if is_mc:
        mu_attrs += ["genpt"]
    leading_muon = muons.select_nth(0, ret_mu["selected_events"], ret_mu["selected_muons"], attributes=mu_attrs)
    subleading_muon = muons.select_nth(1, ret_mu["selected_events"], ret_mu["selected_muons"], attributes=mu_attrs)

    if doverify:
        assert(NUMPY_LIB.all(leading_muon["pt"][leading_muon["pt"]>0] > parameters["muon_pt_leading"][dataset_era]))
        assert(NUMPY_LIB.all(subleading_muon["pt"][subleading_muon["pt"]>0] > parameters["muon_pt"]))

    #Compute lepton scale factors
    if parameters["do_lepton_sf"] and is_mc:
        lepton_sf_values = compute_lepton_sf(leading_muon, subleading_muon,
            analysis_corrections.lepsf_iso[dataset_era], analysis_corrections.lepsf_id[dataset_era], analysis_corrections.lepeff_trig_data[dataset_era],
            analysis_corrections.lepeff_trig_mc[dataset_era], use_cuda, dataset_era, NUMPY_LIB, debug)
        weights_individual["trigger"] = {
            "nominal": lepton_sf_values["trigger"],
            "up": lepton_sf_values["trigger__up"], 
            "down": lepton_sf_values["trigger__down"]
        }
        weights_individual["id"] = {
            "nominal": lepton_sf_values["id"],
            "up": lepton_sf_values["id__up"], 
            "down": lepton_sf_values["id__down"]
        }
        weights_individual["iso"] = {
            "nominal": lepton_sf_values["iso"],
            "up": lepton_sf_values["iso__up"], 
            "down": lepton_sf_values["iso__down"]
        }
        weights_individual["mu1_id"] = {
            "nominal": lepton_sf_values["mu1_id"]
        }
        weights_individual["mu1_iso"] = {
            "nominal": lepton_sf_values["mu1_iso"]
        }
        weights_individual["mu2_id"] = {
            "nominal": lepton_sf_values["mu2_id"]
        }
        weights_individual["mu2_iso"] = {
            "nominal": lepton_sf_values["mu2_iso"]
        }
        if doverify:
            for w in ["trigger", "id", "iso"]:
                m1 = weights_individual[w]["nominal"].mean()
                m2 = weights_individual[w]["up"].mean()
                m3 = weights_individual[w]["down"].mean()
                assert(m1 > m3 and m1 < m2)
    else:
        #set default weights to 1
        def default_weight(n):
            return {
                "nominal": NUMPY_LIB.ones(n, dtype=NUMPY_LIB.float32),
                "up": NUMPY_LIB.ones(n, dtype=NUMPY_LIB.float32),
                "down": NUMPY_LIB.ones(n, dtype=NUMPY_LIB.float32)
            }
        weights_individual["trigger"] = default_weight(len(leading_muon["pt"])) 
        weights_individual["id"] = default_weight(len(leading_muon["pt"])) 
        weights_individual["iso"] = default_weight(len(leading_muon["pt"])) 
        weights_individual["mu1_id"] = default_weight(len(leading_muon["pt"]))
        weights_individual["mu1_iso"] = default_weight(len(leading_muon["pt"]))
        weights_individual["mu2_id"] = default_weight(len(leading_muon["pt"]))
        weights_individual["mu2_iso"] = default_weight(len(leading_muon["pt"]))
            
    # Get the selected electrons
    ret_el = get_selected_electrons(electrons, parameters["extra_electrons_pt"], parameters["extra_electrons_eta"], parameters["extra_electrons_id"])
     
    # Get the invariant mass of the dimuon system and compute mass windows
    higgs_inv_mass, higgs_pt = compute_inv_mass(muons, ret_mu["selected_events"], ret_mu["selected_muons"], use_cuda)
    higgs_inv_mass[NUMPY_LIB.isnan(higgs_inv_mass)] = -1
    higgs_inv_mass[NUMPY_LIB.isinf(higgs_inv_mass)] = -1
    higgs_inv_mass[higgs_inv_mass==0] = -1
    higgs_pt[NUMPY_LIB.isnan(higgs_pt)] = -1
    higgs_pt[NUMPY_LIB.isinf(higgs_pt)] = -1
    higgs_pt[higgs_pt==0] = -1
   
    #Z pT reweighting for DY bkg (CPU only)
    ZpTw = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.float32)
    if is_mc and (dataset_name in parameters["ZpT_reweight"][dataset_era]):
       ZpTw = NUMPY_LIB.array(analysis_corrections.zptreweighting.compute(
           NUMPY_LIB.asnumpy(higgs_pt), parameters["ZpT_reweight"][dataset_era][dataset_name]))

    #Do the jet ID selection and lepton cleaning just once for the nominal jet systematic
    #as that does not depend on jet pt

    jet_attrs = ["pt"]
    #temp_subjet = jets_passing_id.select_nth(
    #            1, ret_mu['selected_events'], temp_ret_jet["selected_jets"],
    #            attributes=jet_attrs)

    # PU ID weights are only applied to 2016 and 2018 so far, as they haven't been validated for 2017
    # https://github.com/jpata/hepaccelerate-cms/pull/66
    #if (parameters["jet_puid"] != "none") and is_mc:
    #    puid_weights = get_puid_weights(jets_wopuid, analysis_corrections.puidreweighting, dataset_era, parameters["jet_puid"], temp_subjet["pt"], parameters["jet_pt_subleading"][dataset_era], parameters["jet_puid_pt_max"], use_cuda)
    #    weights_individual["jet_puid"] = {"nominal": puid_weights, "up": puid_weights, "down": puid_weights}
    if is_mc and parameters["apply_btag"]:
        btagWeights, btagWeights_up, btagWeights_down = get_factorized_btag_weights_shape(jets_passing_id, analysis_corrections.btag_weights, dataset_era, scalars, parameters["jet_pt_subleading"][dataset_era])
        weights_individual["btag_weight"] = {"nominal": btagWeights, "up": NUMPY_LIB.ones_like(btagWeights), "down": NUMPY_LIB.ones_like(btagWeights)}
        for i in range(len(btag_unc)):
            weights_individual[btag_unc[i]] = {"nominal": NUMPY_LIB.ones_like(btagWeights), "up": btagWeights_up[i], "down": btagWeights_down[i]}
    #compute variated weights here to ensure the nominal weight contains all possible other weights  
    compute_event_weights(parameters, weights_individual, scalars,
        genweight_scalefactor, gghnnlopsw, ZpTw,
        LHEScalew, LHEPdfw, analysis_corrections.pu_corrections, is_mc, dataset_era, dataset_name, use_cuda)
 
    jet_attrs += ["eta", "qgl"]
    if is_mc:
        jet_attrs += ["partonFlavour"]
        qglWeights, qglWeights_up, qglWeights_down = get_qglWeights(jets_passing_id, jet_attrs, ret_mu, analysis_corrections.miscvariables, dataset_name)
        weights_individual["qgl_weight"] = {"nominal": qglWeights, "up": qglWeights_up, "down": qglWeights_down}

        if "vbf_powheg_pythia_dipole_125" in dataset_name:
            for stxs_unc_name in VBF_STXS_unc:
                stxs_nominal = NUMPY_LIB.ones_like(qglWeights)
                stxs_up = NUMPY_LIB.ones_like(stxs_nominal)
                stxs_down = NUMPY_LIB.ones_like(stxs_nominal)
                stxs_unc = NUMPY_LIB.array(HSTXS_rel[stxs_unc_name])
                compute_stxs_unc(scalars["HTXS_stage1_1_fine_cat_pTjet25GeV"], stxs_unc, stxs_up, stxs_down)
                weights_individual[stxs_unc_name] = {"nominal": stxs_nominal, "up": stxs_up, "down": stxs_down}

    #actually multiply all the weights together with the appropriate up/down variations.
    #creates a 1-level dictionary with weights "nominal", "puweight__up", "puweight__down", ..." 
    weights_final = finalize_weights(weights_individual, dataset_era)
    '''
    if parameters["do_lepton_sf"] and is_mc:
        lepton_sf_values = compute_lepton_sf(leading_muon, subleading_muon,
            lepsf_iso[dataset_era], lepsf_id[dataset_era], lepeff_trig_data[dataset_era],
            lepeff_trig_mc[dataset_era], use_cuda, dataset_era, NUMPY_LIB, debug)
        weights_individual["mu1_id"] = {
            "nominal": lepton_sf_values["mu1_id"]
        }
        weights_individual["mu1_iso"] = {
            "nominal": lepton_sf_values["mu1_iso"]
        }
        weights_individual["mu2_id"] = {
            "nominal": lepton_sf_values["mu2_id"]
        }
        weights_individual["mu2_iso"] = {
            "nominal": lepton_sf_values["mu2_iso"]
        }
    '''

        
    fill_histograms_several(
        hists, "nominal", "hist__dimuon__",
        [
            (leading_muon["pt"], "leading_muon_pt", histo_bins["muon_pt"]),
            (subleading_muon["pt"], "subleading_muon_pt", histo_bins["muon_pt"]),
            (leading_muon["pt"], "leading_muon_eta", histo_bins["muon_eta"]),
            (subleading_muon["pt"], "subleading_muon_eta", histo_bins["muon_eta"]),
            (higgs_inv_mass, "inv_mass", histo_bins["inv_mass"]),
            (scalars["PV_npvsGood"], "npvs", histo_bins["npvs"])
        ],
        ret_mu["selected_events"],
        weights_final,
        use_cuda
    )
    ret["selected_events_dimuon"] = NUMPY_LIB.sum(ret_mu["selected_events"])
    masswindow_z_peak = ((higgs_inv_mass >= parameters["masswindow_z_peak"][0]) & (higgs_inv_mass < parameters["masswindow_z_peak"][1]))
    masswindow_h_region = ((higgs_inv_mass >= parameters["masswindow_h_sideband"][0]) & (higgs_inv_mass < parameters["masswindow_h_sideband"][1]))
    masswindow_h_peak = ((higgs_inv_mass >= parameters["masswindow_h_peak"][0]) & (higgs_inv_mass < parameters["masswindow_h_peak"][1]))
    masswindow_h_sideband = masswindow_h_region & NUMPY_LIB.invert(masswindow_h_peak)
    #masswindow_z_peak_jer = ((higgs_inv_mass >= parameters["masswindow_z_peak"][0]) & (higgs_inv_mass < parameters["masswindow_z_peak"][1]))
    #get the number of additional muons (not OS) that pass ID and iso cuts
    n_additional_muons = ha.sum_in_offsets(muons.offsets, ret_mu["additional_muon_sel"], ret_mu["selected_events"], ret_mu["additional_muon_sel"], dtype=NUMPY_LIB.int8)
    n_additional_electrons = ha.sum_in_offsets(electrons.offsets, ret_el["additional_electron_sel"], ret_mu["selected_events"], ret_el["additional_electron_sel"], dtype=NUMPY_LIB.int8)

    #This computes the JEC, JER and associated systematics
    if debug:
        print("event selection eff based on 2 muons", ret_mu["selected_events"].sum() / float(len(mask_events)))
        print("Doing nominal jec on {0} jets".format(jets_passing_id.numobjects()))

    jet_systematics = JetTransformer(
        jets_passing_id, scalars,
        parameters,
        analysis_corrections.jetmet_corrections[dataset_era][parameters["jec_tag"][dataset_era]],
        NUMPY_LIB, ha, use_cuda, is_mc, dataset_era, parameters["do_jec"][dataset_era])

    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets_passing_id for systematic loop")
            
            jaggedstruct_print(jets_passing_id, idx,
                               ["pt", "eta", "phi", "mass", "jetId", "puId","qgl"])

    syst_to_consider = ["nominal"]
    if is_mc:
        syst_to_consider += ["Total"]
        if parameters["do_jer"][dataset_era]: 
            jer_syst = jer_unc
            syst_to_consider = syst_to_consider + jer_syst
        if parameters["do_factorized_jec"]:
            syst_to_consider = syst_to_consider + jet_systematics.jet_uncertainty_names

    if debug:
        print("entering jec loop with {0}".format(syst_to_consider))

    #Now actually call the JEC computation for each scenario
    jet_pt_startfrom = "pt_jec"
            
    for uncertainty_name in syst_to_consider:
        # First get the jet mask each uncertainty_name
        pass_jer_bin = NUMPY_LIB.ones_like(ret_mu['selected_events'])
        if 'jer' in uncertainty_name:
            ret_jet_temp = get_selected_jets(
                scalars,
                jets_passing_id,
                ret_mu['selected_events'],
                parameters["jet_pt_subleading"][dataset_era],
                parameters["jet_btag_medium"][dataset_era],
                parameters["jet_btag_loose"][dataset_era],
                is_mc, parameters["do_jec"][dataset_era], debug, use_cuda
            )
            j_attrs = ["pt", "eta", "phi"]
            temp_subleading_jet = jets_passing_id.select_nth(
                1, ret_mu['selected_events'], ret_jet_temp["selected_jets"],
                attributes=j_attrs)
                
            j2_eta_abs = NUMPY_LIB.abs(temp_subleading_jet["eta"])
            pass_jer_bin = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"][uncertainty_name]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"][uncertainty_name]["eta"][1], NUMPY_LIB.logical_and(temp_subleading_jet["pt"] > parameters["jer_pt_eta_bins"][uncertainty_name]["pt"][0],temp_subleading_jet["pt"] < parameters["jer_pt_eta_bins"][uncertainty_name]["pt"][1])))
        #calculate the associated genpt for every reco jet
        jet_genpt = NUMPY_LIB.zeros(jets_passing_id.numobjects(), dtype=NUMPY_LIB.float32)
        if is_mc:
            get_genJetpt_cpu(jets_passing_id.offsets, jets_passing_id.pt, jets_passing_id.genJetIdx, genJet.offsets, genJet.pt, jet_genpt)

        is_jer_event = NUMPY_LIB.logical_and(ret_mu["selected_events"],pass_jer_bin)
        jet_mask_bin = NUMPY_LIB.zeros_like(jets_passing_id.pt)
        ha.broadcast(jets_passing_id.offsets,is_jer_event,jet_mask_bin)
        #This will be the variated pt vector
        #print("computing variated pt for", uncertainty_name)
        var_up_down = jet_systematics.get_variated_pts(uncertainty_name, jet_mask_bin, jet_genpt, startfrom=jet_pt_startfrom)

        for jet_syst_name, jet_pt_vec in var_up_down.items():
            if 'jer' in uncertainty_name:
                jet_syst_name = (uncertainty_name,jet_syst_name[1])
                
            # For events where the JEC/JER was variated, fill only the nominal weight
            weights_selected = select_weights(weights_final, jet_syst_name)
            jet_pt_change = (jet_pt_vec - jets_passing_id.pt).mean()
            
            # Configure the jet pt vector to the variated one
            # Would need to also do the mass here
            jets_passing_id.pt = jet_pt_vec
                    
            #Do the pt-dependent jet analysis now for all jets
            ret_jet = get_selected_jets(
                scalars,
                jets_passing_id,
                ret_mu['selected_events'],
                parameters["jet_pt_subleading"][dataset_era],
                parameters["jet_btag_medium"][dataset_era],
                parameters["jet_btag_loose"][dataset_era],
                is_mc, parameters["do_jec"][dataset_era], debug, use_cuda
            )
            if debug:
                print("jet analysis syst={0} sdir={1} mean_pt_change={2:.4f} num_passing_jets={3} ".format(
                    jet_syst_name[0], jet_syst_name[1], float(jet_pt_change), int(ret_jet["selected_jets"].sum()))
                )
            fill_histograms_several(
                hists, "nominal", "hist__dimuon__",
                [
                    (ret_jet["num_jets"], "num_jets" , histo_bins["numjets"]),
                ],
                ret_mu['selected_events'],
                weights_final,
                use_cuda
            )

            #print("jet selection eff based on ID & pt", ret_jet["selected_jets"].sum() / float(len(ret_jet["selected_jets"])))

            pt_balance = ret_jet["dijet_pt"] / higgs_pt

            # Set this default value as in Nan and Irene's code
            ret_jet["dijet_inv_mass"][ret_jet["num_jets"] < 2] = -1000.0
            # Get the data for the leading and subleading jets as contiguous vectors
            jet_attrs += ["phi", "mass","jetId","puId","btagDeepB"]
            if is_mc:
                jet_attrs += ["hadronFlavour", "genJetIdx"]
            #get the index of the leading and subleading jets
            out_jet_ind0 = NUMPY_LIB.zeros(len(jets_passing_id.offsets)-1, dtype=NUMPY_LIB.int8)
            out_jet_ind1 = NUMPY_LIB.ones(len(jets_passing_id.offsets)-1, dtype=NUMPY_LIB.int8)
            if parameters["do_jec"][dataset_era]:
                get_leadtwo_jet_ind(jets_passing_id.offsets, jets_passing_id.pt, out_jet_ind0, out_jet_ind1)
            
            if debug:
                    for evtid in debug_event_ids:
                        idx = np.where(scalars["event"] == evtid)[0][0]
                        print("leading jet index: ",out_jet_ind0[idx])
                        print("subleading jet index: ",out_jet_ind1[idx])

            leading_jet = jets_passing_id.select_nth(
                out_jet_ind0, ret_mu['selected_events'], ret_jet["selected_jets"],
                attributes=jet_attrs)
            subleading_jet = jets_passing_id.select_nth(
                out_jet_ind1, ret_mu['selected_events'], ret_jet["selected_jets"],
                attributes=jet_attrs)
            #if do_sync and jet_syst_name[0] == "nominal":
                #sync_printout(ret_mu, muons, scalars,
                   # leading_muon, subleading_muon, higgs_inv_mass,
                   # n_additional_muons, n_additional_electrons,
                    #ret_jet, leading_jet, subleading_jet)
          
            if parameters["split_z_peak"][dataset_era]:
                j2_eta_abs = NUMPY_LIB.abs(subleading_jet["eta"])
                pass_jerB1 = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"]["jerB1"]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"]["jerB1"]["eta"][1], subleading_jet["pt"] > parameters["jer_pt_eta_bins"]["jerB1"]["pt"]))
                pass_jerB2 = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"]["jerB2"]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"]["jerB2"]["eta"][1], subleading_jet["pt"] > parameters["jer_pt_eta_bins"]["jerB2"]["pt"]))
                pass_jerF1 = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"]["jerF1"]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"]["jerF1"]["eta"][1], subleading_jet["pt"] > parameters["jer_pt_eta_bins"]["jerF1"]["pt"]))
                pass_jerF2 = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"]["jerF2"]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"]["jerF2"]["eta"][1], subleading_jet["pt"] > parameters["jer_pt_eta_bins"]["jerF2"]["pt"]))
                pass_jerEC1 = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"]["jerEC1"]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"]["jerEC1"]["eta"][1], subleading_jet["pt"] > parameters["jer_pt_eta_bins"]["jerEC1"]["pt"]))
                pass_jerEC2 = NUMPY_LIB.logical_and(j2_eta_abs > parameters["jer_pt_eta_bins"]["jerEC2"]["eta"][0], NUMPY_LIB.logical_and(j2_eta_abs < parameters["jer_pt_eta_bins"]["jerEC2"]["eta"][1], subleading_jet["pt"] > parameters["jer_pt_eta_bins"]["jerEC2"]["pt"]))
                
                is_jerB1_event = NUMPY_LIB.logical_and(ret_mu['selected_events'],pass_jerB1)
                is_jerB2_event = NUMPY_LIB.logical_and(ret_mu['selected_events'],pass_jerB2)
                is_jerF1_event = NUMPY_LIB.logical_and(ret_mu['selected_events'],pass_jerF1)
                is_jerF2_event = NUMPY_LIB.logical_and(ret_mu['selected_events'],pass_jerF2)
                is_jerEC1_event = NUMPY_LIB.logical_and(ret_mu['selected_events'],pass_jerEC1)
                is_jerEC2_event = NUMPY_LIB.logical_and(ret_mu['selected_events'],pass_jerEC2)

                masswindow_z_peak_jerB1 = (masswindow_z_peak) & (is_jerB1_event) #split Z into 6 regions by j2 pt and eta
                masswindow_z_peak_jerB2 = (masswindow_z_peak) & (is_jerB2_event)
                masswindow_z_peak_jerF1 = (masswindow_z_peak) & (is_jerF1_event)
                masswindow_z_peak_jerF2 = (masswindow_z_peak) & (is_jerF2_event)
                masswindow_z_peak_jerEC1 = (masswindow_z_peak) & (is_jerEC1_event)
                masswindow_z_peak_jerEC2 = (masswindow_z_peak) & (is_jerEC2_event)
        
            #compute Nsoft jet variable 5 by removing event footprints
            n_sel_softjet, n_sel_HTsoftjet = nsoftjets(False,
                scalars["SoftActivityJetNjets5"], scalars["SoftActivityJetHT5"],
                muons.numevents(), softjets, leading_muon, subleading_muon,
                leading_jet, subleading_jet, parameters["softjet_pt5"],
                parameters["softjet_evt_dr2"], use_cuda)

            #compute Nsoft jet variable 2 by removing event footprints
            n_sel_softjet2, n_sel_HTsoftjet2 = nsoftjets(True,
                scalars["SoftActivityJetNjets2"], scalars["SoftActivityJetHT2"],
                muons.numevents(), softjets, leading_muon, subleading_muon,
                leading_jet, subleading_jet, parameters["softjet_pt2"],
                parameters["softjet_evt_dr2"], use_cuda)

            #compute DNN input variables in 2 muon, >=2jet region
            dnn_presel = (
                (ret_mu['selected_events']) & (ret_jet["num_jets"] >= 2) &
                #(ret_mu['selected_events']) & (ret_jet["num_jets"] >= 2) & (ret_jet["dijet_inv_mass"]>300) &  
                (leading_jet["pt"] > parameters["jet_pt_leading"][dataset_era])
            )
            if is_mc and 'dy_m105_160' in dataset_name:
                leading_jet_offset = NUMPY_LIB.arange(0,len(leading_jet["pt"])+1)
                
                leading_jets_matched_to_genJet = leading_jet["genJetIdx"]>=0
                subleading_jets_matched_to_genJet = subleading_jet["genJetIdx"]>=0
                
                if debug:
                    for evtid in debug_event_ids:
                        idx = np.where(scalars["event"] == evtid)[0][0]
                        print("leading jets")
                        for e in ["pt", "eta", "phi", "mass"]:
                            print(e, leading_jet[e][idx])

                        print("subleading jets")
                        for e in ["pt", "eta", "phi", "mass"]:
                            print(e, subleading_jet[e][idx])
                        print("gen jets")

                        jaggedstruct_print(genJet, idx,
                               ["pt", "eta", "phi", "mass"])

                        print(leading_jets_matched_to_genJet[idx],subleading_jets_matched_to_genJet[idx])
                        
                        print(ret_jet["num_jets"][idx], ret_jet["num_jets_btag_medium"][idx],ret_jet["num_jets_btag_loose"][idx],n_additional_muons[idx], n_additional_electrons[idx], ret_jet["dijet_inv_mass"][idx],NUMPY_LIB.abs(leading_jet["eta"][idx] - subleading_jet["eta"][idx]))
            
                jets_vbf_filter = NUMPY_LIB.logical_and(leading_jets_matched_to_genJet,subleading_jets_matched_to_genJet)
                if '_2j' in dataset_name:
                    dnn_presel = NUMPY_LIB.logical_and(dnn_presel,jets_vbf_filter)
                    
                else:
                    dnn_presel = NUMPY_LIB.logical_and(dnn_presel,NUMPY_LIB.invert(jets_vbf_filter))
                    
            
            #Histograms after dnn preselection
            fill_histograms_several(
                hists, jet_syst_name, "hist__dnn_presel__",
                [
                    (leading_jet["pt"], "leading_jet_pt", histo_bins["jet_pt"]),
                    (subleading_jet["pt"], "subleading_jet_pt", histo_bins["jet_pt"]),
                    (leading_jet["eta"], "leading_jet_eta", histo_bins["jet_eta"]),
                    (subleading_jet["eta"], "subleading_jet_eta", histo_bins["jet_eta"]),
                    (leading_jet["qgl"], "leading_jet_qgl", histo_bins["jet_qgl"]),
                    (subleading_jet["qgl"], "subleading_jet_qgl", histo_bins["jet_qgl"]),
                    (ret_jet["dijet_inv_mass"], "dijet_inv_mass", histo_bins["dijet_inv_mass"]),
                    (higgs_inv_mass, "inv_mass", histo_bins["inv_mass"]),
                    (scalars["SoftActivityJetNjets5"], "num_soft_jets", histo_bins["numjets"]),
                    (ret_jet["num_jets"], "num_jets" , histo_bins["numjets"]),
                    (pt_balance, "pt_balance", histo_bins["pt_balance"]),
                    #(leading_jet["btagDeepB"], "leading_jet_DeepCSV", histo_bins["DeepCSV"]),
                    #(subleading_jet["btagDeepB"], "subleading_jet_DeepCSV", histo_bins["DeepCSV"]),
                ],
                dnn_presel, 
                weights_selected,
                use_cuda
            )

            #Compute the DNN inputs, the DNN output, fill the DNN input and output variable histograms
            dnn_prediction = None
            dnn_vars, dnn_prediction, dnnPisa_predictions, dnnPisaComb_pred = compute_fill_dnn(analysis_corrections.hrelresolution,
               analysis_corrections.miscvariables, parameters, use_cuda, dnn_presel,
               analysis_corrections.dnn_model, analysis_corrections.dnn_normfactors,
               analysis_corrections.dnnPisa_models, analysis_corrections.dnnPisa_normfactors1, analysis_corrections.dnnPisa_normfactors2,
               scalars, leading_muon, subleading_muon, leading_jet, subleading_jet,
               ret_jet["num_jets"],ret_jet["num_jets_btag_medium"], n_sel_softjet, n_sel_HTsoftjet, n_sel_HTsoftjet2, mass_point, dataset_name, dataset_era, is_mc
            )
            weights_in_dnn_presel = apply_mask(weights_selected, dnn_presel)
            
            if parameters["do_bdt_ucsd"]: 
                if not (analysis_corrections.bdt_ucsd is None):
                    bdt_pred = evaluate_bdt_ucsd(dnn_vars, analysis_corrections.bdt_ucsd)
                    dnn_vars["bdt_ucsd"] = NUMPY_LIB.array(bdt_pred, dtype=NUMPY_LIB.float32)
                #if not ((bdt2j_ucsd is None)):
                #    bdt2j_pred = evaluate_bdt2j_ucsd(dnn_vars, bdt2j_ucsd[dataset_era])
                #    dnn_vars["bdt2j_ucsd"] = bdt2j_pred
                #if not ((bdt01j_ucsd is None)):
                #    bdt01j_pred = evaluate_bdt01j_ucsd(dnn_vars, bdt01j_ucsd[dataset_era])
                #    dnn_vars["bdt01j_ucsd"] = bdt01j_pred

            #Assing a numerical category ID 
            category =  assign_category(
                ret_jet["num_jets"], ret_jet["num_jets_btag_medium"],ret_jet["num_jets_btag_loose"],
                n_additional_muons, n_additional_electrons,
                ret_jet["dijet_inv_mass"],
                leading_jet, subleading_jet,
                parameters["cat5_dijet_inv_mass"],
                parameters["cat5_abs_jj_deta_cut"]
            )
            scalars["category"] = category
                    
            #Assign the final analysis discriminator based on category
            #scalars["final_discriminator"] = NUMPY_LIB.zeros_like(higgs_inv_mass)
            if not (dnn_prediction is None):
                #Add some additional debugging info to the DNN training ntuples
                dnn_vars["cat_index"] = category[dnn_presel]
                dnn_vars["run"] = scalars["run"][dnn_presel]
                dnn_vars["lumi"] = scalars["luminosityBlock"][dnn_presel]
                dnn_vars["event"] = scalars["event"][dnn_presel]
                dnn_vars["dnn_pred"] = dnn_prediction
                #print(weights_individual['trigger']['nominal'].shape)
                #print(dnn_presel.shape)
                if is_mc:
                    dnn_vars["trig_weight"] = weights_individual['trigger']['nominal'][dnn_presel]
                    dnn_vars["L1PreFiringWeight"] = weights_individual['L1PreFiringWeight']['nominal'][dnn_presel]
                    dnn_vars["puWeight"] = weights_individual['puWeight']['nominal'][dnn_presel]
                    dnn_vars["muidWeight"] = weights_individual['id']['nominal'][dnn_presel]*weights_individual['iso']['nominal'][dnn_presel]
                    dnn_vars["m1_id"] = weights_individual['mu1_id']['nominal'][dnn_presel]
                    dnn_vars["m1_iso"] = weights_individual['mu1_iso']['nominal'][dnn_presel]
                    dnn_vars["m2_id"] = weights_individual['mu2_id']['nominal'][dnn_presel]
                    dnn_vars["m2_iso"] = weights_individual['mu2_iso']['nominal'][dnn_presel]
                    dnn_vars["qgl_weight"] = weights_individual['qgl_weight']['nominal'][dnn_presel]
                    if parameters["apply_btag"]:
                        dnn_vars["btag_weight"] = weights_individual['btag_weight']['nominal'][dnn_presel]
                    #if parameters["jet_puid"] != "none":
                    #    dnn_vars["puid_weight"] = weights_individual['jet_puid']['nominal'][dnn_presel]
                    dnn_vars["j1_partonFlavour"] = leading_jet["partonFlavour"][dnn_presel]
                    dnn_vars["j2_partonFlavour"] = subleading_jet["partonFlavour"][dnn_presel]
                dnn_vars["j1_jetId"] = leading_jet["jetId"][dnn_presel]
                dnn_vars["j1_puId"] = leading_jet["puId"][dnn_presel]
                dnn_vars["j2_jetId"] =subleading_jet["jetId"][dnn_presel]
                dnn_vars["j2_puId"] = subleading_jet["puId"][dnn_presel]
                dnn_vars["nmuons"] = n_additional_muons[dnn_presel]
                dnn_vars["nelectrons"] = n_additional_electrons[dnn_presel]
                dnn_vars["Nbjet_med"] = ret_jet["num_jets_btag_medium"][dnn_presel]
                dnn_vars["Nbjet_loose"] = ret_jet["num_jets_btag_loose"][dnn_presel]
                dnn_vars["Njet_loose"] = ret_jet["num_jets"][dnn_presel]
                if not (len(dnnPisa_predictions)==0):
                    mass_index = 0
                    for imass in mass_point:
                        dnn_vars["dnnPisa_pred_"+str(imass)] = dnnPisaComb_pred[mass_index]
                        for imodel in range(int(len(dnnPisa_predictions)/len(mass_point))):
                            dnn_vars["dnnPisa_pred_"+str(imass)+str(imodel)] = dnnPisa_predictions[len(mass_point)*imodel+mass_index]
                        mass_index += 1
                #Save the DNN training ntuples as npy files
                if parameters["save_dnn_vars"] and jet_syst_name[0] == "nominal" and parameter_set_name == "baseline":
                    dnn_vars_np = {k: NUMPY_LIB.asnumpy(v) for k, v in dnn_vars.items()}
                    if is_mc:
                        dnn_vars_np["nomweight"] = NUMPY_LIB.asnumpy(weights_in_dnn_presel["nominal"]/genweight_scalefactor)
                        dnn_vars_np["genweight"] = NUMPY_LIB.asnumpy(scalars["genWeight"][dnn_presel])
                        if "dy" in dataset_name or "ewk" in dataset_name:
                            for iScale in range(9):
                                dnn_vars_np["LHEScaleWeight__"+str(iScale)] = NUMPY_LIB.asnumpy(weights_in_dnn_presel["LHEScaleWeight__"+str(iScale)])
                    arrs = []
                    names = []
                    for k, v in dnn_vars_np.items():
                        #print(k,v.shape)
                        arrs += [v]
                        names += [k]
                    arrdata = np.core.records.fromarrays(arrs, names=names)
                    outpath = "{0}/{1}".format(parameters["dnn_vars_path"], dataset_era) 
                    if not os.path.isdir(outpath):
                        os.makedirs(outpath)
                    np.save("{0}/{1}_{2}.npy".format(outpath, dataset_name, data.num_chunk), arrdata, allow_pickle=False)

            mbins = [
                    ("h_peak", masswindow_h_peak, parameters["masswindow_h_peak"]),
                    ("h_sideband", masswindow_h_sideband, parameters["masswindow_h_sideband"]),
                    ("z_peak", masswindow_z_peak, parameters["masswindow_z_peak"]),
            ]
            if (parameters["split_z_peak"][dataset_era]):
                mbins += [("z_peak_{0}".format("jerB1"), masswindow_z_peak_jerB1, parameters["masswindow_z_peak"]),
                          ("z_peak_{0}".format("jerB2"), masswindow_z_peak_jerB2, parameters["masswindow_z_peak"]),
                          ("z_peak_{0}".format("jerF1"), masswindow_z_peak_jerF1, parameters["masswindow_z_peak"]),
                          ("z_peak_{0}".format("jerF2"), masswindow_z_peak_jerF2, parameters["masswindow_z_peak"]),
                          ("z_peak_{0}".format("jerEC1"), masswindow_z_peak_jerEC1, parameters["masswindow_z_peak"]),
                          ("z_peak_{0}".format("jerEC2"), masswindow_z_peak_jerEC2, parameters["masswindow_z_peak"]),
                      ]
                
                
            #Save histograms for numerical categories (cat5 only right now) and all mass bins
            for massbin_name, massbin_msk, mass_edges in mbins:

                for icat in [5, ]:
                    msk_cat = (category == icat)
                    fill_histograms_several(
                        hists, jet_syst_name, "hist__dimuon_invmass_{0}_cat{1}__".format(massbin_name, icat),
                        [
                            (higgs_inv_mass, "inv_mass", histo_bins["inv_mass_{0}".format(massbin_name)]),
                            (leading_jet["pt"], "leading_jet_pt", histo_bins["jet_pt"]),
                            (subleading_jet["pt"], "subleading_jet_pt", histo_bins["jet_pt"]),
                            (leading_jet["eta"], "leading_jet_eta", histo_bins["jet_eta"]),
                            (subleading_jet["eta"], "subleading_jet_eta", histo_bins["jet_eta"]),
                            (ret_jet["dijet_inv_mass"], "dijet_inv_mass", histo_bins["dijet_inv_mass"]),
                            (scalars["SoftActivityJetNjets5"], "num_soft_jets", histo_bins["numjets"]),
                            (ret_jet["num_jets"], "num_jets" , histo_bins["numjets"]),
                            (pt_balance, "pt_balance", histo_bins["pt_balance"]),
                            (leading_jet["btagDeepB"], "leading_jet_DeepCSV", histo_bins["DeepCSV"]),
                            (subleading_jet["btagDeepB"], "subleading_jet_DeepCSV", histo_bins["DeepCSV"]),
                        ],
                        (dnn_presel & massbin_msk & msk_cat),
                        weights_selected,
                        use_cuda
                    )
                    for imass in mass_point:
                        fill_histograms_several(
                            hists, jet_syst_name, "hist__dimuon_invmass_{0}_cat{1}__".format(massbin_name, icat),
                            [(dnn_vars["dnnPisa_pred_"+str(imass)], "dnnPisa_pred_atanh_"+str(imass), histo_bins["dnnPisa_pred_atanh"][dataset_era][massbin_name])
                            ],
                            (dnn_presel & massbin_msk & msk_cat)[dnn_presel],
                            weights_in_dnn_presel,
                            use_cuda
                        )
                    #end of mass loop

                    fill_histograms_several(
                        hists, jet_syst_name, "hist__dimuon_invmass_{0}_cat{1}__".format(massbin_name, icat),
                        [
                            (dnn_vars[varname], varname, histo_bins[varname])
                            for varname in dnn_vars.keys() if varname in histo_bins.keys()
                        ] + [
                            (dnn_vars["dnn_pred"], "dnn_pred2", histo_bins["dnn_pred2"][massbin_name])
                        ],
                        (dnn_presel & massbin_msk & msk_cat)[dnn_presel],
                        weights_in_dnn_presel,
                        use_cuda
                    )
             #end of mass loop
         #end of isyst loop
    #end of uncertainty_name loop

    for histname, r in hists.items():
        ret[histname] = Results(r)

    if use_cuda:
        from numba import cuda
        cuda.synchronize()
 
    return ret

def fill_histograms_several(hists, systematic_name, histname_prefix, variables, mask, weights, use_cuda):
    all_arrays = []
    all_bins = []
    num_histograms = len(variables)

    for array, varname, bins in variables:
        if len(array) != len(variables[0][0]) or len(array) != len(mask) or len(array) != len(weights["nominal"]):
            raise Exception("Data array {0} is of incompatible size".format(varname))
        all_arrays += [array]
        all_bins += [bins]

    max_bins = max([b.shape[0] for b in all_bins])
    stacked_array = NUMPY_LIB.stack(all_arrays, axis=0)
    stacked_bins = np.concatenate(all_bins)
    nbins = np.array([len(b) for b in all_bins])
    nbins_sum = np.cumsum(nbins)
    nbins_sum = np.insert(nbins_sum, 0, [0])

    for weight_name, weight_array in weights.items():
        if use_cuda:
            nblocks = 32
            out_w = NUMPY_LIB.zeros((len(variables), nblocks, max_bins), dtype=NUMPY_LIB.float32)
            out_w2 = NUMPY_LIB.zeros((len(variables), nblocks, max_bins), dtype=NUMPY_LIB.float32)
            ha.fill_histogram_several[nblocks, 1024](
                stacked_array, weight_array, mask, stacked_bins,
                NUMPY_LIB.array(nbins), NUMPY_LIB.array(nbins_sum), out_w, out_w2
            )
            cuda.synchronize()

            out_w = out_w.sum(axis=1)
            out_w2 = out_w2.sum(axis=1)

            out_w = NUMPY_LIB.asnumpy(out_w)
            out_w2 = NUMPY_LIB.asnumpy(out_w2)
        else:
            out_w = NUMPY_LIB.zeros((len(variables), max_bins), dtype=NUMPY_LIB.float32)
            out_w2 = NUMPY_LIB.zeros((len(variables), max_bins), dtype=NUMPY_LIB.float32)
            ha.fill_histogram_several(
                stacked_array, weight_array, mask, stacked_bins,
                nbins, nbins_sum, out_w, out_w2
            )

        out_w_separated = [out_w[i, 0:nbins[i]-1] for i in range(num_histograms)]
        out_w2_separated = [out_w2[i, 0:nbins[i]-1] for i in range(num_histograms)]

        for ihist in range(num_histograms):
            hist_name = histname_prefix + variables[ihist][1]
            bins = variables[ihist][2]
            target_histogram = Histogram(out_w_separated[ihist], out_w2_separated[ihist], bins)
            target = {weight_name: target_histogram}
            update_histograms_systematic(hists, hist_name, systematic_name, target)
    
def compute_integrated_luminosity(analyzed_runs, analyzed_lumis, lumimask, lumidata, dataset_era, is_mc):
    int_lumi = 0
    if not is_mc:
        runs = NUMPY_LIB.asnumpy(analyzed_runs)
        lumis = NUMPY_LIB.asnumpy(analyzed_lumis)
        mask_events = NUMPY_LIB.ones(len(runs), dtype=NUMPY_LIB.bool)
        if not (lumimask is None):
           #keep events passing golden JSON
           mask_lumi_golden_json = lumimask[dataset_era](runs, lumis)
           lumi_eff = mask_lumi_golden_json.sum()/len(mask_lumi_golden_json)
           if not (lumi_eff > 0.5):
               print("WARNING, data file had low lumi efficiency", lumi_eff)  
           mask_events = mask_events & NUMPY_LIB.array(mask_lumi_golden_json) 
           #get integrated luminosity in this file
           if not (lumidata is None):
               int_lumi = get_int_lumi(runs, lumis, mask_lumi_golden_json, lumidata[dataset_era])
    return int_lumi

def get_genparticles(data, muons, jets, is_mc, use_cuda):
    genJet = None
    genpart = None

    if is_mc:
        genJet = data.structs["GenJet"][0]
        genpart = data.structs["GenPart"][0]
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
    return genJet, genpart

def assign_data_run_id(scalars, data_runs, dataset_era, is_mc, runmap_numerical):
    if not is_mc:
        scalars["run_index"] = NUMPY_LIB.zeros_like(scalars["run"])
        scalars["run_index"][:] = -1
        runranges_list = data_runs[dataset_era]
        for run_start, run_end, run_name in runranges_list:
            msk = (scalars["run"] >= run_start) & (scalars["run"] <= run_end)
            scalars["run_index"][msk] = runmap_numerical[run_name]
        assert(NUMPY_LIB.sum(scalars["run_index"]==-1)==0)

def finalize_weights(weights, dataset_era, all_weight_names=None):
    if all_weight_names is None:
        all_weight_names = weights.keys()
    
    ret = {}
    ret["only_genweight"] = NUMPY_LIB.copy(weights["nominal"]["nominal"])
    #This one will be changed to include all other weights
    ret["nominal"] = NUMPY_LIB.copy(weights["nominal"]["nominal"])

    #multitply up all the nominal weights
    for this_syst in all_weight_names:
        if this_syst == "nominal" or this_syst == "LHEScaleWeight" or this_syst == "LHEPdfWeight" or this_syst == "mu1_id" or this_syst == "mu1_iso" or this_syst == "mu2_id"or this_syst == "mu2_iso" or this_syst in btag_unc :
            continue
        ret["nominal"] *= weights[this_syst]["nominal"]

    #create the variated weights, where just one weight is variated up or down
    for this_syst in all_weight_names:
        if this_syst == "nominal" or this_syst == "mu1_id" or this_syst == "mu1_iso" or this_syst == "mu2_id"or this_syst == "mu2_iso" or this_syst == "btag_weight":
            continue
        elif this_syst == "LHEScaleWeight":
            for sdir in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]:
                wval_this_systematic = weights[this_syst][sdir]
                wtot = NUMPY_LIB.copy(ret["nominal"])
                wtot *= wval_this_systematic
                ret["{0}__{1}".format(this_syst, sdir)] = wtot

        elif this_syst == "LHEPdfWeight":

            for sdir in range(len(weights[this_syst])): 
                wval_this_systematic = weights[this_syst][str(sdir)]
                wtot = NUMPY_LIB.copy(ret["nominal"])
                wtot *= wval_this_systematic
                ret["{0}__{1}".format(this_syst, sdir)] = wtot

        else:
            for sdir in ["up", "down", "off"]:
                #for the particular weight or scenario we are considering, get the variated value
                if sdir == "off":
                    wval_this_systematic = NUMPY_LIB.ones_like(ret["nominal"])
                else:
                    wval_this_systematic = weights[this_syst][sdir]

                #for other weights, get the nominal
                wtot = NUMPY_LIB.copy(weights["nominal"]["nominal"])

                wtot *= wval_this_systematic
                for other_syst in all_weight_names:
                    if (other_syst == this_syst or other_syst == "nominal") or other_syst == "LHEScaleWeight" or other_syst == "LHEPdfWeight" or other_syst == "mu1_id" or other_syst == "mu1_iso" or other_syst == "mu2_id"or other_syst == "mu2_iso":
                        continue
                    #Don't apply the nominal btag weight while considering btag systematics. 
                    #Fine to apply btag_jes nominal (array of ones) while considering shape variation from btag_cferr1 (or vice-versa)
                    if((this_syst in btag_unc) and (other_syst == "btag_weight")):
                        continue
                    #print("Applying ",other_syst, " to variation of ",this_syst) 
                    wtot *= weights[other_syst]["nominal"] 

                ret["{0}__{1}".format(this_syst, sdir)] = wtot

    if debug:
        for k in ret.keys():
            print("finalized weight", k, ret[k].mean())
    return ret

def compute_event_weights(parameters, weights, scalars, genweight_scalefactor, gghw, zptw, LHEScalew, LHEPdfw,  pu_corrections, is_mc, dataset_era, dataset_name, use_cuda):
    if is_mc:
        if dataset_name in parameters["ggh_nnlops_reweight"]:
            weights["nominal"]["nominal"] = scalars["genWeight"] * genweight_scalefactor * gghw
        elif dataset_name in parameters["ZpT_reweight"][dataset_era]:
            weights["nominal"]["nominal"] = scalars["genWeight"] * genweight_scalefactor * zptw
        else:
            weights["nominal"]["nominal"] = scalars["genWeight"] * genweight_scalefactor
 
        if debug:
            print("mean genWeight=", scalars["genWeight"].mean())
            print("sum genWeight=", scalars["genWeight"].sum())

        #NB: PU weights are currently done on the basis of the events received (subset of a file), therefore the MC distribution may vary
        pu_weights, pu_weights_up, pu_weights_down = compute_pu_weights(
            pu_corrections[dataset_era],
            weights["nominal"]["nominal"],
            scalars["Pileup_nTrueInt"],
            scalars["Pileup_nTrueInt"]) #scalars["PV_npvsGood"])

        if debug:
            print("pu_weights", pu_weights.mean(), pu_weights.std())
            print("pu_weights_up", pu_weights_up.mean(), pu_weights_up.std())
            print("pu_weights_down", pu_weights_down.mean(), pu_weights_down.std())
        
        weights["puWeight"] = {"nominal": pu_weights, "up": pu_weights_up, "down": pu_weights_down}
        
        weights["L1PreFiringWeight"] = {
            "nominal": NUMPY_LIB.ones_like(weights["nominal"]["nominal"]),
            "up": NUMPY_LIB.ones_like(weights["nominal"]["nominal"]),
            "down": NUMPY_LIB.ones_like(weights["nominal"]["nominal"]), 
        }
        if dataset_era == "2016" or dataset_era == "2017":
            if debug:
                print("mean L1PreFiringWeight_Nom=", scalars["L1PreFiringWeight_Nom"].mean())
                print("mean L1PreFiringWeight_Up=", scalars["L1PreFiringWeight_Up"].mean())
                print("mean L1PreFiringWeight_Dn=", scalars["L1PreFiringWeight_Dn"].mean())
            weights["L1PreFiringWeight"] = {
                "nominal": scalars["L1PreFiringWeight_Nom"],
                "up": scalars["L1PreFiringWeight_Up"],
                "down": scalars["L1PreFiringWeight_Dn"]}

        #hardcode the number of LHE weights
        n_max_lheweights = 9
        weights["LHEScaleWeight"] = {
            str(n): NUMPY_LIB.ones_like(weights["nominal"]["nominal"])
            for n in range(n_max_lheweights)}


        #only defined for dy and ewk samples
        if ("dy" in dataset_name) or ("ewk" in dataset_name):
            nevt = len(weights["nominal"]["nominal"])
            for iScale in range(n_max_lheweights):
                LHEScalew_all = NUMPY_LIB.zeros(nevt, dtype=NUMPY_LIB.float32)
                get_theoryweights(LHEScalew.offsets, LHEScalew.LHEScaleWeight, iScale, LHEScalew_all, use_cuda)
                weights["LHEScaleWeight"][str(iScale)] = LHEScalew_all
        
        #only defined for dy and ewk samples or signal samples
        if "dy" in dataset_name or "ewk" in dataset_name or "ggh" in dataset_name or "vbf" in dataset_name or "wmh" in dataset_name or "wph" in dataset_name or "zh" in dataset_name or "tth" in dataset_name:
            n_max_pdfw = scalars["nLHEPdfWeight"][0]
            weights["LHEPdfWeight"] = {
                str(n): NUMPY_LIB.ones_like(weights["nominal"]["nominal"])
                for n in range(n_max_pdfw)}
            nevt = len(weights["nominal"]["nominal"])
            for iPdf in range(n_max_pdfw):
                LHEPdfw_all = NUMPY_LIB.zeros(nevt, dtype=NUMPY_LIB.float32)
                get_theoryweights(LHEPdfw.offsets, LHEPdfw.LHEPdfWeight, iPdf, LHEPdfw_all, use_cuda)
                weights["LHEPdfWeight"][str(iPdf)] = LHEPdfw_all

def evaluate_bdt_ucsd(dnn_vars, gbr_bdt):
    # BDT var=hmmpt
    # BDT var=hmmrap
    # BDT var=hmmthetacs
    # BDT var=hmmphics
    # BDT var=j1pt
    # BDT var=j1eta
    # BDT var=j2pt
    # BDT var=detajj
    # BDT var=dphijj
    # BDT var=mjj
    # BDT var=met
    # BDT var=zepen
    # BDT var=hmass
    # BDT var=njets
    # BDT var=drmj
    varnames = [
        "Higgs_pt",
        "Higgs_rapidity",
        "hmmthetacs",
        "hmmphics",
        "leadingJet_pt",
        "leadingJet_eta",
        "subleadingJet_pt",
        "dEta_jj_abs",
        "dPhi_jj_mod_abs",
        "M_jj",
        "MET_pt",
        "Zep_rapidity",
        "Higgs_mass",
        "num_jets",
        "dRmin_mj",
    ]

    X = NUMPY_LIB.asnumpy(NUMPY_LIB.stack([dnn_vars[vname] for vname in varnames], axis=1))
    #print("bdt_ucsd inputs")
    #print(X.mean(axis=0), X.min(axis=0), X.max(axis=0), sep="\n")
    y = gbr_bdt.compute(X)
    if X.shape[0] > 0:
        for ivar in range(len(varnames)):
            print(varnames[ivar], X[:, ivar].min(), X[:, ivar].max())
        print("bdt_ucsd eval", y.mean(), y.std(), y.min(), y.max())
    return y

def evaluate_bdt2j_ucsd(dnn_vars, gbr_bdt):
    varnames = [
        "Higgs_pt",
        "Higgs_rapidity",
        "hmmthetacs",
        "hmmphics",
        "leadingJet_pt",
        "leadingJet_eta",
        "subleadingJet_pt",
        "dEta_jj_abs",
        "dPhi_jj_mod_abs",
        "M_jj",
        "MET_pt",
        "Zep_rapidity",
        "num_jets",
        "dRmin_mj",
        "m1ptOverMass",
        "m2ptOverMass",
        "m1eta",
        "m2eta",
    ]

    X = NUMPY_LIB.asnumpy(NUMPY_LIB.stack([dnn_vars[vname] for vname in varnames], axis=1))
    #print("bdt_ucsd inputs")
    #print(X.mean(axis=0), X.min(axis=0), X.max(axis=0), sep="\n")
    y = gbr_bdt.compute(X)
    if X.shape[0] > 0:
        for ivar in range(len(varnames)):
            print(varnames[ivar], X[:, ivar].min(), X[:, ivar].max())
        print("bdt_ucsd2j eval", y.mean(), y.std(), y.min(), y.max())
    return y

def evaluate_bdt01j_ucsd(dnn_vars, gbr_bdt):
    varnames = [
        "Higgs_pt",
        "Higgs_rapidity",
        "hmmthetacs",
        "hmmphics",
        "leadingJet_pt",
        "leadingJet_eta",
        "MET_pt",
        "num_jets_btag",
        "dRmin_mj",
        "num_jets",
        "m1ptOverMass",
        "m2ptOverMass",
        "m1eta",
        "m2eta",
    ]

    X = NUMPY_LIB.asnumpy(NUMPY_LIB.stack([dnn_vars[vname] for vname in varnames], axis=1))
    #print("bdt_ucsd inputs")
    #print(X.mean(axis=0), X.min(axis=0), X.max(axis=0), sep="\n")
    y = gbr_bdt.compute(X)
    if X.shape[0] > 0:
        for ivar in range(len(varnames)):
            print(varnames[ivar], X[:, ivar].min(), X[:, ivar].max())
        print("bdt_ucsd01j eval", y.mean(), y.std(), y.min(), y.max())
    return y

def vbf_genfilter(genjet_inv_mass, num_good_genjets, parameters, dataset_name):
    mask_dijet_genmass = (genjet_inv_mass > parameters["vbf_filter_mjj_cut"])
    mask_2gj = num_good_genjets >= 2
    invert_mask = parameters["vbf_filter"][dataset_name]
    if invert_mask:
        mask_dijet_genmass = NUMPY_LIB.invert(mask_dijet_genmass)

    mask_out = NUMPY_LIB.ones_like(mask_dijet_genmass)
    mask_out[mask_2gj & NUMPY_LIB.invert(mask_dijet_genmass)] = False
    print("VBF genfilter on sample", dataset_name,
        "numev", len(mask_out), "2gj", mask_2gj.sum(),
        "2gj&&mjj", (mask_2gj&mask_dijet_genmass).sum(), "out", mask_out.sum()
    )
 
    return mask_out

#Main analysis entry point
def run_analysis(
    cmdline_args,
    outpath,
    job_descriptions,
    parameter_sets,
    analysis_corrections,
    numev_per_chunk=100000):

    #Keep track of number of events
    nev_total = 0
    nev_loaded = 0
    t0 = time.time()
            
    processed_size_mb = 0

    #This will load the data
    training_set_generator = InputGen(
        job_descriptions,
        cmdline_args.datapath,
        cmdline_args.do_fsr,
        nthreads=cmdline_args.nthreads,
        events_per_file = numev_per_chunk
    )

    num_processed = 0
   
    tprev = time.time()
    #loop over all data, call the analyze function
    while num_processed < len(training_set_generator):

        ds = training_set_generator.nextone()
        
        #All data has been processed
        if ds is None:
            break

        #Process the dataset
        ret, ds, nev, memsize = event_loop(
            ds,
            analysis_corrections,
            parameter_sets, 
            cmdline_args.do_fsr,
            cmdline_args.use_cuda,
        )

        tnext = time.time()
        print("processed {0:.2E} ev/s".format(nev/float(tnext-tprev)))
        sys.stdout.flush()
        tprev = tnext

        with open("{0}/{1}_{2}_{3}.pkl".format(outpath, ds.name, ds.era, ds.num_chunk), "wb") as fi:
            pickle.dump(ret, fi, protocol=pickle.HIGHEST_PROTOCOL)
            
        processed_size_mb += memsize
        nev_total += ret["num_events"]
        nev_loaded += nev
        num_processed += 1
    print()

    #Here we get the metadata (runs, genweights etc) from the files
    for jd in job_descriptions:
        dataset_name = jd["dataset_name"]
        dataset_era = jd["dataset_era"]
        dataset_num_chunk = jd["dataset_num_chunk"]
        
        #Merge the split results
        partial_results = glob.glob("{0}/{1}_{2}_{3}_*.pkl".format(outpath, dataset_name, dataset_era, dataset_num_chunk))
        res = Results({})
        for res_file in partial_results:
            res += pickle.load(open(res_file, "rb"))
            os.remove(res_file)

        if jd["is_mc"]:
            res["cache_metadata"] = [func_filename_precompute_mc(fn) for fn in jd["filenames"]]
            res["genEventSumw"] = genweight_scalefactor * sum([
                md["genEventSumw"] for md in res["cache_metadata"]
            ])
        else:
            #Compute integrated luminosity on data sample and apply golden JSON
            analyzed_runs, analyzed_lumis = get_lumis(jd["filenames"])  
            int_lumi = compute_integrated_luminosity(analyzed_runs, analyzed_lumis, analysis_corrections.lumimask, analysis_corrections.lumidata, dataset_era, jd["is_mc"])
            res["int_lumi"] = int_lumi

        with open("{0}/{1}_{2}_{3}.pkl".format(outpath, dataset_name, dataset_era, dataset_num_chunk), "wb") as fi:
            pickle.dump(res, fi, protocol=pickle.HIGHEST_PROTOCOL)
    
    t1 = time.time()
    dt = t1 - t0
    print("In run_analysis, processed {nev_loaded} ({nev} raw NanoAOD equivalent) events in total {size:.2f} GB, {dt:.1f} seconds, {evspeed:.2E} Hz, {sizespeed:.2f} MB/s".format(
        nev=nev_total, nev_loaded=nev_loaded, dt=dt,
        size=processed_size_mb/1024.0, evspeed=nev_total/dt, sizespeed=processed_size_mb/dt,
        )
    )

    bench_ret = {}
    bench_ret.update(cmdline_args.__dict__)
    bench_ret["hostname"] = os.uname()[1]
    bench_ret["nev_total"] = nev_total
    bench_ret["nev_loaded"] = nev_loaded
    bench_ret["size"] = processed_size_mb
    bench_ret["total_time"] = dt
    bench_ret["evspeed"] = nev_total/dt
    with open(cmdline_args.out + "/analysis_benchmarks.txt", "a") as of:
        of.write(json.dumps(bench_ret) + '\n')
    return bench_ret

#Analyze the loaded data with multiple parameter sets
def event_loop(ds, analysis_corrections, parameter_sets, do_fsr=False, use_cuda=False):

    #copy dataset to GPU and make sure future operations are done on it
    if use_cuda:
        import cupy
        ds.numpy_lib = cupy
        ds.move_to_device(cupy)

    #Analyze one parameter set
    ret = {}
    for parameter_set_name, parameter_set in parameter_sets.items():
        print("doing analysis on parameter set", parameter_set_name)
        ret[parameter_set_name] = analyze_data(
            ds, analysis_corrections, parameter_set,
            parameter_set_name, ds.random_seed,
            do_fsr=do_fsr, use_cuda=use_cuda) 
    ret["num_events"] = len(ds)

    #clean up CUDA memory
    if use_cuda:
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    ret = Results(ret)
    return ret, ds, len(ds), ds.memsize()/1024.0/1024.0

def get_histogram(data, weights, bins, mask=None):
    """Given N-unit vectors of data and weights, returns the histogram in bins
    """
    return Histogram(*ha.histogram_from_vector(data, weights, bins, mask))
 
#remove parallel running for safety - it's not clear if different loop iterations modify overlapping data, which is not allowed
@numba.njit(parallel=False)
def fix_muon_fsrphoton_index(mu_pt, mu_eta, mu_phi, mu_mass, passes_aeta, passes_subleading_pt, offsets_fsrphotons, offsets_muons, fsrphotons_dROverEt2, fsrphotons_relIso03, fsrphotons_pt, fsrphotons_muonIdx, muons_fsrPhotonIdx, out_muons_fsrPhotonIdx, fsr_dROverEt2_cut, fsr_relIso03_cut, pt_fsr_over_mu_e_cut):
    for iev in range(len(offsets_fsrphotons) - 1):
        k = 0
        for i in range(offsets_fsrphotons[iev], offsets_fsrphotons[iev + 1]):
            #Index of muon associated to FSR
            midx = offsets_muons[iev] + fsrphotons_muonIdx[i]

            #Index of FSR photon associated to muon
            fidx = offsets_fsrphotons[iev] + muons_fsrPhotonIdx[midx]

            px = mu_pt[midx] * np.cos(mu_phi[midx])
            py = mu_pt[midx] * np.sin(mu_phi[midx])
            pz = mu_pt[midx] * np.sinh(mu_eta[midx])
            mu_e = np.sqrt(px**2 + py**2 + pz**2 + mu_mass[midx]**2)
            
            sel_fsr = fsrphotons_dROverEt2[fidx] < fsr_dROverEt2_cut and (fsrphotons_relIso03[fidx] < fsr_relIso03_cut and fsrphotons_pt[i]/mu_pt[midx] < pt_fsr_over_mu_e_cut) and passes_aeta[midx] and passes_subleading_pt[midx]
            sel_check_fsr = fsrphotons_dROverEt2[i] < fsr_dROverEt2_cut and (fsrphotons_relIso03[i] < fsr_relIso03_cut and fsrphotons_pt[i]/mu_pt[midx] < pt_fsr_over_mu_e_cut) and passes_aeta[midx] and passes_subleading_pt[midx]
            if sel_fsr or sel_check_fsr:
                if k != muons_fsrPhotonIdx[midx]:
                    if not sel_fsr:
                        out_muons_fsrPhotonIdx[midx] = k
                    else:
                        if sel_check_fsr:
                            if fsrphotons_dROverEt2[i] < fsrphotons_dROverEt2[fidx]:
                                out_muons_fsrPhotonIdx[midx] = k
            else:
               out_muons_fsrPhotonIdx[midx] = -1
            k = k + 1


def get_selected_muons(
    scalars,
    muons, fsrphotons, trigobj, mask_events,
    mu_pt_cut_leading, mu_pt_cut_subleading,
    mu_aeta_cut, mu_iso_cut, muon_id_type,
        muon_trig_match_dr, mu_iso_trig_matched_cut, muon_id_trig_matched_type, fsr_dROverEt2_cut, 
            fsr_relIso03_cut, pt_fsr_over_mu_e_cut, use_cuda):
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
    mu_iso_trig_matched_cut (float) - tight isolation requirement the trigger matched muon
    muon_id_trig_matched_type (string) - "tight" muon ID requirement for trigger matched muon
    """

    passes_aeta = NUMPY_LIB.abs(muons.eta) < mu_aeta_cut
    passes_subleading_pt = muons.pt > mu_pt_cut_subleading

    if fsrphotons:
        out_muons_fsrPhotonIdx = NUMPY_LIB.asnumpy(muons.fsrPhotonIdx)
        mu_pt = NUMPY_LIB.asnumpy(muons.pt)
        mu_eta = NUMPY_LIB.asnumpy(muons.eta)
        mu_phi = NUMPY_LIB.asnumpy(muons.phi)
        mu_mass = NUMPY_LIB.asnumpy(muons.mass)
        mu_iso = NUMPY_LIB.asnumpy(muons.pfRelIso04_all)
        fix_muon_fsrphoton_index(
            mu_pt, mu_eta, mu_phi, mu_mass, passes_aeta, passes_subleading_pt,
            NUMPY_LIB.asnumpy(fsrphotons.offsets),
            NUMPY_LIB.asnumpy(muons.offsets),
            NUMPY_LIB.asnumpy(fsrphotons.dROverEt2),
            NUMPY_LIB.asnumpy(fsrphotons.relIso03),
            NUMPY_LIB.asnumpy(fsrphotons.pt),
            NUMPY_LIB.asnumpy(fsrphotons.muonIdx),
            NUMPY_LIB.asnumpy(muons.fsrPhotonIdx),
            out_muons_fsrPhotonIdx, fsr_dROverEt2_cut, 
            fsr_relIso03_cut, pt_fsr_over_mu_e_cut
        )
        correct_muon_with_fsr(
            NUMPY_LIB.asnumpy(muons.offsets),
            NUMPY_LIB.asnumpy(fsrphotons.offsets),
            mu_pt, mu_eta, mu_phi, mu_mass, mu_iso,
            out_muons_fsrPhotonIdx,
            NUMPY_LIB.asnumpy(fsrphotons.pt),
            NUMPY_LIB.asnumpy(fsrphotons.eta),
            NUMPY_LIB.asnumpy(fsrphotons.phi)
        )
        #move back to GPU
        if use_cuda:
            mu_pt = NUMPY_LIB.array(mu_pt)
            mu_eta = NUMPY_LIB.array(mu_eta)
            mu_phi = NUMPY_LIB.array(mu_phi)
            mu_mass = NUMPY_LIB.array(mu_mass)
            mu_iso = NUMPY_LIB.array(mu_iso)
        muons.pt = mu_pt
        muons.eta = mu_eta
        muons.phi = mu_phi
        muons.mass = mu_mass
        muons.pfRelIso04_all = mu_iso

    passes_iso = muons.pfRelIso04_all < mu_iso_cut
    passes_iso_trig_matched = muons.pfRelIso04_all < mu_iso_trig_matched_cut

    if muon_id_type == "medium":
        passes_id = muons.mediumId == 1
    elif muon_id_type == "tight":
        passes_id = muons.tightId == 1
    else:
        raise Exception("unknown muon id: {0}".format(muon_id_type))

    if muon_id_trig_matched_type == "tight":
        passes_id_trig_matched = muons.tightId == 1
    else:
        raise Exception("unknown muon id: {0}".format(muon_id_type))

    #find muons that pass ID
    passes_leading_pt = muons.pt > mu_pt_cut_leading
    muons_passing_id =  (
        passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )

    muons_passing_id_trig_matched =  (
        passes_iso_trig_matched & passes_id_trig_matched &
        passes_subleading_pt & passes_aeta
    )

    #Get muons that are high-pt and are matched to trigger object
   # mask_trigger_objects_mu = (trigobj.id == 13)
   # muons_matched_to_trigobj = NUMPY_LIB.invert(ha.mask_deltar_first(
   #     {"eta": muons.eta, "phi": muons.phi, "offsets": muons.offsets},
   #     muons_passing_id_trig_matched & passes_leading_pt,
   #     {"eta": trigobj.eta, "phi": trigobj.phi, "offsets": trigobj.offsets},
   #     mask_trigger_objects_mu, muon_trig_match_dr
   # ))
    #muons.attrs_data["triggermatch"] = muons_matched_to_trigobj
    muons.attrs_data["pass_id"] = muons_passing_id
    muons.attrs_data["passes_leading_pt"] = passes_leading_pt

    #At least one muon must be matched to trigger object, find such events
    #events_passes_triggermatch = ha.sum_in_offsets(
    #    muons.offsets, muons_matched_to_trigobj, mask_events,
    #    muons.masks["all"], NUMPY_LIB.int8
    #) >= 1

    #select events that have muons passing cuts: 2 passing ID, 1 passing leading pt, 2 passing subleading pt
    events_passes_muid = ha.sum_in_offsets(
        muons.offsets, muons_passing_id, mask_events, muons.masks["all"],
        NUMPY_LIB.int8) >= 2
    events_passes_leading_pt = ha.sum_in_offsets(
        muons.offsets, muons_passing_id & passes_leading_pt, mask_events,
        muons.masks["all"], NUMPY_LIB.int8) >= 1
    events_passes_subleading_pt = ha.sum_in_offsets(
        muons.offsets, muons_passing_id & passes_subleading_pt,
        mask_events, muons.masks["all"], NUMPY_LIB.int8) >= 2

    #Get the mask of selected events
    base_event_sel = (
        mask_events &
        #events_passes_triggermatch &
        events_passes_muid &
        events_passes_leading_pt &
        events_passes_subleading_pt
    )

    #Find two opposite sign muons among the muons passing ID and subleading pt
    muons_passing_os = ha.select_opposite_sign(
        muons.offsets, muons.charge, muons_passing_id & passes_subleading_pt)
    events_passes_os = ha.sum_in_offsets(
        muons.offsets, muons_passing_os, mask_events,
        muons.masks["all"], NUMPY_LIB.int32) == 2

    muons.attrs_data["pass_os"] = muons_passing_os
    final_event_sel = base_event_sel & events_passes_os
    final_muon_sel = muons_passing_id & passes_subleading_pt & muons_passing_os
    additional_muon_sel = muons_passing_id & NUMPY_LIB.invert(muons_passing_os)
    muons.masks["iso_id_aeta"] = passes_iso & passes_id & passes_aeta
    
    # To save time apply a mass window cut. get the invariant mass of the dimuon system and compute mass windows
    #higgs_inv_mass, _ = compute_inv_mass(muons, final_event_sel, final_muon_sel, use_cuda)
    #final_event_sel = final_event_sel & (higgs_inv_mass > 100.0)
    
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("muons")
            jaggedstruct_print(muons, idx,
                ["pt", "eta", "phi", "charge", "pfRelIso04_all", "mediumId",
                "isGlobal", "isTracker", 
                 "pass_id", "passes_leading_pt"])

    return {
        "selected_events": final_event_sel,
        "muons_passing_id_pt": muons_passing_id & passes_subleading_pt,
        "selected_muons": final_muon_sel,
        "muons_passing_os": muons_passing_os,
        "additional_muon_sel": additional_muon_sel,
    }

#Corrects the muon momentum and isolation, if a matched FSR photon with dR<0.4 is found
@numba.njit(parallel=True)
def correct_muon_with_fsr(
        muons_offsets, fsr_offsets,
        muons_pt, muons_eta, muons_phi, muons_mass, muons_iso, muons_fsrIndex,
        fsr_pt, fsr_eta, fsr_phi
    ):

    for iev in numba.prange(len(muons_offsets) - 1):

        #loop over muons in event
        mu_first = muons_offsets[iev]
        mu_last = muons_offsets[iev + 1]
        for imu in range(mu_first, mu_last):
            #relative FSR index in the event
            fsr_idx_relative = muons_fsrIndex[imu]

            if (fsr_idx_relative >= 0) and (muons_pt[imu]>20):
                #absolute index in the full FSR vector for all events
                ifsr = fsr_offsets[iev] + fsr_idx_relative
                mu_kin = {"pt": muons_pt[imu], "eta": muons_eta[imu], "phi": muons_phi[imu], "mass": muons_mass[imu]}
                fsr_kin = {"pt": fsr_pt[ifsr], "eta": fsr_eta[ifsr], "phi": fsr_phi[ifsr],"mass": 0}

                # dR between muon and photon
                deta = muons_eta[imu] - fsr_eta[ifsr]
                dphi = backend_cpu.deltaphi(muons_phi[imu], fsr_phi[ifsr])
                dr = np.sqrt(deta**2 + dphi**2)

                    
                #compute and set corrected momentum
                px_total = 0
                py_total = 0
                pz_total = 0
                e_total = 0
                for obj in [mu_kin, fsr_kin]:
                    px = obj["pt"] * np.cos(obj["phi"])
                    py = obj["pt"] * np.sin(obj["phi"])
                    pz = obj["pt"] * np.sinh(obj["eta"])
                    e = np.sqrt(px**2 + py**2 + pz**2 + obj["mass"]**2)
                    px_total += px
                    py_total += py
                    pz_total += pz
                    e_total += e
                out_pt = np.sqrt(px_total**2 + py_total**2)
                out_eta = np.arcsinh(pz_total / out_pt)
                out_phi = np.arctan2(py_total, px_total)

                
                update_iso = dr<0.4

                #reference: https://gitlab.cern.ch/uhh-cmssw/fsr-photon-recovery/tree/master
                if update_iso:
                    muons_iso[imu] = (muons_iso[imu]*muons_pt[imu] - fsr_pt[ifsr])/out_pt

                muons_pt[imu] = out_pt
                muons_eta[imu] = out_eta
                muons_phi[imu] = out_phi


def get_bit_values(array, bit_index):
    """
    Given an array of N binary values (e.g. jet IDs), return the bit value at bit_index in [0, N-1].
    """
    return (array & 2**(bit_index)) >> 1

# Custom kernels to get the number of softJets with pT>5 GEV
def nsoftjets(doEta, nsoft, softht, nevt,softjets, leading_muon, subleading_muon, leading_jet, subleading_jet, ptcut, dr2cut, use_cuda):
    nsjet_out = NUMPY_LIB.zeros(nevt, dtype=NUMPY_LIB.int32)
    HTsjet_out = NUMPY_LIB.zeros(nevt, dtype=NUMPY_LIB.float32)
    if use_cuda:
        nsoftjets_cudakernel[32, 1024](
            doEta, nsoft, softht, nevt,
            softjets.offsets, softjets.pt, softjets.eta, softjets.phi,
            leading_jet["eta"], subleading_jet["eta"], leading_jet["phi"],
            subleading_jet["phi"], leading_muon["eta"], subleading_muon["eta"],
            leading_muon["phi"], subleading_muon["phi"], ptcut, dr2cut,
            nsjet_out, HTsjet_out)
        cuda.synchronize() 
    else:
        nsoftjets_cpu(doEta, nsoft, softht, nevt, softjets.offsets, softjets.pt, softjets.eta, softjets.phi, leading_jet["eta"], subleading_jet["eta"], leading_jet["phi"], subleading_jet["phi"], leading_muon["eta"], subleading_muon["eta"], leading_muon["phi"], subleading_muon["phi"], ptcut, dr2cut, nsjet_out, HTsjet_out)
    return nsjet_out, HTsjet_out

@numba.njit(parallel=True, fastmath=True)
def nsoftjets_cpu(doEta, nsoft, softht, nevt, softjets_offsets, pt, eta, phi, etaj1, etaj2, phij1, phij2, etam1, etam2, phim1, phim2, ptcut, dr2cut, nsjet_out, HTsjet_out):
    phis = [phij1, phij2, phim1, phim2]
    etas = [etaj1, etaj2, etam1, etam2]
    #process events in parallel
    for iev in numba.prange(nevt):
        nbadsjet = 0
        htsjet = 0
        for isoftjets in range(softjets_offsets[iev], softjets_offsets[iev + 1]):
            ptsel = pt[isoftjets] > ptcut
            if ptsel:
                sj_sel = True
                nobj = len(phis)
                for index in range(nobj):
                    dphi = backend_cpu.deltaphi(phi[isoftjets], phis[index][iev])
                    deta = eta[isoftjets] - etas[index][iev]
                    dr = dphi**2 + deta**2
                    if dr < dr2cut:
                        sj_sel = False
                        break

                if not sj_sel:
                    htsjet += pt[isoftjets]
                    nbadsjet += 1
                elif doEta:
                    if ((((eta[isoftjets]>etaj1[iev]) or (eta[isoftjets]<etaj2[iev])) and (etaj1[iev]>etaj2[iev])) or (((eta[isoftjets]>etaj2[iev]) or (eta[isoftjets]<etaj1[iev])) and (etaj1[iev]<etaj2[iev]))):
                        htsjet += pt[isoftjets]
                        nbadsjet += 1

        nsjet_out[iev] = nsoft[iev] - nbadsjet
        HTsjet_out[iev] = softht[iev] - htsjet

@cuda.jit
def nsoftjets_cudakernel(doEta, nsoft, softht, nevt, softjets_offsets, pt, eta, phi, etaj1, etaj2, phij1, phij2, etam1, etam2, phim1, phim2, ptcut, dr2cut, nsjet_out, HTsjet_out):
    #process events in parallel
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for iev in range(xi, nevt, xstride):
        nbadsjet = 0
        htsjet = 0
        phis = cuda.local.array(4, numba.float32)
        phis[0] = phij1[iev]
        phis[1] = phij2[iev]
        phis[2] = phim1[iev]
        phis[3] = phim2[iev]
        etas = cuda.local.array(4, numba.float32)
        etas[0] = etaj1[iev]
        etas[1] = etaj2[iev]
        etas[2] = etam1[iev]
        etas[3] = etam2[iev]
        for isoftjets in range(softjets_offsets[iev], softjets_offsets[iev + 1]):
            sel = pt[isoftjets] > ptcut
            if(doEta):
                sel = sel | ((((eta[isoftjets]>etaj1[iev]) or (eta[isoftjets]<etaj2[iev])) and (etaj1[iev]>etaj2[iev])) or (((eta[isoftjets]>etaj2[iev]) or (eta[isoftjets]<etaj1[iev])) and (etaj1[iev]<etaj2[iev])))
            if (sel):
                sj_sel = True
                nobj = len(phis)
                for index in range(nobj):
                    dphi = backend_cuda.deltaphi_devfunc(phi[isoftjets], phis[index])
                    deta = eta[isoftjets] - etas[index]
                    dr = dphi**2 + deta**2
                    if dr < dr2cut:
                        sj_sel = False
                        break

                if not sj_sel: 
                    htsjet += pt[isoftjets]
                    nbadsjet += 1
                            
        nsjet_out[iev] = nsoft[iev] - nbadsjet
        HTsjet_out[iev] = softht[iev] - htsjet

#The value is a bit representation of the fulfilled working points: tight (1), medium (2), and loose (4).
#As tight is also medium and medium is also loose, there are only 4 different settings: 0 (no WP, 0b000), 4 (loose, 0b100), 6 (medium, 0b110), and 7 (tight, 0b111).
def jet_puid_cut(
    jets,
    jet_puid):
    if jet_puid == "loose":
        pass_jet_puid = NUMPY_LIB.logical_or(NUMPY_LIB.logical_and(jets.puId >= 4 , jets.pt<50.), jets.pt>50.)
    elif jet_puid == "medium":
        pass_jet_puid = NUMPY_LIB.logical_or(NUMPY_LIB.logical_and(jets.puId >= 6, jets.pt<50.), jets.pt>50.)
    elif jet_puid == "tight":
        pass_jet_puid = NUMPY_LIB.logical_or(NUMPY_LIB.logical_and(jets.puId >= 7, jets.pt<50.), jets.pt>50.)
    elif jet_puid == "none":
        pass_jet_puid = NUMPY_LIB.ones(jets.numobjects(), dtype=NUMPY_LIB.bool)
    return pass_jet_puid

def get_selected_jets_id(
    scalars,
    jets,
    muons,
    jet_eta_cut,
    jet_dr_cut,
    jet_id,
    jet_puid,
    jet_veto_eta_lower_cut,
    jet_veto_eta_upper_cut,
    jet_veto_raw_pt,
    dataset_era):
    #2017 and 2018: jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto'))
    #Jet ID flags bit0 is loose (always false in 2017 since it does not exist), bit1 is tight, bit2 is tightLepVeto
    #run2_nanoAOD_94X2016: jetId = Var("userInt('tightIdLepVeto')*4+userInt('tightId')*2+userInt('looseId')",int,doc="Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto"
    if jet_id[dataset_era] == "tight":
        if dataset_era == "2017" or dataset_era == "2018":
            pass_jetid = jets.jetId >= 2
        else:
            pass_jetid = jets.jetId >= 3
    elif jet_id[dataset_era] == "loose": 
        pass_jetid = jets.jetId >= 1

    pass_jet_puid = jet_puid_cut(jets, jet_puid)
    
    abs_eta = NUMPY_LIB.abs(jets.eta)
    jet_eta_pass_veto = NUMPY_LIB.ones(jets.numobjects(), dtype=NUMPY_LIB.bool)
    if dataset_era == "2017":
        jet_eta_pass_veto = NUMPY_LIB.logical_or(NUMPY_LIB.logical_and(jets.puId >= 7, NUMPY_LIB.logical_and(abs_eta < jet_veto_eta_upper_cut, abs_eta > jet_veto_eta_lower_cut)), NUMPY_LIB.logical_or(
                (abs_eta > jet_veto_eta_upper_cut),
                (abs_eta < jet_veto_eta_lower_cut)
            ))

    pass_qgl = jets.qgl > -2 

    selected_jets = (
        (abs_eta < jet_eta_cut) &
            pass_jetid & pass_qgl
    ) & jet_eta_pass_veto & pass_jet_puid
        
    jets_pass_dr = ha.mask_deltar_first(
        {"eta": jets.eta, "phi": jets.phi, "offsets": jets.offsets},
        selected_jets,
        {"eta": muons.eta, "phi": muons.phi, "offsets": muons.offsets},
        muons.masks["iso_id_aeta"], jet_dr_cut)

    jets.masks["pass_dr"] = jets_pass_dr
    
    selected_jets = selected_jets & jets_pass_dr
   
    '''
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets")
            
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi", "mass", "jetId", "puId","qgl"])
    '''
    return selected_jets

def get_selected_jets(
    scalars,
    jets,
    mask_events,
    jet_pt_cut_subleading,
    jet_btag_medium,
    jet_btag_loose,
    is_mc,
    redo_jec,
    debug,
    use_cuda
    ):
    """
    Given jets and selected muons in events, choose jets that pass quality
    criteria and that are not dR-matched to muons.
    """

    selected_jets = (jets.pt > jet_pt_cut_subleading)
 
    #produce a mask that selects the first two selected jets 
    first_two_jets = NUMPY_LIB.zeros_like(selected_jets)

    out_jet_index0 = NUMPY_LIB.zeros(len(jets.offsets)-1, dtype=NUMPY_LIB.int32)
    out_jet_index1 = NUMPY_LIB.ones(len(jets.offsets)-1, dtype=NUMPY_LIB.int32)
    if redo_jec: 
        get_leadtwo_jet_ind(jets.offsets, jets.pt, out_jet_index0, out_jet_index1)
  
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets")
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi", "mass", "jetId", "puId"])
            print("selected leading jet index: ",out_jet_index0[idx])
            print("selected subleading jet index: ",out_jet_index1[idx])
 
    targets = NUMPY_LIB.ones_like(mask_events, dtype=NUMPY_LIB.int32) 
    ha.set_in_offsets(jets.offsets, first_two_jets, out_jet_index0, targets, mask_events, selected_jets)
    ha.set_in_offsets(jets.offsets, first_two_jets, out_jet_index1, targets, mask_events, selected_jets)
    jets.attrs_data["selected"] = selected_jets
    jets.attrs_data["first_two"] = first_two_jets

    dijet_inv_mass, dijet_pt = compute_inv_mass(jets, mask_events, selected_jets & first_two_jets, use_cuda)
    

    selected_jets_btag_medium = selected_jets & (jets.btagDeepB >= jet_btag_medium) & (abs(jets.eta) < 2.5)
    selected_jets_btag_loose = selected_jets & (jets.btagDeepB >= jet_btag_loose) & (abs(jets.eta) <2.5)

    num_jets = ha.sum_in_offsets(jets.offsets, selected_jets, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)

    num_jets_btag_medium = ha.sum_in_offsets(jets.offsets, selected_jets_btag_medium, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)

    num_jets_btag_loose = ha.sum_in_offsets(jets.offsets, selected_jets_btag_loose, mask_events,
        jets.masks["all"], NUMPY_LIB.int8)
    
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets")
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi", "mass", "jetId", "puId",
                                #"pass_dr", 
                                "selected", 
                            "first_two"])
            
    ret = {
        "selected_jets": selected_jets,
        "num_jets": num_jets,
        "num_jets_btag_medium": num_jets_btag_medium,
        "num_jets_btag_loose": num_jets_btag_loose,
        "dijet_inv_mass": dijet_inv_mass,
        "dijet_pt": dijet_pt
    }

    return ret


def get_puid_weights(jets, evaluator, era, wp, subjet_pt, jet_pt_min, jet_pt_max, use_cuda):
    wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
    passed_puid = jet_puid_cut(jets, wp)
    jets_pu_eff, jets_pu_sf = jet_puid_evaluate(evaluator, era, wp_dict[wp], NUMPY_LIB.asnumpy(jets.pt), NUMPY_LIB.asnumpy(jets.eta))
    jet_pt_mask = (NUMPY_LIB.asnumpy(jets.pt)>jet_pt_min) & (NUMPY_LIB.asnumpy(jets.pt)<jet_pt_max)
    p_puid_mc = compute_eff_product(jets.offsets, jets.pt, subjet_pt, jet_pt_mask, passed_puid, jets_pu_eff, use_cuda)
    p_puid_data = compute_eff_product(jets.offsets, jets.pt, subjet_pt, jet_pt_mask, passed_puid, jets_pu_eff*jets_pu_sf, use_cuda)
    eventweight_puid = NUMPY_LIB.divide(p_puid_data, p_puid_mc)
    eventweight_puid[p_puid_mc==0] = 0
    return eventweight_puid

def jet_puid_evaluate(evaluator, era, wp, jet_pt, jet_eta):
    h_eff_name = f"h2_eff_mc{era}_{wp}"
    h_sf_name = f"h2_eff_sf{era}_{wp}"
    puid_eff = evaluator[h_eff_name](jet_pt, jet_eta)
    puid_sf = evaluator[h_sf_name](jet_pt, jet_eta)
    return NUMPY_LIB.array(puid_eff), NUMPY_LIB.array(puid_sf)

def compute_dnnPisaComb(dnnPisaComb_pred, dnnPisa_preds, event_array, n_mass, use_cuda):
    if use_cuda:
        compute_dnnPisaComb_cuda[32,1024](dnnPisaComb_pred, dnnPisa_preds, event_array, n_mass)
        cuda.synchronize()
    else:
        compute_dnnPisaComb_cpu(dnnPisaComb_pred, dnnPisa_preds, event_array, n_mass)

@numba.njit(parallel=True)
def compute_dnnPisaComb_cpu(dnnPisaComb_pred, dnnPisa_preds, event_array, n_mass):
    for i in numba.prange(dnnPisaComb_pred.shape[1]):
        for imass in range(n_mass):
            dnnPisaComb_pred[imass][i] = dnnPisa_preds[n_mass*(event_array[i]%4)+imass][i]
            
@cuda.jit
def compute_dnnPisaComb_cuda(dnnPisaComb_pred, dnnPisa_preds, event_array, n_mass):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for i in range(xi, dnnPisaComb_pred.shape[1], xstride):
        for imass in range(n_mass):
            dnnPisaComb_pred[imass][i] = dnnPisa_preds[n_mass*(event_array[i]%4)+imass][i]

@numba.njit(parallel=True)
def mhfordnn(fixedmH, dimu, movem):
    for i in numba.prange(len(dimu)):
        if (dimu[i]>115) & (dimu[i]<135):
            fixedmH[i] = dimu[i] + movem[i]
        else:
            fixedmH[i] = 125.0

def compute_eff_product(offsets, jet_pt, subjet_pt, jet_pt_mask, jets_mask_passes_id, jets_eff, use_cuda):
    nev = len(offsets) - 1
    p_puid = NUMPY_LIB.ones(nev, dtype=NUMPY_LIB.float32)
    if use_cuda:
        compute_eff_product_cudakernel(offsets, jet_pt_mask, jets_mask_passes_id, jets_eff, p_puid)
        cuda.synchronize()
    else:
        compute_eff_product_cpu(offsets, jet_pt, subjet_pt, jet_pt_mask, jets_mask_passes_id, jets_eff, p_puid)
    return p_puid


@numba.njit(parallel=True)
def compute_event_btag_weight_shape(offsets, jets_sf, out_weight):
    for iev in numba.prange(len(offsets)-1):
        p_tot = 1.0
        #loop over jets in event
        for ij in range(offsets[iev], offsets[iev+1]):
            p_tot *= jets_sf[ij]
        out_weight[iev] = p_tot

def get_btag_weights_shape(jets, evaluator, era, scalars, pt_cut):
    tag_name = 'DeepCSV_'+era
    nev = jets.numevents()
    jet_pt = numpy.copy(jets.pt)
    jet_pt[(jets.pt > 1000.)]=1000.
    pt_eta_mask = NUMPY_LIB.logical_or((NUMPY_LIB.abs(jets.eta)>2.4), (jets.pt < pt_cut))
    p_jetWt = NUMPY_LIB.ones(len(jets.pt), dtype=NUMPY_LIB.float32)
    eventweight_btag = NUMPY_LIB.ones(nev)
    mask1 = NUMPY_LIB.logical_and(NUMPY_LIB.logical_not(pt_eta_mask), (jets.pt > 1000. ))
    mask_pt_bounds =[]
    mask_pt_bounds.append(NUMPY_LIB.logical_and(mask1,jets.hadronFlavour == 5))
    mask_pt_bounds.append(NUMPY_LIB.logical_and(mask1,jets.hadronFlavour == 4))
    mask_pt_bounds.append(NUMPY_LIB.logical_and(mask1,jets.hadronFlavour == 0))
    # Code help from https://github.com/chreissel/hepaccelerate/blob/mass_fit/lib_analysis.py#L118
    # Code help from https://gitlab.cern.ch/uhh-cmssw/CAST/blob/master/BTaggingWeight/plugins/BTaggingReShapeProducer.cc
    
    for tag in ["DeepCSV_3_iterativefit_central_0", "DeepCSV_3_iterativefit_central_1", "DeepCSV_3_iterativefit_central_2"]:
        SF_btag = evaluator[tag_name].evaluator[tag](NUMPY_LIB.abs(jets.eta), jet_pt, jets.btagDeepB)
        if tag.endswith("0"):
            SF_btag[(jets.hadronFlavour != 5)] = 1.
        if tag.endswith("1"):
            SF_btag[jets.hadronFlavour != 4] = 1.
        if tag.endswith("2"):
            SF_btag[jets.hadronFlavour != 0] = 1.
        
        p_jetWt*=SF_btag
    #print("p_JetWt before", p_jetWt, p_jetWt.mean(), p_jetWt.std())
    p_jetWt[pt_eta_mask] = 1.
    #print("p_JetWt after", p_jetWt, p_jetWt.mean(), p_jetWt.std())
    
    compute_event_btag_weight_shape(jets.offsets, p_jetWt, eventweight_btag)
    #print("eventweight_btag", eventweight_btag, eventweight_btag.mean(), eventweight_btag.std())
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets for b tag")
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi","hadronFlavour", "btagDeepB", "jetId", "puId","qgl"])
            print(eventweight_btag[idx])
    #not all syst are for all flavours
    # bFlav - jes, lf, hfstats1, hfstats2
    # cFlav - cferr1, cferr2
    # lFlav - jes, hf, lfstats1, lfstats2
    tag_sys = []
    tag_sys.append(['jes', 'lf', 'hfstats1', 'hfstats2'])
    tag_sys.append([ 'cferr1', 'cferr2'])
    tag_sys.append(['jes', 'hf',  'lfstats1', 'lfstats2'])
    p_jetWt_up= []
    p_jetWt_down=[]
    eventweight_btag_up = []
    eventweight_btag_down = []
    for i in range(0,3): #0 is for b flav, 1 is c flav and 2 is udsg
        p_jetWt_up.append(NUMPY_LIB.ones(len(jets.pt)))
        p_jetWt_down.append(NUMPY_LIB.ones(len(jets.pt)))
        eventweight_btag_up.append(NUMPY_LIB.ones(nev))
        eventweight_btag_down.append(NUMPY_LIB.ones(nev))
    #print(evaluator[tag_name].evaluator.keys())
    for i in range(0,3):
        for sdir in ['up','down']:
            for tsys in tag_sys[i]:
                tsys_name = "DeepCSV_3_iterativefit_" + sdir + '_' + tsys + '_' + str(i)
                #automatically skip syst which aren't for a particular flavour
                if tsys_name not in evaluator[tag_name].evaluator.keys():
                    print(tsys_name, " not found for flavour ",i)
                    continue
                SF_btag = evaluator[tag_name].evaluator[tsys_name](NUMPY_LIB.abs(jets.eta), jet_pt, jets.btagDeepB)
                
                if tsys_name.endswith("0"):
                    SF_btag[jets.hadronFlavour != 5] = 1.
                if tsys_name.endswith("1"):
                    SF_btag[jets.hadronFlavour != 4] = 1.
                if tsys_name.endswith("2"):
                    SF_btag[jets.hadronFlavour != 0] = 1.
                if sdir == 'up':
                    p_jetWt_up[i]*=SF_btag
                else:
                    p_jetWt_down[i]*=SF_btag
            if sdir == 'up':
                p_jetWt_up[i][pt_eta_mask] = 1.
                #For jets with pt > 1000., evaluate with pt =1000. (done automatically) and inflate to double the systematic
                # based on https://github.com/cms-sw/cmssw/blob/master/CondTools/BTau/src/BTagCalibrationReader.cc#L170
                p_jetWt_up[i][mask_pt_bounds[i]] = p_jetWt[mask_pt_bounds[i]]+2*(p_jetWt_up[i][mask_pt_bounds[i]]-p_jetWt[mask_pt_bounds[i]])
                compute_event_btag_weight_shape(jets.offsets, p_jetWt_up[i], eventweight_btag_up[i])
            else:
                p_jetWt_down[i][pt_eta_mask] = 1.
                #For jets with pt > 1000., evaluate with pt =1000. (done automatically) and inflate to double the systematic
                # based on https://github.com/cms-sw/cmssw/blob/master/CondTools/BTau/src/BTagCalibrationReader.cc#L170
                p_jetWt_down[i][mask_pt_bounds[i]] = p_jetWt[mask_pt_bounds[i]]+2*(p_jetWt_down[i][mask_pt_bounds[i]]-p_jetWt[mask_pt_bounds[i]])
                compute_event_btag_weight_shape(jets.offsets, p_jetWt_down[i], eventweight_btag_down[i])

        if debug:
            for evtid in debug_event_ids:
                idx = np.where(scalars["event"] == evtid)[0][0]
                print("jets for b tag")
                jaggedstruct_print(jets, idx,
                                   ["pt", "eta", "phi","hadronFlavour", "btagDeepB", "jetId", "puId","qgl"])
                print(i,eventweight_btag_up[i][idx],eventweight_btag_down[i][idx])
        #print(p_jetWt_up[i][mask_pt_bounds[i]],jets.pt[mask_pt_bounds[i]],jets.eta[mask_pt_bounds[i]],jets.btagDeepB[mask_pt_bounds[i]])
        #print(p_jetWt_down[i][mask_pt_bounds[i]],jets.pt[mask_pt_bounds[i]],jets.eta[mask_pt_bounds[i]],jets.btagDeepB[mask_pt_bounds[i]])
    return eventweight_btag , eventweight_btag_up, eventweight_btag_down

def get_new_btag_weights_shape(jets, evaluator, era, scalars, pt_cut):
    tag_name = 'DeepCSV_'+era
    nev = jets.numevents()
    jet_pt = numpy.copy(jets.pt)
    jet_pt[(jets.pt > 1000.)]=1000.
    pt_eta_mask = NUMPY_LIB.logical_or((NUMPY_LIB.abs(jets.eta)>2.4), (jets.pt < pt_cut))
    p_jetWt = NUMPY_LIB.ones(len(jets.pt), dtype=NUMPY_LIB.float32)
    eventweight_btag = NUMPY_LIB.ones(nev)
    mask1 = NUMPY_LIB.logical_and(NUMPY_LIB.logical_not(pt_eta_mask), (jets.pt > 1000. ))
    mask_pt_bounds =[]
    mask_pt_bounds.append(NUMPY_LIB.logical_and(mask1,jets.hadronFlavour == 5))
    mask_pt_bounds.append(NUMPY_LIB.logical_and(mask1,jets.hadronFlavour == 4))
    mask_pt_bounds.append(NUMPY_LIB.logical_and(mask1,jets.hadronFlavour == 0))
    # Code help from https://github.com/chreissel/hepaccelerate/blob/mass_fit/lib_analysis.py#L118
    # Code help from https://gitlab.cern.ch/uhh-cmssw/CAST/blob/master/BTaggingWeight/plugins/BTaggingReShapeProducer.cc
    
    p_jetWt = evaluator[tag_name].eval('central', jets.hadronFlavour, NUMPY_LIB.abs(jets.eta), jet_pt, jets.btagDeepB,True)
        
    #print("p_JetWt before", p_jetWt, p_jetWt.mean(), p_jetWt.std())
    p_jetWt[pt_eta_mask] = 1.
    #print("p_JetWt after", p_jetWt, p_jetWt.mean(), p_jetWt.std())
    
    compute_event_btag_weight_shape(jets.offsets, p_jetWt, eventweight_btag)
    #print("eventweight_btag", eventweight_btag, eventweight_btag.mean(), eventweight_btag.std())
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets for b tag")
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi","hadronFlavour", "btagDeepB", "jetId", "puId","qgl"])
            print(eventweight_btag[idx])
    #not all syst are for all flavours
    # bFlav - jes, lf, hfstats1, hfstats2
    # cFlav - cferr1, cferr2
    # lFlav - jes, hf, lfstats1, lfstats2
    tag_sys = []
    tag_sys.append(['jes', 'lf', 'hfstats1', 'hfstats2'])
    tag_sys.append([ 'cferr1', 'cferr2'])
    tag_sys.append(['jes', 'hf',  'lfstats1', 'lfstats2'])
    p_jetWt_up= []
    p_jetWt_down=[]
    eventweight_btag_up = []
    eventweight_btag_down = []
    for i in range(0,3): #0 is for b flav, 1 is c flav and 2 is udsg
        p_jetWt_up.append(NUMPY_LIB.ones(len(jets.pt)))
        p_jetWt_down.append(NUMPY_LIB.ones(len(jets.pt)))
        eventweight_btag_up.append(NUMPY_LIB.ones(nev))
        eventweight_btag_down.append(NUMPY_LIB.ones(nev))
    #print(evaluator[tag_name].evaluator.keys())
    for i in range(0,3):
        for sdir in ['up','down']:
            for tsys in tag_sys[i]:
                tsys_name = sdir + '_' + tsys
                SF_btag = evaluator[tag_name].eval(tsys_name, jets.hadronFlavour, NUMPY_LIB.abs(jets.eta), jet_pt, jets.btagDeepB, True)
                if(i==0):
                    SF_btag[(jets.hadronFlavour)!=5] = 1.
                elif(i==1):
                    SF_btag[(jets.hadronFlavour)!=4] = 1.
                else:
                    SF_btag[(jets.hadronFlavour)!=0] = 1.
                if sdir == 'up':
                    p_jetWt_up[i]*=SF_btag
                else:
                    p_jetWt_down[i]*=SF_btag
            if sdir == 'up':
                p_jetWt_up[i][pt_eta_mask] = 1.
                #For jets with pt > 1000., evaluate with pt =1000. (done automatically) and inflate to double the systematic
                # based on https://github.com/cms-sw/cmssw/blob/master/CondTools/BTau/src/BTagCalibrationReader.cc#L170
                p_jetWt_up[i][mask_pt_bounds[i]] = p_jetWt[mask_pt_bounds[i]]+2*(p_jetWt_up[i][mask_pt_bounds[i]]-p_jetWt[mask_pt_bounds[i]])
                compute_event_btag_weight_shape(jets.offsets, p_jetWt_up[i], eventweight_btag_up[i])
            else:
                p_jetWt_down[i][pt_eta_mask] = 1.
                #For jets with pt > 1000., evaluate with pt =1000. (done automatically) and inflate to double the systematic
                # based on https://github.com/cms-sw/cmssw/blob/master/CondTools/BTau/src/BTagCalibrationReader.cc#L170
                p_jetWt_down[i][mask_pt_bounds[i]] = p_jetWt[mask_pt_bounds[i]]+2*(p_jetWt_down[i][mask_pt_bounds[i]]-p_jetWt[mask_pt_bounds[i]])
                compute_event_btag_weight_shape(jets.offsets, p_jetWt_down[i], eventweight_btag_down[i])

        if debug:
            for evtid in debug_event_ids:
                idx = np.where(scalars["event"] == evtid)[0][0]
                print("jets for b tag")
                jaggedstruct_print(jets, idx,
                                   ["pt", "eta", "phi","hadronFlavour", "btagDeepB", "jetId", "puId","qgl"])
                print(i,eventweight_btag_up[i][idx],eventweight_btag_down[i][idx])
        #print(p_jetWt_up[i][mask_pt_bounds[i]],jets.pt[mask_pt_bounds[i]],jets.eta[mask_pt_bounds[i]],jets.btagDeepB[mask_pt_bounds[i]])
        #print(p_jetWt_down[i][mask_pt_bounds[i]],jets.pt[mask_pt_bounds[i]],jets.eta[mask_pt_bounds[i]],jets.btagDeepB[mask_pt_bounds[i]])
    return eventweight_btag , eventweight_btag_up, eventweight_btag_down

def get_factorized_btag_weights_shape(jets, evaluator, era, scalars, pt_cut):
    tag_name = 'DeepCSV_'+era
    nev = jets.numevents()
    jet_pt = numpy.copy(jets.pt)
    jet_pt[(jets.pt > 1000.)]=1000.
    pt_eta_mask = NUMPY_LIB.logical_or((NUMPY_LIB.abs(jets.eta)>2.4), (jets.pt < pt_cut))
    p_jetWt = NUMPY_LIB.ones(len(jets.pt), dtype=NUMPY_LIB.float32)
    eventweight_btag = NUMPY_LIB.ones(nev)
    mask_pt_bounds = NUMPY_LIB.logical_and(NUMPY_LIB.logical_not(pt_eta_mask), (jets.pt > 1000. ))
    # Code help from https://github.com/chreissel/hepaccelerate/blob/mass_fit/lib_analysis.py#L118
    # Code help from https://gitlab.cern.ch/uhh-cmssw/CAST/blob/master/BTaggingWeight/plugins/BTaggingReShapeProducer.cc
    
    p_jetWt = evaluator[tag_name].eval('central', jets.hadronFlavour, NUMPY_LIB.abs(jets.eta), jet_pt, jets.btagDeepB,True)
        
    #print("p_JetWt before", p_jetWt, p_jetWt.mean(), p_jetWt.std())
    p_jetWt[pt_eta_mask] = 1.
    #print("p_JetWt after", p_jetWt, p_jetWt.mean(), p_jetWt.std())
    
    compute_event_btag_weight_shape(jets.offsets, p_jetWt, eventweight_btag)
    #print("eventweight_btag", eventweight_btag, eventweight_btag.mean(), eventweight_btag.std())
    if debug:
        for evtid in debug_event_ids:
            idx = np.where(scalars["event"] == evtid)[0][0]
            print("jets for b tag")
            jaggedstruct_print(jets, idx,
                               ["pt", "eta", "phi","hadronFlavour", "btagDeepB", "jetId", "puId","qgl"])
            print(eventweight_btag[idx])
    #not all syst are for all flavours
    # bFlav - jes, lf, hfstats1, hfstats2
    # cFlav - cferr1, cferr2
    # lFlav - jes, hf, lfstats1, lfstats2
    tag_sys = ['jes', 'lf', 'hfstats1', 'hfstats2', 'cferr1', 'cferr2', 'hf',  'lfstats1', 'lfstats2']
    p_jetWt_up= []
    p_jetWt_down=[]
    eventweight_btag_up = []
    eventweight_btag_down = []
    for i in range(0,9): #0 is for b flav, 1 is c flav and 2 is udsg
        p_jetWt_up.append(NUMPY_LIB.ones(len(jets.pt)))
        p_jetWt_down.append(NUMPY_LIB.ones(len(jets.pt)))
        eventweight_btag_up.append(NUMPY_LIB.ones(nev))
        eventweight_btag_down.append(NUMPY_LIB.ones(nev))
    #print(evaluator[tag_name].evaluator.keys())
    for i in range(0,9):
        for sdir in ['up','down']:
            tsys_name = sdir + '_' + tag_sys[i]
            SF_btag = evaluator[tag_name].eval(tsys_name, jets.hadronFlavour, NUMPY_LIB.abs(jets.eta), jet_pt, jets.btagDeepB, True)
            if sdir == 'up':
                p_jetWt_up[i]*=SF_btag
                p_jetWt_up[i][pt_eta_mask] = 1.
                #For jets with pt > 1000., evaluate with pt =1000. (done automatically) and inflate to double the systematic
                # based on https://github.com/cms-sw/cmssw/blob/master/CondTools/BTau/src/BTagCalibrationReader.cc#L170
                p_jetWt_up[i][mask_pt_bounds] = p_jetWt[mask_pt_bounds]+2*(p_jetWt_up[i][mask_pt_bounds]-p_jetWt[mask_pt_bounds])
                compute_event_btag_weight_shape(jets.offsets, p_jetWt_up[i], eventweight_btag_up[i])
            else:
                p_jetWt_down[i]*=SF_btag
                p_jetWt_down[i][pt_eta_mask] = 1.
                #For jets with pt > 1000., evaluate with pt =1000. (done automatically) and inflate to double the systematic
                # based on https://github.com/cms-sw/cmssw/blob/master/CondTools/BTau/src/BTagCalibrationReader.cc#L170
                p_jetWt_down[i][mask_pt_bounds] = p_jetWt[mask_pt_bounds]+2*(p_jetWt_down[i][mask_pt_bounds]-p_jetWt[mask_pt_bounds])
                compute_event_btag_weight_shape(jets.offsets, p_jetWt_down[i], eventweight_btag_down[i])
                    
        if debug:
            for evtid in debug_event_ids:
                idx = np.where(scalars["event"] == evtid)[0][0]
                print("jets for b tag")
                jaggedstruct_print(jets, idx,
                                   ["pt", "eta", "phi","hadronFlavour", "btagDeepB", "jetId", "puId","qgl"])
                print(i,eventweight_btag_up[i][idx],eventweight_btag_down[i][idx])
                #print(p_jetWt_up[i][mask_pt_bounds],jets.pt[mask_pt_bounds],jets.eta[mask_pt_bounds],jets.btagDeepB[mask_pt_bounds])
                #print(p_jetWt_down[i][mask_pt_bounds],jets.pt[mask_pt_bounds],jets.eta[mask_pt_bounds],jets.btagDeepB[mask_pt_bounds])
    return eventweight_btag , eventweight_btag_up, eventweight_btag_down

#qgl reweighting
def get_qglWeights(jets, jet_attrs, ret_mu, func, dataset_name):
    qglw = NUMPY_LIB.ones(jets.numevents())
    qglw_up = NUMPY_LIB.ones(jets.numevents())
    psgen = 0
    if "herwig" in dataset_name:
        psgen = 1
    leading_jet = jets.select_nth(
                0, ret_mu["selected_events"], None,
                attributes=jet_attrs)
    subleading_jet = jets.select_nth(
                1, ret_mu["selected_events"], None,
                attributes=jet_attrs)
    qglj1w = NUMPY_LIB.array(func.qglJetWeight(
                    NUMPY_LIB.asnumpy(leading_jet["partonFlavour"]),
                    NUMPY_LIB.asnumpy(leading_jet["pt"]),
                    NUMPY_LIB.asnumpy(leading_jet["eta"]),
                    NUMPY_LIB.asnumpy(leading_jet["qgl"]), psgen))
    qglj2w = NUMPY_LIB.array(func.qglJetWeight(
                    NUMPY_LIB.asnumpy(subleading_jet["partonFlavour"]),
                    NUMPY_LIB.asnumpy(subleading_jet["pt"]),
                    NUMPY_LIB.asnumpy(subleading_jet["eta"]),
                    NUMPY_LIB.asnumpy(subleading_jet["qgl"]), psgen))
    qglw = qglj1w*qglj2w
    qglw_down = 2.0*qglw - 1.0
    return qglw, qglw_up, qglw_down

#STXS uncertainties
#https://cms-nanoaod-integration.web.cern.ch/integration/master-cmsswmaster/mc102X_doc.html
#refs: https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG2/STXS/Classification/-/blob/master/HiggsTemplateCrossSections.h https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG2/STXS/VBF-Uncertainties
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsWG/SignalModelingTools
@numba.njit(parallel=True)
def compute_stxs_unc(stxs_bin, stxs_unc, stxs_up, stxs_down):
    for iev in numba.prange(stxs_up.shape[0]):
        stxs_up[iev] = stxs_unc[stxs_bin[iev]-200]+1.0
        stxs_down[iev] = 2.0 - stxs_up[iev]

@numba.njit(parallel=True)
def compute_eff_product_cpu(offsets, jet_pt, subjet_pt, jet_pt_mask, jets_mask_passes_id, jets_eff, out_proba):
    #loop over events in parallel
    for iev in numba.prange(len(offsets)-1):
        p_tot = 1.0
        #loop over jets in event
        for ij in range(offsets[iev], offsets[iev+1]):
            if (not jet_pt_mask[ij]) or (jet_pt[ij] < subjet_pt[iev]):
                continue
            this_jet_passes = jets_mask_passes_id[ij]
            if this_jet_passes:
                p_tot *= jets_eff[ij]
            else:
                p_tot *= 1.0 - jets_eff[ij]
        out_proba[iev] = p_tot

@cuda.jit
def compute_eff_product_cudakernel(offsets, jet_pt, subjet_pt, jet_pt_mask, jets_mask_passes_id, jets_eff, out_proba):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for iev in range(xi, offsets.shape[0]-1, xstride):
        p_tot = np.float32(1.0)
        #loop over jets in event
        for ij in range(offsets[iev], offsets[iev+1]):
            if (not jet_pt_mask[ij]) or (jet_pt[ij] < subjet_pt[iev]): continue
            this_jet_passes = jets_mask_passes_id[ij]
            if this_jet_passes:
                p_tot *= jets_eff[ij]
            else:
                p_tot *= 1.0 - jets_eff[ij]
        out_proba[iev] = p_tot

def get_selected_electrons(electrons, pt_cut, eta_cut, id_type):
    if id_type == "mvaFall17V2Iso_WP90":
        passes_id = electrons.mvaFall17V2Iso_WP90 == 1
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

def compute_jet_raw_pt(jets):
    """
    Computs the raw pt of a jet.
    """
    raw_pt = jets.pt * (1.0 - jets.rawFactor)
    return raw_pt

def sum_four_vectors(objects, size):
    px = NUMPY_LIB.zeros(size, dtype=np.float32)
    py = NUMPY_LIB.zeros(size, dtype=np.float32)
    pz = NUMPY_LIB.zeros(size, dtype=np.float32)
    e = NUMPY_LIB.zeros(size, dtype=np.float32)
    px_total = NUMPY_LIB.zeros(size, dtype=np.float32)
    py_total = NUMPY_LIB.zeros(size, dtype=np.float32)
    pz_total = NUMPY_LIB.zeros(size, dtype=np.float32)
    e_total = NUMPY_LIB.zeros(size, dtype=np.float32)
    for obj in objects:
        px, py, pz, e = ha.spherical_to_cartesian(
            obj["pt"], obj["eta"], obj["phi"], obj["mass"])
        px_total += px
        py_total += py
        pz_total += pz
        e_total += e
    pt_total, eta_total, phi_total, mass_total = ha.cartesian_to_spherical(
        px_total, py_total, pz_total, e_total)
    return {
        "pt": pt_total,
        "eta": eta_total,
        "phi": phi_total,
        "mass_total": mass_total
    }

def compute_inv_mass(objects, mask_events, mask_objects, use_cuda):
    inv_mass = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    pt_total = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    if use_cuda:
        compute_inv_mass_cudakernel[32, 1024](
            objects.offsets, objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
        cuda.synchronize()
    else:
        compute_inv_mass_kernel(objects.offsets,
            objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
    return inv_mass, pt_total

@numba.njit(parallel=True, fastmath=True)
def compute_inv_mass_kernel(offsets, pts, etas, phis, masses, mask_events, mask_objects, out_inv_mass, out_pt_total):
    for iev in numba.prange(offsets.shape[0]-1):
        if mask_events[iev]:
            start = np.uint64(offsets[iev])
            end = np.uint64(offsets[iev + 1])
            
            px_total = np.float32(0.0)
            py_total = np.float32(0.0)
            pz_total = np.float32(0.0)
            e_total = np.float32(0.0)
            
            for iobj in range(start, end):
                if mask_objects[iobj]:
                    pt = pts[iobj]
                    eta = etas[iobj]
                    phi = phis[iobj]
                    mass = masses[iobj]

                    px, py, pz, e = ha.spherical_to_cartesian(pt, eta, phi, mass) 
                    px_total += px 
                    py_total += py 
                    pz_total += pz 
                    e_total += e

            inv_mass = np.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
            pt_total = np.sqrt(px_total**2 + py_total**2)
            out_inv_mass[iev] = inv_mass
            out_pt_total[iev] = pt_total

@cuda.jit
def compute_inv_mass_cudakernel(offsets, pts, etas, phis, masses, mask_events, mask_objects, out_inv_mass, out_pt_total):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for iev in range(xi, offsets.shape[0]-1, xstride):
        if mask_events[iev]:
            start = np.uint64(offsets[iev])
            end = np.uint64(offsets[iev + 1])
            
            px_total = np.float32(0.0)
            py_total = np.float32(0.0)
            pz_total = np.float32(0.0)
            e_total = np.float32(0.0)
            
            for iobj in range(start, end):
                if mask_objects[iobj]:
                    pt = pts[iobj]
                    eta = etas[iobj]
                    phi = phis[iobj]
                    mass = masses[iobj]

                    px, py, pz, e = ha.spherical_to_cartesian_devfunc(pt, eta, phi, mass) 
                    
                    px_total += px 
                    py_total += py 
                    pz_total += pz 
                    e_total += e

            inv_mass = math.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
            pt_total = math.sqrt(px_total**2 + py_total**2)
            out_inv_mass[iev] = inv_mass
            out_pt_total[iev] = pt_total

def fill_with_weights(values, weight_dict, mask, bins):
    ret = {}
    vals = values
    for wn in weight_dict.keys():
        _weights = weight_dict[wn]
        ret[wn] = get_histogram(vals, _weights, bins, mask)
    return ret

def update_histograms_systematic(hists, hist_name, systematic_name, target_histogram):

    if hist_name not in hists:
        hists[hist_name] = {}

    if systematic_name[0] == "nominal" or systematic_name == "nominal":
        hists[hist_name].update(target_histogram)
    else:
        if systematic_name[1] == "":
            syst_string = systematic_name[0]
        else:
            syst_string = systematic_name[0] + "__" + systematic_name[1]
        target_histogram = {syst_string: target_histogram["nominal"]}
        hists[hist_name].update(target_histogram)

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0

def fix_large_weights(weights, maxw=10.0):
    weights[weights > maxw] = maxw
    weights[:] = weights[:] / NUMPY_LIB.mean(weights)

def compute_pu_weights(pu_corrections_target, weights, mc_nvtx, reco_nvtx):
    mc_nvtx = NUMPY_LIB.array(mc_nvtx, dtype=NUMPY_LIB.float32)
    pu_edges, (values_nom, values_up, values_down) = pu_corrections_target

    pu_edges = NUMPY_LIB.array(pu_edges, dtype=NUMPY_LIB.float32)

    src_pu_hist = get_histogram(mc_nvtx, weights, pu_edges)
    norm = sum(src_pu_hist.contents)
    values_target = src_pu_hist.contents/norm

    ratio = values_nom / values_target
    remove_inf_nan(ratio)
    pu_weights = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges, dtype=NUMPY_LIB.float32),
        NUMPY_LIB.array(ratio, dtype=NUMPY_LIB.float32), pu_weights)
    fix_large_weights(pu_weights) 
     
    ratio_up = values_up / values_target
    remove_inf_nan(ratio_up)
    pu_weights_up = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges, dtype=NUMPY_LIB.float32),
        NUMPY_LIB.array(ratio_up, dtype=NUMPY_LIB.float32), pu_weights_up)
    fix_large_weights(pu_weights_up) 
    
    ratio_down = values_down / values_target
    remove_inf_nan(ratio_down)
    pu_weights_down = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges, dtype=NUMPY_LIB.float32),
        NUMPY_LIB.array(ratio_down, dtype=NUMPY_LIB.float32), pu_weights_down)
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
    #pvsel = pvsel & (scalars["PV_ndof"] > parameters["NdfPV"])
    #pvsel = pvsel & (scalars["PV_z"] < parameters["zPV"])

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

#genEventSumw and genEventSumw2 for nanoAODv5 and before, genEventSumw_ and genEventSumw2_ for nanoAODv6
def get_gen_sumweights(filenames):
    sumw = 0
    sumw2 = 0
    for fi in filenames:
        ff = uproot.open(fi)
        bl = ff.get("Runs")
        try:
            print("nanoAODv5-like genEventSumw and genEventSumw2 used")
            arr = bl.array("genEventSumw")
            arr2 = bl.array("genEventSumw2")
        except KeyError:
            print("nanoAODv6-like genEventSumw_ and genEventSumw2_ used")
            arr = bl.array("genEventSumw_")
            arr2 = bl.array("genEventSumw2_")
        sumw += arr.sum()
        sumw2 += arr2.sum()
    return sumw, sumw2

def get_lumis(filenames):
    runs = []
    luminosityBlock = []
    for fi in filenames:
        ff = uproot.open(fi)
        bl = ff.get("LuminosityBlocks")
        arr = bl.array("run")
        arr2 = bl.array("luminosityBlock")
        runs += [arr]
        luminosityBlock += [arr2]
    return np.hstack(runs), np.hstack(luminosityBlock)

"""
Applies Geofit corrections on the selected two muons, returns the corrected pt
    
    returns: nothing
"""
def do_geofit_corrections(
    miscvariables,
    muons,
    dataset_era):
    years = int(dataset_era)*NUMPY_LIB.ones(len(muons.pt), dtype=NUMPY_LIB.int32)
    muon_pt_corr = miscvariables.ptcorrgeofit(
        NUMPY_LIB.asnumpy(muons.dxybs),
        NUMPY_LIB.asnumpy(muons.pt),
        NUMPY_LIB.asnumpy(muons.eta),
        NUMPY_LIB.asnumpy(muons.charge),
        years
    )
    muons.pfRelIso03_chg[:] = muons.pt[:] 
    muons.pt[:] = muon_pt_corr[:]
    return

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
    muons.miniPFRelIso_chg[:] = muons.pt[:]
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

@numba.njit('float32[:], float32[:], float32[:]', parallel=True, fastmath=True)
def deltaphi_cpu(phi1, phi2, out_dphi):
    for iev in numba.prange(len(phi1)):
        out_dphi[iev] = backend_cpu.deltaphi(phi1[iev], phi2[iev])

@cuda.jit
def deltaphi_cudakernel(phi1, phi2, out_dphi):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, len(phi1), xstride):
        out_dphi[iev] = backend_cuda.deltaphi_devfunc(phi1[iev], phi2[iev]) 

@numba.njit(parallel=True, fastmath=True)
def get_theoryweights_cpu(offsets, variations, index, out_var):
    #loop over events
    for iev in numba.prange(len(offsets) - 1):
        if(offsets[iev]+index < len(variations)):
            out_var[iev] = variations[offsets[iev]+index]

@cuda.jit
def get_theoryweights_cuda(offsets, variations, index, out_var):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    #loop over events
    for iev in range(xi, len(offsets) - 1, xstride):
        out_var[iev] = variations[offsets[iev]+index]

def get_theoryweights(offsets, variations, index, out_var, use_cuda):
    if use_cuda:
        get_theoryweights_cuda[32,1014](offsets, variations, index, out_var)
        cuda.synchronize()
    else:
        get_theoryweights_cpu(offsets, variations, index, out_var)

# Custom kernels to get the pt of the genHiggs
def genhpt(genpart, mask, use_cuda):
    nevt = genpart.numevents()
    assert(mask.shape == genpart.status.shape)
    vals_out = np.zeros(nevt, dtype=np.float32)
    if not use_cuda:
        genhpt_cpu(
            nevt, genpart.offsets, genpart.pdgId, genpart.status, genpart.pt, mask, vals_out
        )
    else:
        genhpt_cpu(
            nevt,
            NUMPY_LIB.asnumpy(genpart.offsets),
            NUMPY_LIB.asnumpy(genpart.pdgId),
            NUMPY_LIB.asnumpy(genpart.status),
            NUMPY_LIB.asnumpy(genpart.pt),
            NUMPY_LIB.asnumpy(mask),
            vals_out
        )

    return vals_out

@numba.njit(parallel=True, fastmath=True)
def genhpt_cpu(nevt, genparts_offsets, pdgid, status, pt, mask, out_genhpt):
    #loop over events
    for iev in numba.prange(nevt):
        gen_Higgs_pt = -1
        #loop over genpart, get the first particle in the event that matches the mask
        for igenpart in range(genparts_offsets[iev], genparts_offsets[iev + 1]):
            if mask[igenpart]:
                gen_Higgs_pt = pt[igenpart]
                break 
        out_genhpt[iev] = gen_Higgs_pt

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
            if idx_gp >= 0 and len(genparts_pt) > (genparts_offsets[iev] + idx_gp):
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
    px, py, pz, e = ha.spherical_to_cartesian(pt, eta, phi, mass)
    return {"px": px, "py": py, "pz": pz, "e": e}

def rapidity(e, pz):
    return 0.5*NUMPY_LIB.log((e + pz) / (e - pz))

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
    pt, eta, phi, mass = ha.cartesian_to_spherical(px, py, pz, e)
    rap = rapidity(e, pz)
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass, "rapidity": rap}

"""
Given two arrays of objects with the same length, computes the dr = sqrt(deta^2+dphi^2) between them.
    obj1: array of spherical coordinates (pt, eta, phi, m) for the first object
    obj2: array of spherical coordinates for the second object

    returns: arrays of deta, dphi, dr
"""
def deltar(obj1, obj2, use_cuda):
    deta = obj1["eta"] - obj2["eta"]
    dphi = NUMPY_LIB.zeros(len(deta), dtype=NUMPY_LIB.float32)
    if use_cuda:
        deltaphi_cudakernel[32,1024](obj1["phi"],obj2["phi"],dphi)
        cuda.synchronize()
    else:
        deltaphi_cpu(obj1["phi"], obj2["phi"], dphi)
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
    'Zep' - zeppenfeld variable with pseudorapidity
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
def dnn_variables(hrelresolution, miscvariables, leading_muon, subleading_muon, leading_jet, subleading_jet, nsoft, n_sel_softjet, n_sel_HTsoftjet, n_sel_HTsoftjet2, masses, dataset_name, use_cuda):
    nev = len(leading_muon["pt"])
    #delta eta, phi and R between two muons
    mm_deta, mm_dphi, mm_dr = deltar(leading_muon, subleading_muon, use_cuda)
    
    #delta eta between jets 
    jj_deta = leading_jet["eta"] - subleading_jet["eta"]
    jj_dphi = leading_jet["phi"] - subleading_jet["phi"]
    jj_dphi_mod = NUMPY_LIB.zeros(len(jj_dphi), dtype=NUMPY_LIB.float32)

    if use_cuda:
        deltaphi_cudakernel[32,1024](leading_jet["phi"],subleading_jet["phi"], jj_dphi_mod)
        cuda.synchronize()
    else:
        deltaphi_cpu(leading_jet["phi"],subleading_jet["phi"], jj_dphi_mod)

    #jj_dphi_mod = NUMPY_LIB.mod(jj_dphi + math.pi, math.pi)
    
    #muons in cartesian, create dimuon system 
    m1 = to_cartesian(leading_muon)    
    m2 = to_cartesian(subleading_muon)    
    mm = {k: m1[k] + m2[k] for k in ["px", "py", "pz", "e"]}
    mm_sph = to_spherical(mm)

    #mass resolution
    if not (hrelresolution is None):
        Higgs_mrelreso = NUMPY_LIB.array(hrelresolution.compute(
            NUMPY_LIB.asnumpy(leading_muon["miniPFRelIso_chg"]),
            NUMPY_LIB.asnumpy(leading_muon["eta"]),
            NUMPY_LIB.asnumpy(subleading_muon["pt"]),
            NUMPY_LIB.asnumpy(subleading_muon["eta"])))
    else:
        Higgs_mrelreso = NUMPY_LIB.zeros(nev, dtype=NUMPY_LIB.float32)

    #jets in cartesian, create dijet system 
    j1 = to_cartesian(leading_jet)
    j2 = to_cartesian(subleading_jet)
    leading_jet["rapidity"] = rapidity(j1["e"], j1["pz"]) 
    subleading_jet["rapidity"] = rapidity(j2["e"], j2["pz"]) 
    jj = {k: j1[k] + j2[k] for k in ["px", "py", "pz", "e"]}
    jj_sph = to_spherical(jj)
  
    #create dimuon-dijet system 
    mmjj = {k: j1[k] + j2[k] + m1[k] + m2[k] for k in ["px", "py", "pz", "e"]} 
    mmjj_sph = to_spherical(mmjj)
    #compute deletaEta between Higgs and jet
    EtaHQs = []
    for jet in [leading_jet, subleading_jet]:
        EtaHQs += [NUMPY_LIB.abs(mm_sph["eta"] - jet["eta"])]
    EtaHQ = NUMPY_LIB.vstack(EtaHQs)
    minEtaHQ = NUMPY_LIB.min(EtaHQ, axis=0)

    #compute deldPhi between Higgs and jet
    PhiHQs = []
    for jet in [leading_jet, subleading_jet]:
        PhiHQs += [NUMPY_LIB.abs(mm_sph["phi"] - jet["phi"])]
    PhiHQ = NUMPY_LIB.vstack(PhiHQs)
    minPhiHQ = NUMPY_LIB.min(PhiHQ, axis=0)
    #compute deltaR between all muons and jets
    dr_mjs = []
    for mu in [leading_muon, subleading_muon]:
        for jet in [leading_jet, subleading_jet]:
            _, _, dr_mj = deltar(mu, jet, use_cuda)
            dr_mjs += [dr_mj]
    dr_mj = NUMPY_LIB.vstack(dr_mjs)
    dRmin_mj = NUMPY_LIB.min(dr_mj, axis=0) 
    dRmax_mj = NUMPY_LIB.max(dr_mj, axis=0) 
    #compute deltaR between dimuon system and both jets 
    dr_mmjs = []
    for jet in [leading_jet, subleading_jet]:
        _, _, dr_mmj = deltar(mm_sph, jet, use_cuda)
        dr_mmjs += [dr_mmj]
    dr_mmj = NUMPY_LIB.vstack(dr_mmjs)
    dRmin_mmj = NUMPY_LIB.min(dr_mmj, axis=0) 
    dRmax_mmj = NUMPY_LIB.max(dr_mmj, axis=0)
   
    #Zeppenfeld variable
    Zep = (mm_sph["eta"] - 0.5*(leading_jet["eta"] + subleading_jet["eta"]))
    Zep_rapidity = (mm_sph["rapidity"] - 0.5*(leading_jet["rapidity"] + subleading_jet["rapidity"]))/(leading_jet["rapidity"]-subleading_jet["rapidity"])

    #Collin-Soper frame variable
    cthetaCS = 2*(m1["pz"] * m2["e"] - m1["e"]*m2["pz"]) / (mm_sph["mass"] * NUMPY_LIB.sqrt(NUMPY_LIB.power(mm_sph["mass"], 2) + NUMPY_LIB.power(mm_sph["pt"], 2)))

    #Collin-Soper frame variable Pisa definition
    CS_theta, CS_phi = miscvariables.csanglesPisa(
            NUMPY_LIB.asnumpy(leading_muon["pt"]),
            NUMPY_LIB.asnumpy(leading_muon["eta"]),
            NUMPY_LIB.asnumpy(leading_muon["phi"]),
            NUMPY_LIB.asnumpy(leading_muon["mass"]),
            NUMPY_LIB.asnumpy(subleading_muon["pt"]),
            NUMPY_LIB.asnumpy(subleading_muon["eta"]),
            NUMPY_LIB.asnumpy(subleading_muon["phi"]),
            NUMPY_LIB.asnumpy(subleading_muon["mass"]),
            NUMPY_LIB.asnumpy(subleading_muon["charge"]),
            )

    ret = {
        "leading_muon_pt": leading_muon["pt"],
        "leading_muon_pt_nanoAOD": leading_muon["miniPFRelIso_chg"],
        "leading_muon_pt_roch_fsr": leading_muon["pfRelIso03_chg"],
        "leading_muon_eta": leading_muon["eta"],
        "leading_muon_phi": leading_muon["phi"],
        "leading_muon_charge": leading_muon["charge"],
        #"leading_muon_mass": leading_muon["mass"],
        "subleading_muon_pt": subleading_muon["pt"],
        "subleading_muon_pt_nanoAOD": subleading_muon["miniPFRelIso_chg"],
        "subleading_muon_pt_roch_fsr": subleading_muon["pfRelIso03_chg"],
        "subleading_muon_eta": subleading_muon["eta"],
        "subleading_muon_phi": subleading_muon["phi"],
        "subleading_muon_charge": subleading_muon["charge"],
        #"subleading_muon_mass": subleading_muon["mass"],
        "dEtamm": mm_deta, "dPhimm": mm_dphi, "dRmm": mm_dr,
        "M_jj": jj_sph["mass"], "pt_jj": jj_sph["pt"], "eta_jj": jj_sph["eta"], "phi_jj": jj_sph["phi"],
        "M_mmjj": mmjj_sph["mass"], "eta_mmjj": mmjj_sph["eta"], "phi_mmjj": mmjj_sph["phi"],
        "dEta_jj": jj_deta,
        "dEta_jj_abs": NUMPY_LIB.abs(jj_deta),
        "dPhi_jj": jj_dphi,
        "dPhi_jj_mod": jj_dphi_mod,
        "dPhi_jj_mod_abs": NUMPY_LIB.abs(jj_dphi_mod),
        "leadingJet_pt": leading_jet["pt"],
        "subleadingJet_pt": subleading_jet["pt"],
        "leadingJet_eta": leading_jet["eta"],
        "subleadingJet_eta": subleading_jet["eta"],
        "dRmin_mj": dRmin_mj,
        "dRmax_mj": dRmax_mj,
        "dRmin_mmj": dRmin_mmj,
        "dRmax_mmj": dRmax_mmj,
        "Zep": Zep,
        "Zep_rapidity": Zep_rapidity,
        "leadingJet_qgl": leading_jet["qgl"],
        "subleadingJet_qgl": subleading_jet["qgl"], 
        "CS_phi": CS_phi,
        "CS_theta": CS_theta,
        "cthetaCS": cthetaCS,
        "softJet5": nsoft,
        "Higgs_pt": mm_sph["pt"],
        "Higgs_eta": mm_sph["eta"],
        "Higgs_rapidity": mm_sph["rapidity"],
        "Higgs_mass": mm_sph["mass"],
        #DNN pisa variable
        "Mqq_log": NUMPY_LIB.log(jj_sph["mass"] ),
        "Rpt": mmjj_sph["pt"]/(mm_sph["pt"]+leading_jet["pt"]+subleading_jet["pt"]),
        "qqDeltaEta": NUMPY_LIB.abs(jj_deta),
        "log(ll_zstar)": NUMPY_LIB.log(NUMPY_LIB.abs((mm_sph["rapidity"] - 0.5*(leading_jet["rapidity"] + subleading_jet["rapidity"]))/(leading_jet["rapidity"]-subleading_jet["rapidity"]))),
        "NSoft5": n_sel_softjet,
        "HTSoft2": n_sel_HTsoftjet2,
        "minEtaHQ": minEtaHQ,
        "minPhiHQ": minPhiHQ,
        "log(Higgs_pt)": NUMPY_LIB.log(mm_sph["pt"]),
        "Mqq": jj_sph["mass"],
        "QJet0_pt_touse": leading_jet["pt"],
        "QJet1_pt_touse": subleading_jet["pt"],
        "QJet0_eta": leading_jet["eta"],
        "QJet1_eta": subleading_jet["eta"],
        "QJet0_phi": leading_jet["phi"],
        "QJet1_phi": subleading_jet["phi"],
        "QJet0_qgl": leading_jet["qgl"],
        "QJet1_qgl": subleading_jet["qgl"],
        "Higgs_mRelReso": Higgs_mrelreso,
        "HTSoft5": n_sel_HTsoftjet,
    }
   
    nw = len(leading_jet["eta"])
    for imass in masses:
        fixm = NUMPY_LIB.full(nw, -1, dtype=NUMPY_LIB.float32)
        movem = NUMPY_LIB.full(nw, 125.0-imass, dtype=NUMPY_LIB.float32)
        mhfordnn(fixm, mm_sph["mass"],movem)
        ret.update( {"Higgs_m_"+str(imass): fixm} )
        ret.update( {"Higgs_mReso_"+str(imass): fixm*Higgs_mrelreso,} )
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
def compute_fill_dnn(
    hrelresolution,
    miscvariables,
    parameters,
    use_cuda,
    dnn_presel,
    dnn_model,
    dnn_normfactors,
    dnnPisa_models,
    dnnPisa_normfactors1,
    dnnPisa_normfactors2,
    scalars,
    leading_muon,
    subleading_muon,
    leading_jet,
    subleading_jet,
    num_jets,
    num_jets_btag,
    n_sel_softjet,
    n_sel_HTsoftjet,
    n_sel_HTsoftjet2,
    masses,
    dataset_name,
    dataset_era,
    is_mc):

    nev_dnn_presel = NUMPY_LIB.sum(dnn_presel)

    #for some reason, on the cuda backend, the sum does not return a simple number
    if use_cuda:
        nev_dnn_presel = int(NUMPY_LIB.asnumpy(nev_dnn_presel).flatten()[0])

    leading_muon_s = apply_mask(leading_muon, dnn_presel)
    subleading_muon_s = apply_mask(subleading_muon, dnn_presel)
    leading_jet_s = apply_mask(leading_jet, dnn_presel)
    subleading_jet_s = apply_mask(subleading_jet, dnn_presel)
    nsoft = scalars["SoftActivityJetNjets5"][dnn_presel]
    nsoftNew = n_sel_softjet[dnn_presel]
    HTsoft = n_sel_HTsoftjet[dnn_presel]
    HTsoft2 = n_sel_HTsoftjet2[dnn_presel]
    dnn_vars = dnn_variables(hrelresolution, miscvariables, leading_muon_s, subleading_muon_s, leading_jet_s, subleading_jet_s, nsoft, nsoftNew, HTsoft, HTsoft2, masses, dataset_name, use_cuda)
    # event-by-event mass resolution
    dpt1 = (leading_muon_s["ptErr"]*dnn_vars["Higgs_mass"]) / (2*leading_muon_s["pt"])
    dpt2 = (subleading_muon_s["ptErr"]*dnn_vars["Higgs_mass"]) / (2*subleading_muon_s["pt"])
    calibration = get_massErr_calib_factors(leading_muon_s["pt"], NUMPY_LIB.abs(leading_muon_s["eta"]), NUMPY_LIB.abs(subleading_muon_s["eta"]), dataset_era, is_mc)
    mm_massErr = NUMPY_LIB.sqrt(dpt1*dpt1 +dpt2*dpt2) * calibration
    dnn_vars["massErr"] = mm_massErr
    dnn_vars["massErr_rel"] = mm_massErr / dnn_vars["Higgs_mass"]

    if dataset_era == "2017":
    	dnn_vars["MET_pt"] = scalars["METFixEE2017_pt"][dnn_presel]
    else:
    	dnn_vars["MET_pt"] = scalars["MET_pt"][dnn_presel]
    dnn_vars["num_jets"] = num_jets[dnn_presel]
    dnn_vars["num_jets_btag"] = num_jets_btag[dnn_presel]

    year_var = float(dataset_era)*NUMPY_LIB.ones(nev_dnn_presel, dtype=NUMPY_LIB.float32)
    dnn_vars["year"] = year_var
 
    if (not (dnn_model is None)) and (nev_dnn_presel > 0) and parameters["do_dnn_cit"]:
        dnn_vars_arr = NUMPY_LIB.vstack([dnn_vars[k] for k in parameters["dnn_varlist_order"]]).T
        
        #Normalize the DNN with the normalization factors from preprocessing in training 
        dnn_vars_arr -= dnn_normfactors[0]
        dnn_vars_arr /= dnn_normfactors[1]

        #for TF, need to convert library to numpy, as it doesn't accept cupy arrays
        dnn_pred = NUMPY_LIB.array(dnn_model.predict(
            NUMPY_LIB.asnumpy(dnn_vars_arr),
            batch_size=dnn_vars_arr.shape[0])[:, 0]
        )
        if len(dnn_pred) > 0:
            print("dnn_pred", dnn_pred.min(), dnn_pred.max(), dnn_pred.mean(), dnn_pred.std())
        dnn_pred = NUMPY_LIB.array(dnn_pred, dtype=NUMPY_LIB.float32)
    else:
        dnn_pred = NUMPY_LIB.zeros(nev_dnn_presel, dtype=NUMPY_LIB.float32)

    #Pisa DNN
    dnnPisaComb_pred = NUMPY_LIB.zeros((len(masses), nev_dnn_presel), dtype=NUMPY_LIB.float32)
    dnnPisa_preds = NUMPY_LIB.zeros((len(dnnPisa_models)*len(masses), nev_dnn_presel), dtype=NUMPY_LIB.float32)
    if parameters["do_dnn_pisa"] and len(dnnPisa_models) > 0:
        imodel = 0
        for dnnPisa_model in dnnPisa_models:
            if (not (dnnPisa_model is None)) and nev_dnn_presel > 0:
                dnnPisa_vars1_arr = NUMPY_LIB.vstack([dnn_vars[k] for k in parameters["dnnPisa_varlist1_order"]]).T
                dnnPisa_vars1_arr -= dnnPisa_normfactors1[0]
                dnnPisa_vars1_arr /= dnnPisa_normfactors1[1]
                mass_index = 0
                for imass in masses:
                    dnnPisa_vars2_arr = NUMPY_LIB.vstack([dnn_vars[k] for k in parameters["dnnPisa_varlist2_order_"+str(imass)]]).T
                    dnnPisa_vars2_arr -= dnnPisa_normfactors2[0]
                    dnnPisa_vars2_arr /= dnnPisa_normfactors2[1]
                    dnnPisa_preds[imodel*len(masses)+mass_index, :] = NUMPY_LIB.array(dnnPisa_model.predict([
                    NUMPY_LIB.asnumpy(dnnPisa_vars1_arr), NUMPY_LIB.asnumpy(dnnPisa_vars2_arr)], batch_size=len(dnnPisa_vars1_arr))[:, 0])
                    mass_index += 1
            imodel += 1
        compute_dnnPisaComb(dnnPisaComb_pred, dnnPisa_preds, scalars["event"][dnn_presel], len(masses), use_cuda)

        dnnPisaComb_pred = NUMPY_LIB.arctanh(dnnPisaComb_pred)
        #### Calculating atanh is expensive, skipping for individual models#####
       
    if parameters["do_bdt_ucsd"]:
        hmmthetacs, hmmphics = miscvariables.csangles(
            NUMPY_LIB.asnumpy(leading_muon_s["pt"]),
            NUMPY_LIB.asnumpy(leading_muon_s["eta"]),
            NUMPY_LIB.asnumpy(leading_muon_s["phi"]),
            NUMPY_LIB.asnumpy(leading_muon_s["mass"]),
            NUMPY_LIB.asnumpy(subleading_muon_s["pt"]),
            NUMPY_LIB.asnumpy(subleading_muon_s["eta"]),
            NUMPY_LIB.asnumpy(subleading_muon_s["phi"]),
            NUMPY_LIB.asnumpy(subleading_muon_s["mass"]),
            NUMPY_LIB.asnumpy(leading_muon_s["charge"]),
            )
        dnn_vars["hmmthetacs"] = NUMPY_LIB.array(hmmthetacs)
        dnn_vars["hmmphics"] = NUMPY_LIB.array(hmmphics)

    dnn_vars["m1eta"] = NUMPY_LIB.array(leading_muon_s["eta"])
    dnn_vars["m2eta"] = NUMPY_LIB.array(subleading_muon_s["eta"])
    dnn_vars["m1ptOverMass"] = NUMPY_LIB.divide(leading_muon_s["pt"],dnn_vars["Higgs_mass"])
    dnn_vars["m2ptOverMass"] = NUMPY_LIB.divide(subleading_muon_s["pt"],dnn_vars["Higgs_mass"])
    return dnn_vars, dnn_pred, dnnPisa_preds, dnnPisaComb_pred

def get_massErr_calib_factors(pt1, abs_eta1, abs_eta2, era, is_mc):
    mode = "MC" if is_mc else "Data"
    label = "res_calib_{0}_{1}".format(mode, era)
    file_path = "data/res_calib/{0}.root".format(label)

    ext = extractor()
    ext.add_weight_sets(["{0} {0} {1}".format(label, file_path)])
    ext.finalize()

    evaluator = ext.make_evaluator()
    calib_factors = evaluator[label](NUMPY_LIB.asnumpy(pt1), NUMPY_LIB.asnumpy(abs_eta1), NUMPY_LIB.asnumpy(abs_eta2))
    calib_factors = NUMPY_LIB.array(calib_factors)

    return calib_factors

#based on https://github.com/cms-nanoAOD/nanoAOD-tools/blob/master/python/postprocessing/modules/jme/jetSmearer.py#L114
#and https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/PatUtils/interface/SmearedJetProducerT.h
def get_jer_smearfactors(pt_or_m, ratio_jet_genjet, msk_no_genjet, resos, resosfs, NUMPY_LIB, ha):
    
    #scale factor for matched jets
    smear_matched_n = 1.0 + (resosfs[:, 0] - 1.0) * ratio_jet_genjet
    smear_matched_u = 1.0 + (resosfs[:, 1] - 1.0) * ratio_jet_genjet
    smear_matched_d = 1.0 + (resosfs[:, 2] - 1.0) * ratio_jet_genjet

    #result vector
    smear_n = NUMPY_LIB.array(smear_matched_n)
    smear_u = NUMPY_LIB.array(smear_matched_u)
    smear_d = NUMPY_LIB.array(smear_matched_d)

    #compute random smearing for unmatched jets
    #note that we currently do not use a deterministic seed, this could be implemented
    rand_reso = NUMPY_LIB.clip(NUMPY_LIB.random.normal(loc=NUMPY_LIB.zeros_like(pt_or_m), scale=resos, size=len(pt_or_m)), -5, 5)
    
    smear_rnd_n = 1. + rand_reso * NUMPY_LIB.sqrt(NUMPY_LIB.clip(resosfs[:, 0]**2 - 1.0, 0, None))
    smear_rnd_u = 1. + rand_reso * NUMPY_LIB.sqrt(NUMPY_LIB.clip(resosfs[:, 1]**2 - 1.0, 0, None))
    smear_rnd_d = 1. + rand_reso * NUMPY_LIB.sqrt(NUMPY_LIB.clip(resosfs[:, 2]**2 - 1.0, 0, None))

    inds_no_genjet = NUMPY_LIB.nonzero(msk_no_genjet)[0]
    
    #set the smear factor for the unmatched jets
    ha.copyto_dst_indices(smear_n, smear_rnd_n[msk_no_genjet], inds_no_genjet)
    ha.copyto_dst_indices(smear_u, smear_rnd_u[msk_no_genjet], inds_no_genjet)
    ha.copyto_dst_indices(smear_d, smear_rnd_d[msk_no_genjet], inds_no_genjet)
    
    #in case smearing accidentally flipped a jet pt, we don't want that
    smear_n[(smear_n * pt_or_m) < 0.01] = 0.01
    smear_u[(smear_u * pt_or_m) < 0.01] = 0.01
    smear_d[(smear_d * pt_or_m) < 0.01] = 0.01

    #jets that have no genjet and the resolution in data is better than the one in MC (sf < 1)
    smear_n[msk_no_genjet & (resosfs[:, 0]<1.0)] = 1
    smear_u[msk_no_genjet & (resosfs[:, 1]<1.0)] = 1
    smear_d[msk_no_genjet & (resosfs[:, 2]<1.0)] = 1
    
    #Oddly this happens in batch jobs but not on login-1
    fix_inf_nan(smear_n, 1)
    fix_inf_nan(smear_u, 1)
    fix_inf_nan(smear_d, 1)

    return smear_n, smear_u, smear_d


class JetTransformer:
    def __init__(self, jets, scalars, parameters, jetmet_corrections, NUMPY_LIB, ha, use_cuda, is_mc, era, redoJEC):
        self.jets = jets
        self.scalars = scalars
        self.jetmet_corrections = jetmet_corrections
        self.NUMPY_LIB = NUMPY_LIB
        self.ha = ha 
        self.use_cuda = use_cuda
        self.is_mc = is_mc

        self.jets_rho = NUMPY_LIB.zeros_like(jets.pt)
        self.ha.broadcast(self.jets.offsets, scalars["fixedGridRhoFastjetAll"], self.jets_rho)
        
        # Get the uncorrected jet pt and mass
        self.raw_pt = (self.jets.pt * (1.0 - self.jets.rawFactor))
        self.raw_mass = (self.jets.mass * (1.0 - self.jets.rawFactor))

        self.jet_uncertainty_names = list(self.jetmet_corrections.jesunc.levels)
        self.jet_uncertainty_names.pop(self.jet_uncertainty_names.index("jes"))

        # Need to use the CPU for JEC/JER currently
        if self.use_cuda:
            self.raw_pt = self.NUMPY_LIB.asnumpy(self.raw_pt)
            self.eta = self.NUMPY_LIB.asnumpy(self.jets.eta)
            self.rho = self.NUMPY_LIB.asnumpy(self.jets_rho)
            self.area = self.NUMPY_LIB.asnumpy(self.jets.area)
        else:
            self.raw_pt = self.raw_pt
            self.eta = self.jets.eta
            self.rho = self.jets_rho
            self.area = self.jets.area

        if redoJEC:
            if self.is_mc:
                self.corr_jec = self.apply_jec_mc()
            else:
                self.corr_jec = self.apply_jec_data()
            self.corr_jec = self.NUMPY_LIB.array(self.corr_jec)
            self.pt_jec = self.NUMPY_LIB.array(self.raw_pt) * self.corr_jec
            self.jets.pt = self.pt_jec
        else:
            self.pt_jec = self.jets.pt 
            
        if self.is_mc:
            self.msk_no_genjet = (self.jets.genpt==0)
            self.jer_nominal, self.jer_up, self.jer_down = self.apply_jer()
            check_inf_nan(self.jer_nominal) 
            self.pt_jec_jer = self.pt_jec * self.jer_nominal
            self.pt_jec_jer_ms = (self.pt_jec + self.pt_jec_jer)/2.0

    def apply_jer(self, startfrom="pt_jec"):
        ptvec = getattr(self, startfrom)
        
        #This is done only on CPU
        resos = self.jetmet_corrections.jer.getResolution(JetPt=NUMPY_LIB.asnumpy(ptvec), JetEta=self.eta, Rho=self.rho) 
        resosfs = self.jetmet_corrections.jersf.getScaleFactor(JetPt=NUMPY_LIB.asnumpy(ptvec), JetEta=self.eta)

        #The following is done either on CPU or GPU
        resos = self.NUMPY_LIB.array(resos)
        resosfs = self.NUMPY_LIB.array(resosfs)

        dpt_jet_genjet = ptvec - self.jets.genpt
        dpt_jet_genjet[self.jets.genpt == 0] = 0
        ratio_jet_genjet_pt = dpt_jet_genjet / ptvec
        
        smear_n, smear_u, smear_d = get_jer_smearfactors(
            ptvec, ratio_jet_genjet_pt, self.msk_no_genjet, resos, resosfs, self.NUMPY_LIB, self.ha)
        return smear_n, smear_u, smear_d

    def apply_jec_mc(self):
        corr = self.jetmet_corrections.jec_mc.getCorrection(
            JetPt=self.raw_pt.copy(),
            Rho=self.rho,
            JetEta=self.eta,
            JetA=self.area)
        return corr

    def apply_jec_data(self):
        final_corr = self.NUMPY_LIB.zeros_like(self.jets.pt)

        #final correction is run-dependent, compute that for each run separately
        for run_idx in self.NUMPY_LIB.unique(self.scalars["run_index"]):
            
            if self.use_cuda:
                run_idx = int(run_idx)
            msk = self.scalars["run_index"] == run_idx
            
            #find the jets in the events that pass this run index cut
            jets_msk = self.NUMPY_LIB.zeros(self.jets.numobjects(), dtype=self.NUMPY_LIB.bool)
            self.ha.broadcast(self.jets.offsets, msk, jets_msk)
            inds_nonzero = self.NUMPY_LIB.nonzero(jets_msk)[0]

            #Evaluate jet correction (on CPU only currently)
            if self.use_cuda:
                jets_msk = self.NUMPY_LIB.asnumpy(jets_msk)
            run_name = runmap_numerical_r[run_idx]

            corr = self.jetmet_corrections.jec_data[run_name].getCorrection(
                JetPt=self.raw_pt[jets_msk].copy(),
                Rho=self.rho[jets_msk],
                JetEta=self.eta[jets_msk],
                JetA=self.area[jets_msk])
            if debug:
                print("run_idx=", run_idx, corr.mean(), corr.std())

            #update the final jet correction for the jets in the events in this run
            if len(inds_nonzero) > 0:
                self.ha.copyto_dst_indices(final_corr, corr, inds_nonzero)
        corr = final_corr
        return corr

    def apply_jec_unc(self, startfrom="pt_jec", uncertainty_name="Total"):
        ptvec = getattr(self, startfrom)

        idx_func = self.jetmet_corrections.jesunc.levels.index(uncertainty_name)
        jec_unc_func = self.jetmet_corrections.jesunc._funcs[idx_func]
        function_signature = self.jetmet_corrections.jesunc._funcs[idx_func].signature

        args = {
            "JetPt": self.NUMPY_LIB.array(ptvec),
            "JetEta": self.NUMPY_LIB.array(self.eta)
        }
        print("apply_jec_unc", startfrom, uncertainty_name, args["JetPt"])

        #Get the arguments in the required format
        func_args = tuple([args[s] for s in function_signature])

        #compute the JEC uncertainty
        jec_unc_vec = jec_unc_func(*func_args)
        return self.NUMPY_LIB.array(jec_unc_vec)

    def get_variated_pts(self, variation_name, jet_mask_bin, jet_genpt, startfrom="pt_jec_jer"):
        ptvec = getattr(self, startfrom)
        if variation_name in self.jet_uncertainty_names:
            corrs_up_down = self.NUMPY_LIB.array(self.apply_jec_unc(startfrom, variation_name), dtype=self.NUMPY_LIB.float32)
            return {
                (variation_name, "up"): ptvec*corrs_up_down[:, 0],
                (variation_name, "down"): ptvec*corrs_up_down[:, 1]
            }
        elif "jer" in variation_name :
            # Based in part from A. Rizzi's code: https://github.com/arizzi/PisaHmm/blob/49cb2a112f326b07dd133118d973be0901d45287/systematics.py
            jerSF = (self.pt_jec_jer-jet_genpt)/(ptvec-jet_genpt+(ptvec==jet_genpt)*(self.pt_jec_jer-ptvec))
            jerDownSF = ((ptvec*self.jer_down)-jet_genpt)/(ptvec-jet_genpt+(ptvec==jet_genpt)*10.)
            jerDown_pt = jet_genpt+ (ptvec - jet_genpt)*(jerDownSF/jerSF)
            return {
                ("jer", "up"): np.where(jet_mask_bin, ptvec, self.pt_jec_jer),
                ("jer", "down"): np.where(jet_mask_bin, ptvec, jerDown_pt)
            }
        elif variation_name == "nominal":
            return {("nominal", ""): ptvec}
        else:
            raise KeyError("Variation name {0} was not defined in JetMetCorrections corrections".format(variation_name))

@numba.njit(parallel=True, fastmath=True)
def get_genJetpt_cpu(reco_offsets, reco_pt, reco_genPartIdx, genparts_offsets, genparts_pt, out_reco_genpt):
    #loop over events
    for iev in numba.prange(len(reco_offsets) - 1):
        #loop over muons
        for imu in range(reco_offsets[iev], reco_offsets[iev + 1]):
            #get index of genparticle that reco particle was matched to
            idx_gp = reco_genPartIdx[imu]
            if idx_gp >= 0 and len(genparts_pt) > (genparts_offsets[iev] + idx_gp):
                genpt = genparts_pt[genparts_offsets[iev] + idx_gp]
                out_reco_genpt[imu] = genpt
            else :
                out_reco_genpt[imu] = reco_pt[imu]

@numba.njit(parallel=True, fastmath=True)
def get_leadtwo_jet_ind(reco_offsets, reco_pt, out_jet_ind0, out_jet_ind1):
    #loop over events
    for iev in numba.prange(len(reco_offsets) - 1):
        #loop over physics object
        pt0 = 0.0
        pt1 = 0.0
        ind0 = 0
        ind1 = 1
        for iobj in range(reco_offsets[iev], reco_offsets[iev + 1]):
            if pt0 < reco_pt[iobj]:
                pt0 = reco_pt[iobj]
                ind0 = iobj - reco_offsets[iev]
        for iobj in range(reco_offsets[iev], reco_offsets[iev + 1]):
            if (pt1 < reco_pt[iobj]) and pt0 > reco_pt[iobj]:
                pt1 = reco_pt[iobj]
                ind1 = iobj - reco_offsets[iev]
        out_jet_ind0[iev] = ind0 
        out_jet_ind1[iev] = ind1

def multiply_all(weight_list):
    ret = NUMPY_LIB.copy(weight_list[0])
    for w in weight_list[1:]:
        ret *= w
    return ret

def multiply_all_trig(weight_list):
    ones_array = NUMPY_LIB.ones_like(weight_list[0])
    ret_tmp = ones_array - NUMPY_LIB.copy(weight_list[0])
    for w in weight_list[1:]:
        ret_tmp *= ones_array - w
    ret = ones_array - ret_tmp
    return ret

def compute_lepton_sf(leading_muon, subleading_muon, lepsf_iso, lepsf_id, lepeff_trig_data, lepeff_trig_mc, use_cuda, dataset_era, NUMPY_LIB, debug):
    sfs_id = []
    sfs_iso = []
    effs_trig_data = []
    effs_trig_mc = []
    sfs_id_up = []
    sfs_id_down = []
    sfs_iso_up = []
    sfs_iso_down = []
    effs_trig_data_up = []
    effs_trig_data_down = []
    effs_trig_mc_up = []
    effs_trig_mc_down = []

    #compute weight for both leading and subleading muon
    for mu in [leading_muon, subleading_muon]:
        #lepton SF computed on CPU 
        if use_cuda:
            mu = {k: NUMPY_LIB.asnumpy(v) for k, v in mu.items()}
        pdgid = numpy.array(mu["pdgId"])
        
        #In 2016, the histograms are flipped
        if dataset_era == "2016":
            pdgid[:] = 11

        sf_iso = NUMPY_LIB.array(lepsf_iso.compute(pdgid, mu["pt"], mu["eta"]))
        sf_iso_err = NUMPY_LIB.array(lepsf_iso.compute_error(pdgid, mu["pt"], mu["eta"]))

        sf_id = NUMPY_LIB.array(lepsf_id.compute(pdgid, mu["pt"], mu["eta"]))
        sf_id_err = NUMPY_LIB.array(lepsf_id.compute_error(pdgid, mu["pt"], mu["eta"]))

        eff_trig_data = lepeff_trig_data.compute(pdgid, mu["pt"], NUMPY_LIB.abs(mu["eta"]))
        eff_trig_data_err = lepeff_trig_data.compute_error(pdgid, mu["pt"], NUMPY_LIB.abs(mu["eta"]))
        eff_trig_mc = lepeff_trig_mc.compute(pdgid, mu["pt"], NUMPY_LIB.abs(mu["eta"]))
        eff_trig_mc_err = lepeff_trig_mc.compute_error(pdgid, mu["pt"], NUMPY_LIB.abs(mu["eta"]))

        sf_id_up = (sf_id + sf_id_err)
        sf_id_down = (sf_id - sf_id_err)
        sf_iso_up = (sf_iso + sf_iso_err)
        sf_iso_down = (sf_iso - sf_iso_err)
        eff_trig_data_up = eff_trig_data + eff_trig_data_err
        eff_trig_data_down = eff_trig_data - eff_trig_data_err
        eff_trig_mc_up = eff_trig_mc + eff_trig_mc_err
        eff_trig_mc_down = eff_trig_mc - eff_trig_mc_err

        if debug:
            print("sf_iso: ", sf_iso.mean(), "+-", sf_iso.std())
            print("sf_id: ", sf_id.mean(), "+-", sf_id.std())
            print("sf_id_up: ", sf_id_up.mean(), "+-", sf_id_up.std())
            print("sf_id_down: ", sf_id_down.mean(), "+-", sf_id_down.std())
            print("sf_iso_up: ", sf_iso_up.mean(), "+-", sf_iso_up.std())
            print("sf_iso_down: ", sf_iso_down.mean(), "+-", sf_iso_down.std())

        sfs_id += [sf_id]
        sfs_iso += [sf_iso]
        effs_trig_data += [eff_trig_data]
        effs_trig_mc += [eff_trig_mc]

        sfs_id_up += [sf_id_up]
        sfs_id_down += [sf_id_down]
        sfs_iso_up += [sf_iso_up]
        sfs_iso_down += [sf_iso_down]
        effs_trig_data_up += [eff_trig_data_up]
        effs_trig_data_down += [eff_trig_data_down]
        effs_trig_mc_up += [eff_trig_mc_up]
        effs_trig_mc_down += [eff_trig_mc_down]
    
    #multiply all ID, iso, trigger weights for leading and subleading muons
    sf_id = multiply_all(sfs_id)
    sf_iso = multiply_all(sfs_iso)
    sf_trig = multiply_all_trig(effs_trig_data)/multiply_all_trig(effs_trig_mc)
    
    sf_id_up = multiply_all(sfs_id_up)
    sf_id_down = multiply_all(sfs_id_down)
    sf_iso_up = multiply_all(sfs_iso_up)
    sf_iso_down = multiply_all(sfs_iso_down)
    sf_trig_up = multiply_all_trig(effs_trig_data_up)/multiply_all_trig(effs_trig_mc_up)
    sf_trig_down = multiply_all_trig(effs_trig_data_down)/multiply_all_trig(effs_trig_mc_down)

    ret = {
        "mu1_id" : sfs_id[0],
        "mu1_iso" : sfs_iso[0],
        "mu2_id" : sfs_id[1],
        "mu2_iso" : sfs_iso[1],
        "id": sf_id,
        "iso": sf_iso,
        "trigger": sf_trig,
        "id__up": sf_id_up,
        "id__down": sf_id_down,
        "iso__up": sf_iso_up,
        "iso__down": sf_iso_down,
        "trigger__up": sf_trig_up,
        "trigger__down": sf_trig_down
    }

    if use_cuda:
        for k in ret.keys():
            ret[k] = NUMPY_LIB.array(ret[k])
    return ret

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

def assign_category(
    njet, nbjet_medium, nbjet_loose, n_additional_muons, n_additional_electrons,
    dijet_inv_mass, leading_jet, subleading_jet, cat5_dijet_inv_mass_cut, cat5_abs_jj_deta_cut):
    cats = NUMPY_LIB.zeros_like(njet)
    cats[:] = -9999

    msk_prev = NUMPY_LIB.zeros_like(cats, dtype=NUMPY_LIB.bool)

    jj_deta = NUMPY_LIB.abs(leading_jet["eta"] - subleading_jet["eta"])

    #cat 1, ttH
    msk_1 = NUMPY_LIB.logical_or(nbjet_medium > 0, nbjet_loose > 1) & NUMPY_LIB.logical_or(n_additional_muons > 0, n_additional_electrons > 0)
    cats[NUMPY_LIB.invert(msk_prev) & msk_1] = 1
    msk_prev = NUMPY_LIB.logical_or(msk_prev, msk_1)

    #cat 2
    msk_2 = NUMPY_LIB.logical_or(nbjet_medium > 0, nbjet_loose > 1) & (njet > 1)
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
    msk_5 = (dijet_inv_mass > cat5_dijet_inv_mass_cut) & (jj_deta > cat5_abs_jj_deta_cut)
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

"""Given a ROOT file, run any checks that can only be done
on the original file. In our case, we need to access the number
of generated events.
"""
def func_filename_precompute_mc(filename):
    sumw, sumw2 = get_gen_sumweights([filename])
    ret = {"genEventSumw": sumw, "genEventSumw2": sumw2}
    return ret
 
def create_dataset(name, filenames, datastructures, datapath, is_mc):
    ds = Dataset(name, filenames, datastructures, datapath=datapath, treename="Events", is_mc=is_mc)
    return ds

#Branches to load from the ROOT files
def create_datastructure(dataset_name, is_mc, dataset_era, do_fsr=False):
    datastructures = {
        "Muon": [
            ("Muon_pt", "float32"), ("Muon_eta", "float32"),
            ("Muon_phi", "float32"), ("Muon_mass", "float32"),
            ("Muon_pdgId", "int32"), ("Muon_dxybs", "float32"),
            ("Muon_pfRelIso04_all", "float32"), ("Muon_mediumId", "bool"),
            ("Muon_tightId", "bool"), ("Muon_charge", "int32"),
            ("Muon_isGlobal", "bool"), ("Muon_isTracker", "bool"),
            ("Muon_nTrackerLayers", "int32"), ("Muon_ptErr", "float32"),
            ("Muon_pfRelIso03_chg", "float32"), ("Muon_miniPFRelIso_chg", "float32"),
        ],
        "Electron": [
            ("Electron_pt", "float32"), ("Electron_eta", "float32"),
            ("Electron_phi", "float32"), ("Electron_mass", "float32"),
            ("Electron_pfRelIso03_all", "float32"),
            ("Electron_mvaFall17V2Iso_WP90", "bool"),
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
            ("Jet_rawFactor", "float32"),
        ],

     "SoftActivityJet": [
            ("SoftActivityJet_pt", "float32"),
            ("SoftActivityJet_eta", "float32"),
            ("SoftActivityJet_phi", "float32"),
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
            ("SoftActivityJetNjets2", "int32"),
            ("SoftActivityJetHT5", "float32"),
            ("SoftActivityJetHT2", "float32"),
            ("fixedGridRhoFastjetAll", "float32"),
        ],
    }

    if do_fsr:
        datastructures["Muon"] += [
            ("Muon_fsrPhotonIdx", "int32"),
        ]
        datastructures["FsrPhoton"] = [
            ("FsrPhoton_pt", "float32"),
            ("FsrPhoton_eta", "float32"),
            ("FsrPhoton_phi", "float32"),            
            ("FsrPhoton_relIso03", "float32"),
            ("FsrPhoton_dROverEt2", "float32"),
            ("FsrPhoton_muonIdx", "int32")
        ]

    if is_mc:
        datastructures["EventVariables"] += [
            ("Pileup_nTrueInt", "uint32"),
            ("Generator_weight", "float32"),
            ("genWeight", "float32"),
            ]
        if "dy" in dataset_name or "ewk" in dataset_name or "ggh" in dataset_name or "vbf" in dataset_name or "wmh" in dataset_name or "wph" in dataset_name or "zh" in dataset_name or "tth" in dataset_name:
            datastructures["EventVariables"] += [
                ("nLHEPdfWeight", "uint32")
            ]
        if "vbf_powheg_pythia_dipole_125" in dataset_name:
            datastructures["EventVariables"] += [
                ("HTXS_stage1_1_fine_cat_pTjet25GeV", "int32")
            ]
        if dataset_era == "2016" or dataset_era == "2017":
            datastructures["EventVariables"] += [
                ("L1PreFiringWeight_Nom", "float32"),
                ("L1PreFiringWeight_Dn", "float32"),
                ("L1PreFiringWeight_Up", "float32")
            ]
        datastructures["Muon"] += [
            ("Muon_genPartIdx", "int32"),
        ]
        datastructures["Jet"] += [
            ("Jet_genJetIdx", "int32"),
            ("Jet_hadronFlavour","int32"),
            ("Jet_partonFlavour","int32"),
        ]
        datastructures["GenPart"] = [
            ("GenPart_pt", "float32"),
            ("GenPart_eta", "float32"),
            ("GenPart_phi", "float32"),
            ("GenPart_pdgId", "int32"),
            ("GenPart_status", "int32"),
        ]
        if "psweight" in dataset_name:
            datastructures["psweight"] = [
                ("PSWeight", "float32"),
            ]
        if "dy" in dataset_name or "ewk" in dataset_name or "ggh" in dataset_name or "vbf" in dataset_name or "wmh" in dataset_name or "wph" in dataset_name or "zh" in dataset_name or "tth" in dataset_name:
            datastructures["LHEPdfWeight"] = [
                ("LHEPdfWeight", "float32"),
                
            ]
            
        if "dy" in dataset_name or "ewk" in dataset_name:
            
            datastructures["LHEScaleWeight"] = [
                ("LHEScaleWeight", "float32"),
            ]
        datastructures["Jet"] += [
            ("Jet_genJetIdx", "int32"),
            ("Jet_hadronFlavour", "int32")
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
            ("MET_pt", "float32"),
        ]
    elif dataset_era == "2017":
        datastructures["EventVariables"] += [
            ("HLT_IsoMu27", "bool"),
            ("METFixEE2017_pt", "float32"),
        ]
    elif dataset_era == "2018":
        datastructures["EventVariables"] += [
            ("HLT_IsoMu24", "bool"),
            ("MET_pt", "float32"),
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
    def __init__(self, job_descriptions, datapath, do_fsr, nthreads=1, events_per_file=100000):
        self.job_descriptions = job_descriptions
        self.chunk_lock = threading.Lock()
        self.loaded_lock = threading.Lock()
        self.num_chunk = 0
        self.num_loaded = 0
        self.events_per_file = events_per_file
        self.nthreads = nthreads

        self.datapath = datapath
        self.do_fsr = do_fsr

        #For each provided file, find the number of events
        #Split the processing of large files into smaller parts using entrystart and entrysop
        self.nev = []
        self.new_jds = []
        for jd in self.job_descriptions:
            self.nev += [self.get_num_events(jd)]
        for jd, nev in zip(self.job_descriptions, self.nev):
            ichunk = 0
            for evs_chunk in chunks(range(nev), self.events_per_file): 
                new_jd = copy.deepcopy(jd)

                #need a unique identifier for each file_eventchunk
                new_jd["dataset_num_chunk"] = str(new_jd["dataset_num_chunk"]) + "_" + str(ichunk)

                new_jd["entrystart"] = evs_chunk[0]  
                new_jd["entrystop"] = evs_chunk[-1] + 1
                self.new_jds += [new_jd]
                ichunk += 1
        #overwrite the previous job descriptions
        self.job_descriptions = self.new_jds 

    @staticmethod
    def get_num_events(job_desc):
        nev = 0
        for fn in job_desc["filenames"]:
            print("get_num_events", fn)
            nev += len(uproot.open(fn).get("Events"))
        return nev
 
    def is_done(self):
        return (self.num_chunk == len(self)) and (self.num_loaded == len(self))

    #did not make this a generator to simplify handling the thread locks
    def nextone(self):
        self.chunk_lock.acquire()

        if self.num_chunk > 0 and self.num_chunk == len(self):
            self.chunk_lock.release()
            print("Generator is done: num_chunk={0}, len(self.job_descriptions)={1}".format(self.num_chunk, len(self)))
            return None

        job_desc = self.job_descriptions[self.num_chunk]
        print("Loading dataset {0} job desc {1}/{2}, {3}, entrystart={4}, entrystop={5}".format(
            job_desc["dataset_name"], self.num_chunk, len(self.job_descriptions), job_desc["filenames"], job_desc["entrystart"], job_desc["entrystop"]))

        datastructures = create_datastructure(job_desc["dataset_name"], job_desc["is_mc"], job_desc["dataset_era"], self.do_fsr)

        ds = create_dataset(
            job_desc["dataset_name"],
            job_desc["filenames"],
            datastructures,
            self.datapath,
            job_desc["is_mc"])

        ds.random_seed = job_desc["random_seed"]
        ds.era = job_desc["dataset_era"]
        ds.numpy_lib = numpy
        ds.num_chunk = job_desc["dataset_num_chunk"]
        self.num_chunk += 1
        self.chunk_lock.release()

        ds.load_root(nthreads=self.nthreads, entrystart=job_desc["entrystart"], entrystop=job_desc["entrystop"])
        
        #Merge data arrays from multiple files (if specified) to one big array
        ds.merge_inplace()

        # Increment the counter for number of loaded datasets
        with self.loaded_lock:
            self.num_loaded += 1

        return ds

    def __call__(self):
        return self.__iter__()

    def __len__(self):
        return len(self.job_descriptions)

#each job will have a new seed number
def seed_generator(start=0):
    while True:
        yield start
        start += 1

def create_dataset_jobfiles(
    dataset_name, dataset_era,
    filenames, is_mc, chunksize, jobfile_path, seed_generator):

    job_descriptions = []
    ijob = 0
    for files_chunk in chunks(filenames, chunksize):
   
        job_description = {
            "dataset_name": dataset_name,
            "dataset_era": dataset_era,
            "filenames": files_chunk,
            "is_mc": is_mc,
            "dataset_num_chunk": ijob,
            "random_seed": next(seed_generator)
        }

        job_descriptions += [job_description]
        fn = jobfile_path + "/{0}_{1}_{2}.json".format(dataset_name, dataset_era, ijob)
        with open(fn, "w") as fi:
            fi.write(json.dumps(job_description, indent=2))

        ijob += 1
    return job_descriptions
