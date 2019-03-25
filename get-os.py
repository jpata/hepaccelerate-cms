import uproot, awkward
import numpy as np

def get_os_muons_awkward(muon_charges, out_muon_mask):

    ch = muon_charges
        
    #select events with at least 2 muons
    events_min2_muons = ch.count()>=2
    
    #get the charges of the muons in these events
    ch2 = ch[events_min2_muons]

    #get the index pairs of all muons on an event-by-event basis
    all_muon_pairs = ch2.argcross(ch2)

    #get only those index pairs where the muon is not paired with itself and is paired with another muon with a higher index
    pairs_mask = (all_muon_pairs['0'] != all_muon_pairs['1']) & ((all_muon_pairs['0'] < all_muon_pairs['1']))
    all_muon_pairs = all_muon_pairs[pairs_mask]
    
    #get the pairs with the opposite sign charges
    pairs_with_os = ch2[all_muon_pairs['0']] != ch2[all_muon_pairs['1']]
    
    #get the indices of the pairs with the opposite sign
    idxs = all_muon_pairs[pairs_with_os]

    #get the events that had at least one such good pair
    events_min1_os_pair = idxs['0'].count()>=1
    idxs2 = idxs[events_min1_os_pair]
    bestpair = idxs2[:, 0]
    
    first_muon_idx = bestpair['0']
    second_muon_idx = bestpair['1']

    #set the leading and subleading muons to pass the mask according to the pair
    muon_mask_active = out_muon_mask[events_min2_muons][events_min1_os_pair]
    muon_mask_active.content[muon_mask_active.starts + first_muon_idx] = True
    muon_mask_active.content[muon_mask_active.starts + second_muon_idx] = True
    
    return

infile = '/nvmedata/store/mc/RunIIFall17NanoAOD/GluGluHToMuMu_M125_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/20000/0C2B3A66-B042-E811-8C6D-44A8423DE2C0.root'

fi = uproot.open(infile)
tt = fi.get("Events")
arr = tt.array("Jet_pt")

muon_charges = tt.array("Muon_charge")

mask_awkward = awkward.array.jagged.JaggedArray(
    muon_charges.starts,
    muon_charges.stops,
    np.zeros(shape=muon_charges.content.shape, dtype=bool)
)

for i in range(100):
    get_os_muons_awkward(muon_charges, mask_awkward)
