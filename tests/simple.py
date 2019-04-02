import os, glob
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import argparse

import uproot
import hepaccelerate
from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#This function will be called for every file in the dataset
def analyze_data(data, NUMPY_LIB=None, parameters={}):
    #Output structure that will be returned and added up among the files.
    #Should be relatively small.
    ret = Results()
    
    muons = data["Muon"]
    #Get the total number of events
    num_events = muons.numevents()
    num_muons = muons.numobjects()
    mask_events = NUMPY_LIB.ones(num_events, dtype=NUMPY_LIB.bool)
    print("Processing {0} events, {1} muons".format(num_events, num_muons))
    
    #Add a small summary of what we analyzed to the return output 
    ret["num_events"] = float(num_events)
    
    #Find all the muons that pass a certain pt cut 
    #Depending on the hepaccelerate backend, this will be done on the CPU or the GPU
    mask_muons_pass_pt = muons.pt > parameters["muons_ptcut"]
   
    #Add up the number of passing muons per event 
    #Depending on the hepaccelerate backend, this will be done on the CPU or the GPU
    mask_events_dimuon = ha.sum_in_offsets(muons, mask_muons_pass_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 2
    ret["events_dimuon"] = float(mask_events_dimuon.sum())
   
    #if we have N events, create an array of length N, where every element is the pt of the leading (0th) muon
    #We only consider events that pass the `mask_events_dimuon` event selection and muons that pass the `mask_muons_pass_pt` selection.
    #If there is no such muon in the event, the array element will be 0
    inds = NUMPY_LIB.zeros(num_events, dtype=NUMPY_LIB.int32)
    leading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, mask_events_dimuon, mask_muons_pass_pt)
    print("leading_muon_pt[all_events] =", leading_muon_pt[:10])
    
    #create a histogram of the leading muon pt, using only the events that passed the muon selection
    weights = NUMPY_LIB.ones(num_events, dtype=NUMPY_LIB.float32)
    bins = NUMPY_LIB.linspace(0,300,101)
    hist_muons_pt = Histogram(*ha.histogram_from_vector(leading_muon_pt[mask_events_dimuon], weights, bins))
    print("leading_muon_pt[mask_events_dimuon] =", leading_muon_pt[mask_events_dimuon][:10])
    ret["hist_leading_muon_pt"] = hist_muons_pt
 
    return ret
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--from-cache', action='store_true', help='Load from cache (otherwise create it)')
    parser.add_argument('--nthreads', action='store', help='Number of CPU threads to use', type=int, default=4, required=False)
    parser.add_argument('--files-per-batch', action='store', help='Number of files to process per batch', type=int, default=1, required=False)
    parser.add_argument('--cache-location', action='store', help='Path prefix for the cache, must be writable', type=str, default=os.path.join(os.getcwd(), 'cache'))
    parser.add_argument('--filelist', action='store', help='List of files to load', type=str, default=None, required=False)
    parser.add_argument('filenames', nargs=argparse.REMAINDER)
    args = parser.parse_args()
 
    NUMPY_LIB, ha = choose_backend(args.use_cuda)
    NanoAODDataset.numpy_lib = NUMPY_LIB
   
    #define arrays to load: these are objects that will be kept together 
    arrays_objects = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId",
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso04_all", "Muon_mediumId", "Muon_charge",
        "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id",
    ]
    #these are variables per event
    arrays_event = [
        "PV_npvsGood",
        "HLT_IsoMu24",
        "run", "luminosityBlock", "event"
    ]

    filenames = None
    if not args.filelist is None:
        filenames = [l.strip() for l in open(args.filelist).readlines()]
    else:
        filenames = args.filenames

    for fn in filenames:
        if not fn.endswith(".root"):
            raise Exception("Must supply ROOT filename, but got {0}".format(fn))

    results = Results()
    for ibatch, files_in_batch in enumerate(chunks(filenames, args.files_per_batch)): 
        #define our dataset
        dataset = NanoAODDataset(files_in_batch, arrays_objects + arrays_event, "Events", ["Jet", "Muon", "TrigObj"], arrays_event)
        dataset.get_cache_dir = lambda fn,loc=args.cache_location: os.path.join(loc, fn)

        if not args.from_cache:
            #Load data from ROOT files
            dataset.preload(nthreads=args.nthreads, verbose=True)

            #prepare the object arrays on the host or device
            dataset.make_objects()

            print("preparing dataset cache")
            #save arrays for future use in cache
            dataset.to_cache(verbose=True, nthreads=args.nthreads)

        #Optionally, load the dataset from an uncompressed format
        else:
            print("loading dataset from cache")
            dataset.from_cache(verbose=True, nthreads=args.nthreads)
        if ibatch == 0:
            print(dataset.printout())

        #Run the analyze_data function on all files
        results += dataset.analyze(analyze_data, parameters={"muons_ptcut": 30.0})
             
    print(results)
    print("Efficiency of dimuon events: {0:.2f}".format(results["events_dimuon"]/results["num_events"]))
    
    #Save the results 
    results.save_json("out.json")
