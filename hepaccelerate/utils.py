import sys
import os
import time
import json
import numpy as np
from collections import OrderedDict

import uproot

def choose_backend(use_cuda=False):
    if use_cuda:
        print("Using the GPU CUDA backend")
        import cupy
        NUMPY_LIB = cupy
        import hepaccelerate.backend_cuda as ha
        NUMPY_LIB.searchsorted = ha.searchsorted
    else:
        print("Using the numpy CPU backend")
        import numpy as numpy
        NUMPY_LIB = numpy
        import hepaccelerate.backend_cpu as ha
        NUMPY_LIB.asnumpy = numpy.array
    return NUMPY_LIB, ha

class Histogram:
    def __init__(self, contents, contents_w2, edges):
        self.contents = np.array(contents)
        self.contents_w2 = np.array(contents_w2)
        self.edges = np.array(edges)
    
    def __add__(self, other):
        assert(np.all(self.edges == other.edges))
        return Histogram(self.contents +  other.contents, self.contents_w2 +  other.contents_w2, self.edges)

class JaggedStruct(object):
    def __init__(self, offsets, attrs_data, numpy_lib):
        self.numpy_lib = numpy_lib
        
        self.offsets = offsets
        self.attrs_data = attrs_data
        
        num_items = None
        for (k, v) in self.attrs_data.items():
            num_items_next = len(v)
            if num_items and num_items != num_items_next:
                raise AttributeError("Mismatched attribute {0}".format(k))
            else:
                num_items = num_items_next
        self.num_items = num_items
    
        self.masks = {}
        self.masks["all"] = self.make_mask()
    
    def make_mask(self):
        return self.numpy_lib.ones(self.num_items, dtype=self.numpy_lib.bool)
    
    def mask(self, name):
        if not name in self.masks.keys():
            self.masks[name] = self.make_mask()
        return self.masks[name]
    
    def memsize(self):
        size_tot = self.offsets.size
        for k, v in self.attrs_data.items():
            size_tot += v.size
        return size_tot
    
    def numevents(self):
        return len(self.offsets) - 1

    def numobjects(self):
        for k, v in self.attrs_data.items():
            return len(self.attrs_data[k])
    
    @staticmethod
    def from_arraydict(arraydict, prefix, numpy_lib):
        ks = [k for k in arraydict.keys() if prefix in str(k, 'ascii')]
        k0 = ks[0]
        return JaggedStruct(
            numpy_lib.array(arraydict[k0].offsets),
            {str(k, 'ascii').replace(prefix, ""): numpy_lib.array(v.content)
             for (k,v) in arraydict.items()},
            numpy_lib=numpy_lib
        )

    def savez(self, path):
        with open(path, "wb") as of:
            self.numpy_lib.savez(of, offsets=self.offsets, **self.attrs_data)
    
    @staticmethod 
    def load(path, numpy_lib):
        with open(path, "rb") as of:
            fi = numpy_lib.load(of)
   
            #workaround for cupy
            npz_file = fi
            if hasattr(fi, "npz_file"):
                npz_file = fi.npz_file
 
            ks = [f for f in npz_file.files if f!="offsets"]
            return JaggedStruct(
                numpy_lib.array(fi["offsets"]),
                {k: numpy_lib.array(npz_file[k]) for k in ks},
                numpy_lib=numpy_lib
            )

    def move_to_device(self, numpy_lib):
        self.numpy_lib = numpy_lib
        new_offsets = self.numpy_lib.array(self.offsets)
        new_attrs_data = {k: self.numpy_lib.array(v) for k, v in self.attrs_data.items()}
        self.offsets = new_offsets
        self.attrs_data = new_attrs_data
 
    def __getattr__(self, attr):
        if attr in self.attrs_data.keys():
            return self.attrs_data[attr]
        return self.__getattribute__(attr)
 
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Histogram):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

class Results(dict):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __add__(self, other):
        d0 = self
        d1 = other
        
        d_ret = Results({})
        k0 = set(d0.keys())
        k1 = set(d1.keys())

        for k in k0.intersection(k1):
            d_ret[k] = d0[k] + d1[k]

        for k in k0.difference(k1):
            d_ret[k] = d0[k]

        for k in k1.difference(k0):
            d_ret[k] = d1[k]

        return d_ret
    
    def save_json(self, outfn):
        with open(outfn, "w") as fi:
            fi.write(json.dumps(dict(self), indent=2, cls=NumpyEncoder))

def progress(count, total, status=''):
    sys.stdout.write('.')
    sys.stdout.flush()


class Dataset(object):
    def __init__(self, filenames, arrays_to_load, treename):
        self.filenames = filenames
        self.arrays_to_load = arrays_to_load
        self.data_host = []
        self.treename = treename
        self.do_progress = False

    def preload(self, nthreads=1, verbose=False):
        if verbose:
            print("Loading data from {0} ROOT files to memory".format(len(self.filenames)))
        t0 = time.time()
        for ifn, fn in enumerate(self.filenames):
            fi = uproot.open(fn)
            tt = fi.get(self.treename)
            if nthreads > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=nthreads) as executor:
                    arrs = tt.arrays(self.arrays_to_load, executor=executor)
            else:
                arrs = tt.arrays(self.arrays_to_load)
            self.data_host += [arrs]
            if self.do_progress:
                progress(ifn, len(self.filenames))
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("Loaded {0:.2E} events in {1:.1f} seconds, {2:.2E} Hz".format(len(self), dt, len(self)/dt))

    def num_events_raw(self):
        nev = 0
        for arrs in self.data_host:
            k0 = list(arrs.keys())[0]
            nev += len(arrs[k0])
        return nev

    def __len__(self):
        return self.num_events_raw()

class NanoAODDataset(Dataset):
    numpy_lib = np
    def __init__(self, filenames, arrays_to_load, treename, names_structs, names_eventvars):
        super(NanoAODDataset, self).__init__(filenames, arrays_to_load, treename)
        self.names_structs = names_structs
        self.names_eventvars = names_eventvars
        self.cache_prefix = ""
         
        #lists of data, one per file
        self.structs = {}
        for struct in self.names_structs:
            self.structs[struct] = []
        self.eventvars = []

    def move_to_device(self, numpy_lib):
        for ifile in range(len(self.filenames)):
            for structname in self.names_structs:
                self.structs[structname][ifile].move_to_device(numpy_lib)
            for evvar in self.names_eventvars:
                self.eventvars[ifile][evvar] = numpy_lib.array(self.eventvars[ifile][evvar])

    def memsize(self):
        tot = 0
        for ifile in range(len(self.filenames)):
            for structname in self.names_structs:
                tot += self.structs[structname][ifile].memsize()
            for evvar in self.names_eventvars:
                tot += self.eventvars[ifile][evvar].size
        return tot
 
    def __repr__(self):
        s = "NanoAODDataset(files={0}, events={1}, {2})".format(len(self.filenames), len(self), ", ".join(self.structs.keys()))
        return s
    def get_cache_dir(self, fn):
        return os.path.join(self.cache_prefix, fn)

    def printout(self):
        s = str(self) 
        for structname in self.structs.keys():
            s += "\n"
            s += "  {0}({1}, {2})".format(structname, self.num_objects_loaded(structname), ", ".join(self.structs[structname][0].attrs_data.keys()))
        s += "\n"
        s += "  EventVariables({0}, {1})".format(len(self), ", ".join(self.names_eventvars))
        return s    

    def preload(self, nthreads=1, verbose=False):
        super(NanoAODDataset, self).preload(nthreads, verbose)
 
    def build_structs(self, prefix): 
        struct_array = [
            JaggedStruct.from_arraydict(
                {k: v for k, v in arrs.items() if prefix in str(k)},
                prefix, self.numpy_lib 
            ) for arrs in self.data_host
        ]
        return struct_array

    def make_objects(self):
        if self.do_progress:
            print("Making objects with backend={0}".format(self.numpy_lib.__name__))
        t0 = time.time()
        for structname in self.names_structs:
            self.structs[structname] = self.build_structs(structname + "_")

        self.eventvars = [{
            k: self.numpy_lib.array(data[bytes(k, encoding='ascii')]) for k in self.names_eventvars
        } for data in self.data_host]
  
        t1 = time.time()
        dt = t1 - t0
        if self.do_progress:
            print("Made objects in {0:.2E} events in {1:.1f} seconds, {2:.2E} Hz".format(len(self), dt, len(self)/dt))

    def analyze(self, analyze_data, verbose=False, **kwargs):
        t0 = time.time()
        rets = []
        for ifile in range(len(self.filenames)):
            data = {}
            for structname in self.names_structs:
                data[structname] = self.structs[structname][ifile]
            data["num_events"] = self.structs[structname][ifile].numevents()
            data["eventvars"] = self.eventvars[ifile]
            ret = analyze_data(data, **kwargs)
            rets += [ret]
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("analyze: processed analysis with {0:.2E} events in {1:.1f} seconds, {2:.2E} Hz".format(len(self), dt, len(self)/dt))
        return sum(rets, Results({}))

    def to_cache(self, nthreads=1, verbose=False):
        if self.do_progress:
            print("Caching dataset")
        t0 = time.time()
        if nthreads == 1:
            for ifn in range(len(self.filenames)):
                self.to_cache_worker(ifn)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                results = executor.map(self.to_cache_worker, range(len(self.filenames)))
            results = [r for r in results]

        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("to_cache: created cache for {1:.2E} events in {0:.1f} seconds, speed {2:.2E} Hz".format(
                dt, len(self), len(self)/dt
            ))
    
    def to_cache_worker(self, ifn):
        if self.do_progress:
            progress(ifn, len(self.filenames))
        fn = self.filenames[ifn] 
        bfn = os.path.basename(fn).replace(".root", "")
        dn = os.path.dirname(self.get_cache_dir(fn))

        #maybe directory was already created by another worker
        try:
            os.makedirs(dn)
        except FileExistsError as e:
            pass

        for structname in self.names_structs:
            self.structs[structname][ifn].savez(os.path.join(dn, bfn + ".{0}.npz".format(structname)))
        with open(os.path.join(dn, bfn + ".eventvars.npz"), "wb") as fi:
            self.numpy_lib.savez(fi, **self.eventvars[ifn])

    def from_cache(self, nthreads=1, verbose=False):
        t0 = time.time()

        if nthreads == 1:
            for ifn in range(len(self.filenames)):
                ifn, loaded_structs, eventvars = self.from_cache_worker(ifn)
                for structname in self.names_structs:
                    self.structs[structname] += [loaded_structs[structname]]
                self.eventvars += [eventvars]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                results = executor.map(self.from_cache_worker, range(len(self.filenames)))
            results = list(sorted(results, key=lambda x: x[0]))
            for structname in self.names_structs:
                self.structs[structname] = [r[1][structname] for r in results]
            self.eventvars = [r[2] for r in results] 

        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("from_cache: loaded cache for {1:.2E} events in {0:.1f} seconds, speed {2:.2E} Hz".format(
                dt, len(self), len(self)/dt
            ))

    def from_cache_worker(self, ifn):
        if self.do_progress:
            progress(ifn, len(self.filenames))
        fn = self.filenames[ifn]
        bfn = os.path.basename(fn).replace(".root", "")
        
        dn = os.path.dirname(self.get_cache_dir(fn))

        loaded_structs = {}
        for struct in self.names_structs:
            loaded_structs[struct]= JaggedStruct.load(os.path.join(dn, bfn+".{0}.npz".format(struct)), self.numpy_lib)
        with open(os.path.join(dn, bfn+".eventvars.npz"), "rb") as fi:
            npz_file = self.numpy_lib.load(fi)
            if hasattr(npz_file, "npz_file"):
                npz_file = npz_file.npz_file
            eventvars = {k: self.numpy_lib.array(npz_file[k]) for k in npz_file.files}
        return ifn, loaded_structs, eventvars
 
    def num_objects_loaded(self, structname):
        n_objects = 0
        for ifn in range(len(self.structs[structname])):
            n_objects += self.structs[structname][ifn].numobjects()
        return n_objects
    
    def num_events_loaded(self, structname):
        n_events = 0
        for ifn in range(len(self.structs[structname])):
            n_events += self.structs[structname][ifn].numevents()
        return n_events

    def __len__(self):
        n_events_raw = self.num_events_raw()
        n_events_loaded = {k: self.num_events_loaded(k) for k in self.names_structs}
        kfirst = self.names_structs[0]
        if n_events_raw == 0:
            return n_events_loaded[kfirst]
        else:
            return n_events_raw
