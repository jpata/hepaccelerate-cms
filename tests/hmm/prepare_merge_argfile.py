import sys, glob
import yaml
import argparse
import fnmatch
import os

def match_filenames(path, exclude):
    files = glob.glob(path, recursive=True)
    files = [fn for fn in files if not fnmatch.fnmatch(fn, exclude)]
    return list(files)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunks_filesize(files, maxsize=1e9):
    size = 0
    current_files = []
    for fi in files:
        size += os.path.getsize(fi)
        current_files += [fi]
        if size >= maxsize:
            yield current_files
            size = 0
            current_files = []
    if len(current_files) > 0:
        yield current_files

def parse_args():
    parser = argparse.ArgumentParser(description='Skim and merge NanoAOD files')
    parser.add_argument('--input', '-i', action='store', help='List of input files', required=True)
    parser.add_argument('--datapath', '-d', action='store', help='Data input path', required=True)
    parser.add_argument('--outpath', '-o', action='store', help='Data output path', required=True)
    parser.add_argument('--cmd', '-c', action='store', help='Merge command',
        required=False,
        default="singularity exec -B /storage -B /data ~jpata/gpuservers/singularity/images/cupy2.simg python2 tests/hmm/skim_and_recompress.py")
    args = parser.parse_args()
    return args

shellscript_template = """
"""

if __name__ == "__main__":
    args = parse_args()
    datasets = yaml.load(open(args.input), Loader=yaml.FullLoader)

    argfile = open("args_merge.txt", "w")
    for ds in datasets["datasets"]:
        print(ds) 
        dataset_name = ds["name"] 
        dataset_era = ds["era"] 
        path = ds["files_nano_in"]
        skim_cut = ds["skim_cut"] 
        num_files_to_merge = ds["num_files_to_merge"]
 
        filenames = match_filenames(args.datapath + path, ds["files_merged"])
        nfiles = 0
        for ich, ch in enumerate(chunks_filesize(filenames)):
            nfiles += len(ch)
            infiles_name = "{0}_{1}_{2}.txt".format(dataset_name, dataset_era, ich)
            fi = open(infiles_name, "w")
            for fn in ch:
                print(fn, file=fi)
            
            outfile = args.outpath + ds["files_merged"].replace("*.root", "{0}.root".format(ich))
            #print("{0} -i {1} -o {2} -t {3}".format(args.cmd, infiles_name, outfile, args.temppath), file=argfile)
            print(infiles_name, outfile, skim_cut.replace(" ", ""), file=argfile)
        assert(nfiles == len(filenames))
