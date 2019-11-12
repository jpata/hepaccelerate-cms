from __future__ import print_function
import ROOT, sys
import subprocess
import glob
import os
import tempfile
import argparse

#Given one NanoAOD file, open the Events tree, select events that pass the skim_cut, drop unneeded branches
#and recompress with LZ4 (~30% larger, but much faster to load)
def skim_recompress_one_file(outfile, infile, skim_cut):
    print("skim_one_file", infile, outfile)
    tf = ROOT.TFile(infile)

    of = ROOT.TFile(outfile, "RECREATE")

    #use LZ4 for fast decompression
    of.SetCompressionAlgorithm(4)
    of.SetCompressionLevel(9)

    if skim_cut == "1":
        t1 = tf.Get("Events").CloneTree()
    else:
        tt = tf.Get("Events")
        tt.SetBranchStatus("FatJet*", False)
        tt.SetBranchStatus("GenVisTau*", False)
        tt.SetBranchStatus("Tau*", False)
        tt.SetBranchStatus("IsoTrack*", False)
        tt.SetBranchStatus("SubJet*", False)
        t1 = tt.CopyTree(skim_cut)
    t3 = tf.Get("LuminosityBlocks").CloneTree()
    t4 = tf.Get("Runs").CloneTree()

    of.Write()
    of.Close()

def hadd(outfile, infiles):
    """ Uses hadd to merge ROOT files.

    Args:
        args (tuple): a 2-element tuple (outfile, infiles), where outfile is a str
            and infiles is a list of strs with the paths to the files

    Returns:
        str: Path to output file

    Raises:
        Exception if call to merge failed.
    """
    print("hadd {0} {1}".format(outfile, " ".join(infiles)))
    d = os.path.dirname(outfile)
    try:
        os.makedirs(d)
    except:
        pass
    cmd = ["hadd", "-ff", outfile] + infiles
    print(" ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise Exception("could not merge {0}".format(outfile))
    return outfile

def skim_and_merge(args):
    outfile, infiles, skim_str, temppath = args

    if os.path.isfile(outfile):
        print("skipping {0}".format(outfile))
        return

    tmp_files = []
    for infile in infiles:
        _, new_fn = tempfile.mkstemp(suffix=".root", dir=temppath)
        skim_recompress_one_file(new_fn, infile, skim_str)
        tmp_files += [new_fn]
    hadd(outfile, tmp_files)
    
    #Delete intermediate files
    for fn in tmp_files:
        os.remove(fn)

def merge_no_recompression(args):
    outfile, infiles, skim_str, temppath = args
    if os.path.isfile(outfile):
        return
    hadd(outfile, infiles)

def parse_args():
    parser = argparse.ArgumentParser(description='Skim, slim, recompress and merge nanoaod files')
    parser.add_argument('--infiles', '-i', action='store', help='filename with list of input files', required=True)
    parser.add_argument('--out', '-o', action='store', help='output file', required=True)
    parser.add_argument('--skim', '-s', action='store', help='skim cut', default="1")
    parser.add_argument('--temppath', '-t', action='store', help='temporary file path', required=True)
    parser.add_argument('--lz4', action='store_true', help='recompress with lz4')
    args = parser.parse_args()
    return args
 
if __name__ == "__main__":
    print(sys.argv)
    args = parse_args()
    infiles = map(lambda x: x.strip(), open(args.infiles).readlines())
    if args.lz4 or args.skim != "1":
        skim_and_merge((args.out, infiles, args.skim, args.temppath))
    else:
        merge_no_recompression((args.out, infiles, args.skim, args.temppath))
