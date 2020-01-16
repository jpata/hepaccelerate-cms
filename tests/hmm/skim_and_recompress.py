from __future__ import print_function
import ROOT, sys
import subprocess
import glob
import os
import tempfile
import argparse

branch_status_flag = {"keep": True, "drop": False}
 
#Given one NanoAOD file, open the Events tree, select events that pass the skim_cut, drop unneeded branches
#and recompress with LZ4 (~30% larger, but much faster to load)
def skim_recompress_one_file(outfile, infile, skim_cut, drop_branches):
    print("skim_one_file", infile, outfile)
    tf = ROOT.TFile(infile)
    of = ROOT.TFile(outfile, "RECREATE")

    #change the output to use LZ4 for fast decompression
    of.SetCompressionAlgorithm(4)
    of.SetCompressionLevel(9)

    tt = tf.Get("Events")
    for cmd, bname in keep_drop_commands:
        tt.SetBranchStatus(bname, branch_status_flag[cmd])

    t1 = tt.CopyTree(skim_cut)
    nev = t1.GetEntries()
    print(nev)
    t3 = tf.Get("LuminosityBlocks").CloneTree()
    t4 = tf.Get("Runs").CloneTree()

    of.Write()
    of.Close()
    return nev

def get_file_entries(fn):
    nev = 0
    tf = ROOT.TFile.Open(fn)
    tt = tf.Get("Events")
    nev = tt.GetEntries()
    return nev

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

    #Check the input TTree entries 
    nev = sum([get_file_entries(inf) for inf in infiles])

    #Run the actual merging 
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
  
    #Check the output TTree entries 
    nev2 = get_file_entries(outfile) 
    if nev != nev2:
        raise Exception("Merged TTree did not have the same number of events as the input TTrees: {}!={}. ".format(nev, nev2) + 
            "Please check the output of 'hadd' for warnings!")  

    return outfile

def skim_and_merge(args):
    outfile, infiles, skim_str, temppath, keep_drop_commands = args

    if os.path.isfile(outfile):
        print("skipping {0}".format(outfile))
        return

    tmp_files = []
    nev = 0
    for infile in infiles:
        _, new_fn = tempfile.mkstemp(suffix=".root", dir=temppath)
        nev += skim_recompress_one_file(new_fn, infile, skim_str, keep_drop_commands)
        tmp_files += [new_fn]
    hadd(outfile, tmp_files)
    
    #Delete intermediate files
    for fn in tmp_files:
        os.remove(fn)

def get_branches(fn, treename):
    tf = ROOT.TFile(fn)
    tt = tf.Get(treename)
    brs = [b.GetName() for b in tt.GetListOfBranches()]
    return set(brs)

def get_common_branches(filenames):
    sets = []
    for infile in infiles:
        brs = get_branches(infile, "Events")
        sets.append(brs)
    
    common_branches = set.intersection(*sets)
    return sorted(list(common_branches))

def parse_args():
    parser = argparse.ArgumentParser(description='Skim, slim, merge and recompress NANOAOD files')
    parser.add_argument('--infiles', '-i', action='store', help='filename with list of input files', required=True)
    parser.add_argument('--out', '-o', action='store', help='output file', required=True)
    parser.add_argument('--skim', '-s', action='store', help='skim cut', default="1")
    parser.add_argument('--temppath', '-t', action='store', help='temporary file path', required=True)
    parser.add_argument('--branches', '-b', action='store', help='Filename with keep/drop commands for branches')
    args = parser.parse_args()
    return args

def parse_keep_drop(fn):
    fi = open(fn)
    cmds = []
    for line in fi.readlines():
        cmd = line.strip().split()
        if not (len(cmd) == 2 and (cmd[0] in ["keep", "drop"])):
            raise Exception("Invalid keep/drop command: {}".format(line))
        cmds += [cmd]
    return cmds
 
if __name__ == "__main__":
    print(sys.argv)
    args = parse_args()
    infiles = list(map(lambda x: x.strip(), open(args.infiles).readlines()))
    if len(infiles) == 0:
        raise Exception("No input files specified, please check {}".format(args.infiles))

    keep_drop_commands = []

    #Find the common branches among the files
    common_branches = get_common_branches(infiles)
    keep_drop_commands = [("drop", "*")] + [("keep", b) for b in common_branches]

    #Additional keep/drop commands
    if args.branches:
        keep_drop_commands += parse_keep_drop(args.branches)

    for cmd in keep_drop_commands:
        print(cmd)
    skim_and_merge((args.out, infiles, args.skim, args.temppath, keep_drop_commands))
