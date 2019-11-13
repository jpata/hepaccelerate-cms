from __future__ import print_function
import sys

def parse_aborted(line):
    spl = line.split()
    clusterid, jobid, step = spl[1].split(".")
    return int(jobid)

def parse_terminate(line, next_line):
    spl = line.split()
    clusterid, jobid, step = spl[1].split(".")
    
    retcode = int(next_line.split()[-1][:-1])
    return jobid, retcode
 
if __name__ == "__main__":
    #Condor logfile
    input_logfile = sys.argv[1]

    #Get the list of input arguments
    input_args = [l.strip() for l in open("args_analyze.txt").readlines()]
   
    num_failed = 0
    successful_jobs = []

    log_lines = open(input_logfile).readlines() 
    #Find the aborted jobs
    for iline, line in enumerate(log_lines):
        line = line.strip()
        if "aborted" in line:
            jobid = parse_aborted(line)
            print(input_args[jobid])
            num_failed += 1
        elif "terminated" in line:
            jobid, retcode = parse_terminate(line, log_lines[iline+1])
            if retcode != 0:
                print(input_args[jobid])
                num_failed += 1
            else:
                successful_jobs.append(jobid)

    print("Successful jobs: {0}/{1}".format(len(successful_jobs), len(input_args)), file=sys.stderr)

    if num_failed > 0:
        print("Warning: {0}/{1} jobs failed".format(num_failed, len(input_args)), file=sys.stderr) 
