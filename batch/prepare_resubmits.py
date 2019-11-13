import sys

def parse_aborted(line):
    spl = line.split()
    clusterid, jobid, step = spl[1].split(".")
    return int(jobid)

if __name__ == "__main__":
    #Condor logfile
    input_logfile = sys.argv[1]

    #Get the list of input arguments
    input_args = [l.strip() for l in open("args_analyze.txt").readlines()]
    
    #Find the aborted jobs
    for line in open(input_logfile).readlines():
        line = line.strip()
        if "aborted" in line:
            jobid = parse_aborted(line)
            print(input_args[jobid])
