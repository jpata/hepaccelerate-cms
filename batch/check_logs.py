import sys, glob
import numpy as np

def parse_log(fn):
    lines = open(fn).readlines()
    spd = None
    maxrss = None
    for line in lines:
        if "maxrss" in line:
            maxrss = float(line.split()[0].split("=")[1])
        elif "run_analysis" in line:
            spd = float(line.split()[15])
    return maxrss, spd

if __name__ == "__main__":
    logdir = sys.argv[1]
    maxrss = []
    speeds = []
    for fn in glob.glob("{0}/example_job.out.*".format(logdir)):
        m_rss, spd = parse_log(fn)
        if m_rss is None or spd is None:
            print("Could not parse log {0}".format(fn))
            continue 
        maxrss += [m_rss]
        speeds += [spd]
    
    print("maxrss = {0:.2f} +- {1:.2f} MB".format(np.mean(maxrss), np.std(maxrss)))
    print("speeds = {0:.2f} +- {1:.2f} Hz".format(np.mean(speeds), np.std(speeds)))
