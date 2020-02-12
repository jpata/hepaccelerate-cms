#!/usr/bin/env python
# Analyzes the job logs for the hmm_analyze.sh step 
import sys, glob
import numpy as np

def parse_log(fn):
    lines = open(fn).readlines()
    spd = None
    maxrss = None
    time = None
    evs = None
    size_gb = None
    for line in lines:
        if "maxrss" in line:
            maxrss = float(line.split()[0].split("=")[1])
        elif "run_analysis" in line:
            spd = float(line.split()[15])
            time = float(line.split()[13])
            size_gb = float(line.split()[11])
            evs = float(line.split()[3])
    return time, maxrss, spd, evs, size_gb

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('./check_logs.py "logs/example_job.out.JOB_ID.*"')

    logpattern = sys.argv[1]
    maxrss = []
    speeds = []
    times = []
    num_events = 0
    size_gb = 0

    for fn in glob.glob(logpattern):
        time, m_rss, spd, evs, _size_gb = parse_log(fn)
        if m_rss is None or spd is None:
            print("Could not parse log {0}".format(fn))
            continue 
        times += [time]
        maxrss += [m_rss]
        speeds += [spd]
        num_events += evs
        size_gb += _size_gb

    print("number of jobs: {0}".format(len(times))) 
    print("runtime: {0:.2f} +- {1:.2f} s".format(np.mean(times), np.std(times)))
    print("maxrss: {0:.2f} +- {1:.2f} MB".format(np.mean(maxrss), np.std(maxrss)))
    print("speeds: {0:.2f} +- {1:.2f} Hz".format(np.mean(speeds), np.std(speeds)))
    print("total processed: {} events, {:.2f} GB".format(int(num_events), size_gb))
