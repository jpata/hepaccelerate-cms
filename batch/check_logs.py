import sys, glob
import numpy as np

def parse_log(fn):
    lines = open(fn).readlines()
    spd = None
    maxrss = None
    time = None
    for line in lines:
        if "maxrss" in line:
            maxrss = float(line.split()[0].split("=")[1])
        elif "run_analysis" in line:
            spd = float(line.split()[15])
            time = float(line.split()[13])
    return time, maxrss, spd

if __name__ == "__main__":
    logpattern = sys.argv[1]
    maxrss = []
    speeds = []
    times = []
    for fn in glob.glob(logpattern):
        time, m_rss, spd = parse_log(fn)
        if m_rss is None or spd is None:
            print("Could not parse log {0}".format(fn))
            continue 
        times += [time]
        maxrss += [m_rss]
        speeds += [spd]

    print("number of jobs: {0}".format(len(times))) 
    print("runtime: {0:.2f} +- {1:.2f} s".format(np.mean(times), np.std(times)))
    print("maxrss: {0:.2f} +- {1:.2f} MB".format(np.mean(maxrss), np.std(maxrss)))
    print("speeds: {0:.2f} +- {1:.2f} Hz".format(np.mean(speeds), np.std(speeds)))
