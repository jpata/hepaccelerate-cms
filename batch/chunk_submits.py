import random, sys

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":
    #How many different files to process per job
    PERJOB = int(sys.argv[1])
    
    infile = open(sys.argv[2])
    lines = [l.strip() for l in infile.readlines()]

    #shuffle the files across jobs so they all have about the same runtime
    idx_rand = list(range(len(lines)))
    random.shuffle(idx_rand)
    lines = [lines[idx] for idx in idx_rand]
   
    #put PERJOB on each line
    for line in chunks(lines, PERJOB):
         print(" ".join(line))
    

