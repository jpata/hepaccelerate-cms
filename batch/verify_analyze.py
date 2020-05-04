from __future__ import print_function
import sys, os

#argument file
infile = sys.argv[1]

resubmits = open(infile + ".resubmit", "w")
is_fine = True
num_outputs_found = 0
num_outputs_missing = 0

for argline in open(infile).readlines():
    argline = argline.strip()
    #output file is the second-to-last argument on the line
    outfile = argline.split()[-2]
    if not os.path.isfile(outfile):
        print(argline, file=resubmits)
        is_fine = False
        num_outputs_missing += 1
    else:
        num_outputs_found += 1
print("is_fine={0}".format(is_fine),
    "num_outputs_found={0}".format(num_outputs_found),
    "num_outputs_missing={0}".format(num_outputs_missing)
)
