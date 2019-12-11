from __future__ import print_function
import sys, os

resubmits = open("args_analyze_resubmit.txt", "w")
is_fine = True
num_outputs_found = 0
num_outputs_missing = 0

for argline in open("args_analyze.txt").readlines():
    argline = argline.strip()
    outfile = argline.split()[0]
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
