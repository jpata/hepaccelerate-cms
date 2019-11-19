from __future__ import print_function
import sys, os

is_fine = True
fi = open("skim_merge/args_merge.txt")
resubmits = open("skim_merge/args_resubmit.txt", "w")
num_outputs_found = 0
num_outputs_missing = 0

for line in fi:
    line = line.strip()
    output = line.split()[1]
    if not os.path.isfile(output):
        print(line, file=resubmits)
        is_fine = False
        num_outputs_missing += 1
    else:
        num_outputs_found += 1

print("is_fine={0}".format(is_fine),
    "num_outputs_found={0}".format(num_outputs_found),
    "num_outputs_missing={0}".format(num_outputs_missing)
)
