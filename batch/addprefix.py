import sys
pref = sys.argv[1]

for line in sys.stdin.readlines():
    print(pref + line.strip())
