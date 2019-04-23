#!/bin/bash
sync; echo 1 > /proc/sys/vm/drop_caches
PYTHONPATH=./fnal-column-analysis-tools:./ python3 `pwd`/tests/analysis_hmumu.py $@
