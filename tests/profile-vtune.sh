#!/bin/bash
sync; echo 1 > /proc/sys/vm/drop_caches
source /opt/intel/vtune_amplifier_2019/amplxe-vars.sh
amplxe-cl -collect advanced-hotspots ./tests/run.sh
