#!/bin/bash
sync; echo 1 > /proc/sys/vm/drop_caches
source /opt/intel/vtune_amplifier_2019/amplxe-vars.sh
amplxe-cl -help collect hotspots
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -knob sampling-mode=sw -run-pass-thru=-timestamp=sys ./tests/run.sh $@
