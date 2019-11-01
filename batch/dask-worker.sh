#!/bin/bash
dask-worker --nthreads 1 --nprocs 1 --memory-limit 0 --no-nanny --death-timeout 120 --no-dashboard tcp://10.3.18.196:8786
