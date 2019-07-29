#!/bin/bash
time ./tests/hmm/test_gpu.sh &> log_gpu
time ./tests/hmm/test_cpu.sh &> log_cpu
