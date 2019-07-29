#!/bin/bash

set -e
export PYTHONPATH=hepaccelerate:coffea:.
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND="tensorflow"

python3 tests/hmm/testmatrix.py
