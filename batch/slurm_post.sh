#!/bin/bash

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "hmmplot"   # job name

export PYTHONPATH=coffea:hepaccelerate:.
export WORKDIR=/central/groups/smaria/$USER/hmm/hepaccelerate-cms
export OUTDIR=/central/groups/smaria/$USER/hmm/out
export NTHREADS=16

set -e

cd /central/groups/smaria/$USER/hmm

\ls -1 out_*.tgz | xargs -P $NTHREADS -n 1 tar --skip-old-files -xf

cd $WORKDIR

#Run merge
python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads $NTHREADS \
    --datasets-yaml data/datasets_NanoAODv6_Run2_mixv1.yml \
    --out $OUTDIR

#Run plots
python3 tests/hmm/plotting.py --input $OUTDIR --nthreads $NTHREADS
