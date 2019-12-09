#!/bin/bash

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "hmmplot"   # job name

export PYTHONPATH=coffea:hepaccelerate:.
export WORKDIR=/central/groups/smaria/jpata/hmm/hepaccelerate-cms
export OUTDIR=/central/groups/smaria/jpata/hmm/out

set -e

cd $WORKDIR

#Run merge
python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads 16 \
    --out $OUTDIR

#Run plots
python3 tests/hmm/plotting.py --input $OUTDIR --nthreads 16
