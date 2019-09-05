#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=3G   # memory per CPU core
#SBATCH -J "hmm"   # job name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load singularity/3.2.0

export NTHREADS=2

export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export SINGULARITY_IMAGE=/central/groups/smaria/jpata/software/cupy2.simg
export WORKDIR=/central/groups/smaria/$USER/hmm/hepaccelerate-cms
export JOB_TMPDIR=$TMPDIR/$SLURM_JOB_ID

mkdir $JOB_TMPDIR
cd $JOB_TMPDIR

env

for word in "$@"; do
    echo $WORKDIR/out/jobfiles/$word.json >> args.txt
done

cd $WORKDIR

PYTHONPATH=hepaccelerate:coffea:. singularity exec -B /central $SINGULARITY_IMAGE \
     python3 tests/hmm/analysis_hmumu.py --action analyze \
    --cache-location /central/groups/smaria/jpata/hmm/cache --nthreads $NTHREADS \
    --async-data \
    --jobfiles-load $JOB_TMPDIR/args.txt --out $JOB_TMPDIR/out

cp -R $JOB_TMPDIR/out /central/groups/smaria/$USER/hmm/

rm -Rf $JOB_TMPDIR
