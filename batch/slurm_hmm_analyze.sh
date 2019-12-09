#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "hmm"   # job name

export NTHREADS=2

export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export NUMBA_THREADING_LAYER=workqueue
export WORKDIR=/central/groups/smaria/$USER/hmm/hepaccelerate-cms
export CACHEPATH=/central/groups/smaria/jpata/hmm/skim_merged
export JOB_TMPDIR=$TMPDIR/$SLURM_JOB_ID
export OUTDIR=out

OUTFILE=$1
INFILE=$2

mkdir $JOB_TMPDIR
cd $JOB_TMPDIR

cp -R $SUBMIT_DIR/batch/jobfiles ./

mkdir $OUTDIR
mv jobfiles/datasets.json $OUTDIR/
mv jobfiles $OUTDIR/

python3 $SUBMIT_DIR/batch/addprefix.py $JOB_TMPDIR/$OUTDIR/ < $OUTDIR/$INFILE > $OUTDIR/$INFILE.tmp
mv $OUTDIR/$INFILE.tmp $OUTDIR/$INFILE

cd $SUBMIT_DIR

PYTHONPATH=hepaccelerate:coffea:. python3 tests/hmm/analysis_hmumu.py \
    --action analyze \
    --cachepath $CACHEPATH \
    --nthreads $NTHREADS \
    --datasets-yaml data/datasets_NanoAODv5.yml \
    --jobfiles-load $JOB_TMPDIR/$OUTDIR/$INFILE \
    --out $JOB_TMPDIR/out --do-factorized-jec

cd $WORKDIR
tar -cvzf out.tgz $OUTDIR
cp -R out.tgz /central/groups/smaria/$USER/hmm/$OUTFILE

cd $SUBMIT_DIR

rm -Rf $JOB_TMPDIR
