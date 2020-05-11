#!/bin/bash
set -e

#number of files to process per job
#For factorized JEC, 5-50 is a good starting point
export NCHUNK=10
export SUBMIT_DIR=$(dirname `pwd`) 

echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

if [[ ! -f ../out/datasets.json ]]; then
    echo "../out/datasets.json does not exist, please create it using tests/hmm/run.sh or otherwise"
    exit 1 
fi

#Clean old job files, copy from output directory
rm -Rf jobfiles jobfiles.tgz

echo "Copying jobfiles"
cp -R ../out/jobfiles ./

#Create archive of job arguments
\ls -1 jobfiles/*.json | sort -R > jobfiles/jobfiles.txt

#must be after the creation of job arguments 
cp ../out/datasets.json jobfiles/

#Run N different random chunks per job
echo "Preparing job chunks"
split -l$NCHUNK jobfiles/jobfiles.txt jobfiles/jobfiles_split.txt.

rm -f args_analyze.txt
rm -f slurm_submit.sh

#Split on line, not on space
IFS=$'\n'
NJOB=0
echo "Preparing condor argument file args_analyze.txt"
for f in `\ls -1 jobfiles/jobfiles_split.txt.*`; do

    #create condor submit files
    echo /storage/user/$USER/hmm/out_$NJOB.tgz $f >> args_analyze.txt 
    echo sbatch slurm_hmm_analyze.sh /central/groups/smaria/$USER/hmm/out_$NJOB.tgz $f >> slurm_submit.sh

    NJOB=$((NJOB + 1))
done

echo "Creating jobfile archive"
tar -cvzf jobfiles.tgz jobfiles

NJOBS=`wc -l args_analyze.txt`
echo "Prepared jobs: $NJOBS"
echo "Please run 'export SUBMIT_DIR=$SUBMIT_DIR'"
echo "On T2_US_Caltech, to submit the jobs, just run 'condor_submit analyze.jdl'" 
echo "On Caltech HPC, to submit the jobs, just run 'source slurm_submit.sh'" 
