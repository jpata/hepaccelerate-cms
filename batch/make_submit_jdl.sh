#!/bin/bash
set -e

#number of files to process per job
#For factorized JEC, 5-50 is a good starting point
export NCHUNK=10
export SUBMIT_DIR=`pwd`/..

echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

#Clean old job files, copy from output directory
rm -Rf jobfiles jobfiles.txt jobfiles.tgz
echo "Copying jobfiles"
cp -R ../out/jobfiles ./

#Create archive of job arguments
echo "Creating archive"
tar -cvzf jobfiles.tgz jobfiles
\ls -1 jobfiles/*.json | sed "s/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt

echo "Preparing job chunks"
#Run N different random chunks per job
python chunk_submits.py $NCHUNK jobfiles.txt > jobfiles_merged.txt

rm -f args_analyze.txt
rm -f slurm_submit.sh

#Split on line, not on space
IFS=$'\n'
NJOB=0
for f in `cat jobfiles_merged.txt`; do

    #create condor submit files
    echo $f >> args_analyze.txt 

    #create SLURM submit file
    echo "sbatch slurm_hmm_analyze.sh "$f >> slurm_submit.sh
    NJOB=$((NJOB + 1))
done
echo "Please run 'export SUBMIT_DIR=`pwd`/..'"
echo "To submit the jobs, just run 'condor_submit analyze.jdl'" 

