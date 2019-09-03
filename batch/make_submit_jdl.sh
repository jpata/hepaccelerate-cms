#!/bin/bash
set -e

#number of files to process per job
#For factorized JEC, 5 is a good starting point
export NCHUNK=5
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
python chunk_submits.py $NCHUNK > jobfiles_merged.txt

#Prepare submit script
cat analyze.jdl > submit.jdl

#Split on line, not on space
IFS=$'\n'
for f in `cat jobfiles_merged.txt`; do

    #create condor submit files
    echo "Arguments = "$f >> submit.jdl
    echo "Queue" >> submit.jdl
    echo >> submit.jdl

    #create SLURM submit file
    echo "sbatch slurm_hmm_analyze.sh "$f >> slurm_submit.sh 
done
echo "Please run 'export SUBMIT_DIR=`pwd`/..'"

