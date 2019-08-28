#!/bin/bash

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
#Run 50 different random chunks per job
python chunk_submits.py 5 > jobfiles_merged.txt

#Prepare submit script
cat analyze.jdl > submit.jdl

#Split on line, not on space
IFS=$'\n'
for f in `cat jobfiles_merged.txt`; do
    echo "Arguments = "$f >> submit.jdl
    echo "Queue" >> submit.jdl
    echo >> submit.jdl
    echo "sbatch slurm_hmm_analyze.sh "$f >> slurm_submit.sh 
done
echo "Please run 'export SUBMIT_DIR=`pwd`/..'"

