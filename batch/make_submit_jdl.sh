#!/bin/bash

export SUBMIT_DIR=`pwd`/..
echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

#Clean old job files, copy from output directory
rm -Rf jobfiles jobfiles.txt jobfiles.tgz
cp -R ../out/jobfiles ./

#Create archive of job arguments
tar -cvzf jobfiles.tgz jobfiles
\ls -1 jobfiles/*.json | sed "s/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt

#Run 50 different random chunks per job
python chunk_submits.py 20 > jobfiles_merged.txt

#Prepare submit script
cat analyze.jdl > submit.jdl

#Split on line, not on space
IFS=$'\n'
for f in `cat jobfiles_merged.txt`; do
    echo "Arguments = "$f >> submit.jdl
    echo "Queue" >> submit.jdl
    echo >> submit.jdl
done

