#!/bin/bash
env

cd $TMP

cp $SUBMIT_DIR/batch/skim_merge.tgz ./

tar xf skim_merge.tgz
mv skim_merge/*.txt ./

ls -al

#Run the code
python2 $SUBMIT_DIR/tests/hmm/skim_and_recompress.py -i $1 -o $2 -t ./ -s "$3"

#This didn't use to be necessary??
rm -Rf *.txt *.tgz

echo "job done"
