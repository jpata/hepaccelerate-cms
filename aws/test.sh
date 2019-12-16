#!/bin/bash
set -e
set -o xtrace

env
df -h
python --version
aws s3 cp s3://hepaccelerate-hmm-skim-merged/files.txt ./files.txt
cat files.txt

aws s3 cp s3://hepaccelerate-hmm-skim-merged/sandbox.tgz ./
tar xf sandbox.tgz
cd hepaccelerate-cms
git pull
git submodule update

cd tests/hmm
make

cd ../..

aws s3 cp s3://hepaccelerate-hmm-skim-merged/store/mc/RunIISummer16NanoAODv5/DYToLL_0J_13TeV-amcatnloFXFX-pythia8/merged_55.root ./input.root
cat > desc.json <<EOF
{
  "dataset_name": "dy_0j",
  "dataset_era": "2016",
  "filenames": [
    "input.root"
  ],
  "is_mc": true,
  "dataset_num_chunk": 0,
  "random_seed": 1752
}
EOF

mkdir out
aws s3 cp s3://hepaccelerate-hmm-skim-merged/datasets.json ./out/

PYTHONPATH=coffea:hepaccelerate:. python tests/hmm/analysis_hmumu.py --jobfiles desc.json --datasets-yaml ./data/datasets_NanoAODv5.yml

aws s3 cp ./out/partial_results/dy_0j_2016_0.pkl s3://hepaccelerate-hmm-skim-merged/out/dy_0j_2016_0.pkl
