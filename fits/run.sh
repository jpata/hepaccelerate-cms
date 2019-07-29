#!/bin/bash

set -e

#dcard="hist__dimuon__inv_mass"
#dcard="hist__dimuon_invmass_70_110_cat5__dijet_inv_mass"
#dcard="hist__dimuon_invmass_70_110_cat5__leading_jet_pt"
#dcard="hist__dimuon_invmass_70_110_cat5__inv_mass"
#dcard="hist__dimuon_invmass_70_110_cat5__numt_jets"

path="../out2/baseline/datacards/2018"
dcard1="hist__dimuon_invmass_z_peak_cat5__inv_mass"
dcard2="hist__dimuon_invmass_z_peak_cat5__num_soft_jets"
dcard3="hist__dimuon_invmass_z_peak_cat5__leadingJet_pt"

cd $path

#Create a combined card from the input cards, ch1 will be the first card, ch2,ch3 etc the other ones
combineCards.py $dcard1.txt $dcard2.txt $dcard3.txt > comb.txt

#the map command has to assign the parameter of interest "r" to all signal processes-> need to be able to match them via regex
#now set just ggh, need to assign vbf too, for this, we need to change the name in the datacard to like ggh_Hmm, vbf_Hmm etc
text2workspace.py -D data -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --channel-masks comb.txt --PO 'map=.*ggh.*:r[1.,-10,10]'

#Run a first fit on the workspace
combine -D data -M MultiDimFit -ntest --saveWorkspace --setParameters mask_ch1=0,mask_ch2=1,mask_ch3=1 comb.root --verbose 9

#Now make the postfit plots, the fit will actually not be rerun I think
combine -D data -M FitDiagnostics -ntest --saveWorkspace --setParameters mask_ch1=0,mask_ch2=1,mask_ch3=1 --saveShapes --saveWithUncertainties higgsCombinetest.MultiDimFit.mH120.root

