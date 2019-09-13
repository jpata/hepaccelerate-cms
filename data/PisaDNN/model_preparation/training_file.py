FOLD=0 #1,2,3,4

import math
import ROOT, numpy, root_numpy
import keras


from keras.layers import Dense, Dropout, Lambda, concatenate
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from random import shuffle

#-rw-r--r-- 1 mandorli cms 2.6M May  8 17:55 main_tmva_tree_EWK_LLJJ_pythia8_2017_mu_QCDScalenom_JESnom_JERnom_PUnom.root                                                                                                                                                       
#-rw-r--r-- 1 mandorli cms 2.7M May  8 17:55 main_tmva_tree_TT_2017_mu_QCDScalenom_JESnom_JERnom_PUnom.root                                                                                                                                                                     
#-rw-r--r-- 1 mandorli cms 105M May  8 17:55 main_tmva_tree_VBF_HToMuMu_2017_mu_QCDScalenom_JESnom_JERnom_PUnom.root                                                                          
#-rw-r--r-- 1 mandorli cms  10M May  9 12:55 main_tmva_tree_DYJetsToLL_M-105To160-madgraphMLM_2017_mu_QCDScalenom_JESnom_JERnom_PUnom.root                                                                                                                                      
#-rw-r--r-- 1 mandorli cms  49M May  9 12:55 main_tmva_tree_DYJetsToLL_M-105To160_VBFFilter-madgraphMLM_2017_mu_QCDScalenom_JESnom_JERnom_PUnom.root

lbranches=["Higgs_m","Higgs_mRelReso","Higgs_mReso","Mqq_log", "Rpt", "qqDeltaEta", "ll_zstar", "NSoft5", "minEtaHQ",
           
           "Higgs_pt","log(Higgs_pt)","Higgs_eta","Mqq","QJet0_pt_touse","QJet1_pt_touse","QJet0_eta","QJet1_eta","QJet0_phi","QJet1_phi","QJet0_qgl","QJet1_qgl",
           
           "DNNClassifier", "event", "genWeight", "genWeight*puWeight*btagWeight*muEffWeight"]


files=[]

path2018="/scratch/mandorli/Hmumu/fileSkimFromNanoAOD/fileSkim2018_tmp/"
#0
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/DY105_2018AMCPYSnapshot.root", path2018+"DYJetsToLL_M-105To160-amcatnloFXFX_nano2018.root" , 41.81 ])
#1
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/DY105VBF_2018AMCPYSnapshot.root", path2018+"DYJetsToLL_M-105To160-amcatnloFXFX_VBFFilter_nano2018.root", 41.81*0.0425242 ])
#2
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/DY105_2018MGPYSnapshot.root", path2018+"DYJetsToLL_M-105To160-madgraphMLM_nano2018.root", 41.25 ])
#3
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/DY105VBF_2018MGPYSnapshot.root", path2018+"DYJetsToLL_M-105To160-madgraphMLM_VBFFilter_nano2018.root" , 41.25*0.0419533  ])
#4
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/EWKZ105_2018MGHERWIGSnapshot.root", path2018+"EWK_LLJJ_MLL_105-160_herwig_nano2018.root",0.0508896  ])
#5
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/EWKZ_2018MGPYSnapshot.root", path2018+"EWK_LLJJ_pythia8_nano2018.root",1.664  ])
#6
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/TT_2018MGPYSnapshot.root", path2018+"TT_madgraph_nano2018.root" , 809 ])
#7
files.append(["/scratch/lgiannini/HmmPisa/nail/PisaHmm/out/vbfHmm_2018POWPYSnapshot.root", path2018+"VBF_HToMuMu_nano2018.root" , 0.0008210722 ])

print files
#quit()


def totevents(fn):
    f=ROOT.TFile.Open(fn)
    run=f.Get("Runs")   
    hw=ROOT.TH1F("hw","", 5,0,5)
    run.Project("hw","1","genEventSumw")
    return hw.GetSumOfWeights()

print "Open"

#ewkzjj
f1=ROOT.TFile(files[4][0])
tree1=f1.Get("Events")
ewkzjj_=root_numpy.tree2array(tree1, branches=lbranches)
ewkzjj_=root_numpy.rec2array(ewkzjj_)
n1=totevents(files[4][1])
print n1

ewkzjj_[:,-1]*=files[4][2]
ewkzjj_[:,-1]/=n1

ewkzjj_=numpy.hstack((ewkzjj_, numpy.ones((ewkzjj_.shape[0], 1))*16))
ewkzjj=ewkzjj_

#ttbar
f2=ROOT.TFile(files[6][0])
tree2=f2.Get("Events")
ttbar_=root_numpy.tree2array(tree2, branches=lbranches)
ttbar_=root_numpy.rec2array(ttbar_)
n2=totevents(files[6][1])
print n2

ttbar_[:,-1]*=files[6][2]
ttbar_[:,-1]/=n2

ttbar_=numpy.hstack((ttbar_, numpy.ones((ttbar_.shape[0], 1))*16))
ttbar=ttbar_

#signal
f2=ROOT.TFile(files[7][0])
tree3=f2.Get("Events")
signal_=root_numpy.tree2array(tree3, branches=lbranches)
signal_=root_numpy.rec2array(signal_)
n3=totevents(files[7][1])
print n3

signal_[:,-1]*=files[7][2]
signal_[:,-1]/=n3

signal_=numpy.hstack((signal_, numpy.ones((signal_.shape[0], 1))*16))
signal=signal_

#drell yan
f2=ROOT.TFile(files[2][0])
tree4=f2.Get("Events")
dy1_=root_numpy.tree2array(tree4, branches=lbranches)
dy1_=root_numpy.rec2array(dy1_)
n4=totevents(files[2][1])
print n4

dy1_[:,-1]*=files[2][2]*1.31
dy1_[:,-1]/=n4

dy1_=numpy.hstack((dy1_, numpy.ones((dy1_.shape[0], 1))*16))
dy1=dy1_

f2=ROOT.TFile(files[3][0])
tree5=f2.Get("Events")
dy2_=root_numpy.tree2array(tree5, branches=lbranches)
dy2_=root_numpy.rec2array(dy2_)
n5=totevents(files[3][1])
print n5

dy2_[:,-1]*=files[3][2]*1.31
dy2_[:,-1]/=n5

dy2_=numpy.hstack((dy2_, numpy.ones((dy2_.shape[0], 1))*16))
dy2=dy2_

DY=data=numpy.concatenate((dy1,dy2))

signal = numpy.hstack((signal, numpy.zeros((signal.shape[0], 1))))
ewkzjj = numpy.hstack((ewkzjj, numpy.ones((ewkzjj.shape[0], 1))))
ttbar = numpy.hstack((ttbar, 2*numpy.ones((ttbar.shape[0], 1))))
DY = numpy.hstack((DY, 3*numpy.ones((DY.shape[0], 1))))
dy2 = numpy.hstack((dy2, 3*numpy.ones((dy2.shape[0], 1))))
dy1 = numpy.hstack((dy1, 3*numpy.ones((dy1.shape[0], 1))))

avg_DYweight=numpy.mean(abs(DY[:,-3]), axis=0)
avg_ewkzjjweight=numpy.mean(abs(ewkzjj[:,-3]), axis=0)
avg_ttbarweight=numpy.mean(abs(ttbar[:,-3]), axis=0)
avg_dy2weight=numpy.mean(abs(dy2[:,-3]), axis=0)
avg_dy1weight=numpy.mean(abs(dy1[:,-3]), axis=0)
avg_sweight=numpy.mean(abs(signal[:,-3]), axis=0)

sum_DYweight=numpy.sum(abs(DY[:,-3]))
sum_ewkzjjweight=numpy.sum(abs(ewkzjj[:,-3]))
sum_ttbarweight=numpy.sum(abs(ttbar[:,-3]))
sum_dy2weight=numpy.sum(abs(dy2[:,-3]))
sum_dy1weight=numpy.sum(abs(dy1[:,-3]))
sum_sweight=numpy.sum(abs(signal[:,-3]))


print "BACKGROUND DY : ", avg_DYweight,      sum_DYweight
print "BACKGROUND e  : ", avg_ewkzjjweight,  sum_ewkzjjweight
print "BACKGROUND t  : ", avg_ttbarweight,   sum_ttbarweight
print "BACKGROUND dy1: ", avg_dy1weight,     sum_dy1weight
print "BACKGROUND dy2: ", avg_dy2weight,     sum_dy2weight
print "SIGNAL        : ", avg_sweight,       sum_sweight


data=numpy.concatenate((DY,signal))
data=numpy.concatenate((data,ttbar))
data=numpy.concatenate((data,ewkzjj))

toBeSampleweight=data[:,-3]*( (data[:, -1]==0)/avg_sweight + (data[:, -1]==3)/avg_DYweight + (data[:, -1]==2)/avg_ttbarweight + (data[:, -1]==1)/avg_ewkzjjweight)
print toBeSampleweight

print "BACKGROUND check: ", numpy.mean(abs(toBeSampleweight[ data[:, -1]==3 ]), axis=0), numpy.sum(abs(toBeSampleweight[ data[:, -1]==3 ]))
print "BACKGROUND check: ", numpy.mean(abs(toBeSampleweight[ data[:, -1]==2 ]), axis=0), numpy.sum(abs(toBeSampleweight[ data[:, -1]==2 ]))
print "BACKGROUND check: ", numpy.mean(abs(toBeSampleweight[ data[:, -1]==1 ]), axis=0), numpy.sum(abs(toBeSampleweight[ data[:, -1]==1 ]))
print "SIGNAL check: ", numpy.mean((toBeSampleweight[ data[:, -1]==0 ]), axis=0), numpy.sum(abs(toBeSampleweight[ data[:, -1]==0 ]))

print "signal dy factor:", numpy.sum(abs(toBeSampleweight[ data[:, -1]==3 ]))/numpy.sum(abs(toBeSampleweight[ data[:, -1]==0 ]))
print "signal ewk factor:", numpy.sum(abs(toBeSampleweight[ data[:, -1]==1 ]))/numpy.sum(abs(toBeSampleweight[ data[:, -1]==0 ]))

avg_bweight=numpy.mean(abs(data[ data[:, -1]>0 ][:,-3]), axis=0)
sum_bweight=numpy.mean(abs(data[ data[:, -1]>0 ][:,-3]), axis=0)

print "BGGG        : ", avg_bweight,       sum_bweight

tobBW=data[:,-3]*(1./avg_bweight)
print "BACKGROUND check: ", numpy.mean(abs(tobBW[ data[:, -1]>0 ]), axis=0), numpy.sum(abs(tobBW[ data[:, -1]>0 ]))

print "signal bg factor:", numpy.sum(abs(toBeSampleweight[ data[:, -1]>0]))/numpy.sum(abs(toBeSampleweight[ data[:, -1]==0 ]))


train_idx=((data[:, -5]%4)==2)+((data[:, -5]%4)==3)
val_idx=(data[:, -5]%4)==0
#final set ==3

idxVBFonly=(data[:, -1]<=1)
idxSigDY=(data[:, -1]==0)+(data[:, -1]>1)

print len(train_idx), len(val_idx)


print data.shape
print "+++++++++++++++++++++"

def scaler(x, a, b):
    #mask = keras.backend.not_equal(x, 0)
    return ((x - a)/b)#*keras.backend.cast(mask, keras.backend.dtype(x))


print numpy.where(numpy.isnan(data))
print data[numpy.where(numpy.isnan(data))]
print data[numpy.where(numpy.isnan(data))[0]]
data[numpy.where(numpy.isnan(data))]=0.5

mean = numpy.average(data[:,:-6],axis=0,weights=data[:,-3])   
variance = numpy.average((data[:,:-6]-mean)**2,axis=0, weights=data[:,-3])
std=numpy.sqrt(variance)

print mean, variance

print std.shape
print mean.shape

#numpy.random.shuffle(data)

class OutHistory(keras.callbacks.Callback):
    def __init__(self, name="", monitor="val_binary_crossentropy"):
        self.name = name
        self.monitor = monitor
        self.monitor_op = numpy.less
        self.best = numpy.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        
        self.model.save("MODEL_atEpoch"+str(epoch)+self.name+".h5") 
        
        logs = logs or {}
        
        current = logs.get(self.monitor)
        
        if current is None:
            print "1"
        else:
            if 1: #self.monitor_op(current, self.best):
                
                print '\nEpoch '+str(epoch)+' improved from '+str(self.best)+' to '+str(logs.get(self.monitor))+' saving model to '+self.name+' .root and .h5'
                self.best = current
                
                print "prediction in callback"
                
                if self.name=="ymass":
                    pred_train=self.model.predict(data[train_idx][:, [0,1,2]])
                    pred_eval=self.model.predict(data[val_idx][:, [0,1,2]])
                
                elif self.name=="nomass":
                    pred_train=self.model.predict(data[train_idx][:, 3:-6])
                    pred_eval=self.model.predict(data[val_idx][:, 3:-6])
                
                else:
                    pred_train=self.model.predict([data[train_idx][:, 3:-6], data[train_idx][:, [0,1,2]]])
                    pred_eval=self.model.predict([data[val_idx][:, 3:-6], data[val_idx][:, [0,1,2]]])
                
                
                print "OUTFILE"
                print "DNN_trained_epoch"+str(epoch)+self.name+".root", "RECREATE"
                ofile=ROOT.TFile("DNN_trained_epoch"+str(epoch)+self.name+".root", "RECREATE")
                
                print data[train_idx][:, -1:].shape
                print data[train_idx][:, -2:-1].shape
                print data[train_idx][:, -3:-2].shape
                print (data[train_idx][:, -3:-2]*data[train_idx][:, -2:-1]).shape
                print pred_train[:,0:1].shape
                print data[train_idx][:, -4:-3].shape
                print data[train_idx][:, -5:-4].shape
                
                array=numpy.concatenate((data[train_idx][:, -1:],   #classID (signal==1)
                                        data[train_idx][:, -2:-1],  #year  
                                        data[train_idx][:, -3:-2],  #genweight_161718 
                                        data[train_idx][:, -4:-3],  #genweight
                                        pred_train[:,0:1],
                                        data[train_idx][:, -5:-4],  #evt
                                        data[train_idx][:, -6:-5], #BDT old
                                        data[train_idx][:, 0:1]), #mll
                                        axis=1)

                a = numpy.array( list(map(tuple, array)),                        
                            dtype=[('classID', numpy.int32),
                                   ('year', numpy.int32),
                                   ('genweight_161718', numpy.float32),
                                   ('genweight', numpy.float32),
                                   ('BDTG', numpy.float64),
                                   ('evt', numpy.int32),
                                   ('BDToutput', numpy.float64),
                                   ('mll', numpy.float32)
                                   ] )
                
                tree1 = root_numpy.array2tree(a, name="TrainTree")
                
                array=numpy.concatenate((data[val_idx][:, -1:],    #classID (signal==1)
                                        data[val_idx][:, -2:-1],   #year 
                                        data[val_idx][:, -3:-2],   #genweight_161718 
                                        data[val_idx][:, -4:-3],   #genweight
                                        pred_eval[:,0:1],
                                        data[val_idx][:, -5:-4],   #evt
                                        data[val_idx][:, -6:-5],  #BDT old
                                        data[val_idx][:, 0:1]), #mll
                                        axis=1)
                
                a = numpy.array( list(map(tuple, array)),                        
                            dtype=[('classID', numpy.int32),
                                   ('year', numpy.int32),
                                   ('genweight_161718', numpy.float32),
                                   ('genweight', numpy.float32),
                                   ('BDTG', numpy.float64),
                                   ('evt', numpy.int32),
                                   ('BDToutput', numpy.float64),
                                   ('mll', numpy.float32)
                                   ] )
                
                tree2 = root_numpy.array2tree(a, name="TestTree")
                
                ofile.cd()
                tree1.Write()
                tree2.Write()
                ofile.Close()
                
                print "DNN_trained_epoch"+str(epoch)+self.name+".root", "CLOSED"
                
                del tree2, tree1      
        

ohistory = OutHistory()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

dropoutRate=0.2
dim=data.shape[1]-6

numpy.random.seed(7)

#Inputs=Input(shape=(dim,))

Inputs=[]

Inputs+=[Input(shape=(dim-3,), name="nomass") ]
Inputs+=[Input(shape=(3,), name="mass") ]

print "VERY LOUD "
print -mean[3:], std[3:]**(-1)
print "AGAIN"
print -mean[:3], std[:3]**(-1)


#add normalization layers
InputsN = Lambda(scaler, arguments={'a':mean[3:], 'b':std[3:]})(Inputs[0])
InputsM = Lambda(scaler, arguments={'a':mean[:3], 'b':std[:3]})(Inputs[1])

InputsN = (Inputs[0])
InputsM = (Inputs[1])

InputsALL = concatenate( [InputsN , InputsM] )

ewkZjj = Dense(50, kernel_initializer="lecun_uniform", activation="relu", name="ewk_1")(InputsALL)
ewkZjj = Dropout(dropoutRate, name="ewk_d1")(ewkZjj)
ewkZjj = Dense(40, kernel_initializer="lecun_uniform", activation="relu", name="ewk_2")(ewkZjj)
ewkZjj = Dropout(dropoutRate, name="ewk_d2")(ewkZjj)
ewkZjj = Dense(20, kernel_initializer="lecun_uniform", activation="relu", name="ewk_3")(ewkZjj)
ewkZjj = Dropout(dropoutRate, name="ewk_d3")(ewkZjj)
ewz_vs_h = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform', name="ewk_4")(ewkZjj)

DY = Dense(50, kernel_initializer="lecun_uniform", activation="relu", name="dy_1")(InputsALL)
DY = Dropout(dropoutRate, name="dy_d1")(DY)
DY = Dense(40, kernel_initializer="lecun_uniform", activation="relu", name="dy_2")(DY)
DY = Dropout(dropoutRate, name="dy_d2")(DY)
DY = Dense(20, kernel_initializer="lecun_uniform", activation="relu", name="dy_3")(DY)
DY = Dropout(dropoutRate, name="dy_d3")(DY)
dy_vs_h = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform', name="dy_4")(DY)

x = Dense(80, kernel_initializer="lecun_uniform", activation="relu", name="nomass_1")(InputsN)
x = Dropout(dropoutRate, name="nomass_d1")(x)
x = Dense(60, kernel_initializer="lecun_uniform", activation="relu", name="nomass_2")(x)
x = Dropout(dropoutRate, name="nomass_d2")(x)
x = Dense(40, kernel_initializer="lecun_uniform", activation="relu", name="nomass_3")(x)
x = Dropout(dropoutRate, name="nomass_d3")(x)

x2 = Dense(80, kernel_initializer="lecun_uniform", activation="relu", name="ymass_1")(InputsM)
x2 = Dropout(dropoutRate, name="ymass_d1")(x2)
x2 = Dense(60, kernel_initializer="lecun_uniform", activation="relu", name="ymass_2")(x2)
x2 = Dropout(dropoutRate, name="ymass_d2")(x2)
x2 = Dense(40, kernel_initializer="lecun_uniform", activation="relu", name="ymass_3")(x2)
x2 = Dropout(dropoutRate, name="ymass_d3")(x2)

p1 = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform')(x)
p2 = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform')(x2)

x3 = concatenate( [x , x2, ewkZjj, DY] )
x3 = Dense(50, kernel_initializer="lecun_uniform", activation="relu")(x3)
x3 = Dropout(dropoutRate)(x3)
x3 = Dense(30, kernel_initializer="lecun_uniform", activation="relu")(x3)
x3 = Dropout(dropoutRate)(x3)

predictions = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform')(x3)

model_ZvsH= Model(inputs=Inputs, outputs=ewz_vs_h)
model_DYvsH= Model(inputs=Inputs, outputs=dy_vs_h)

model = Model(inputs=Inputs, outputs=predictions)
model1 = Model(inputs=Inputs[0], outputs=p1)
model2 = Model(inputs=Inputs[1], outputs=p2)

val_loss=[]
loss=[]
acc=[]
val_acc=[]
lr=[]

#ADAM=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#model_ZvsH.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy', 'binary_crossentropy'])
#model_ZvsH.summary()

#model_DYvsH.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy', 'binary_crossentropy'])
#model_DYvsH.summary()

#model1.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', 'binary_crossentropy'])
#model1.summary()

#model2.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', 'binary_crossentropy'])
#model2.summary()



#ohistory = OutHistory(name=str("_ZvsH"))

#history=model_ZvsH.fit([data[train_idx*idxVBFonly][:, 3:-6],data[train_idx*idxVBFonly][:, [0,1,2]]],data[train_idx*idxVBFonly][:,-1]==0, 
            #sample_weight=data[train_idx*idxVBFonly][:,-3]*( (data[train_idx*idxVBFonly][:, -1]==0)/avg_sweight + (data[train_idx*idxVBFonly][:, -1]>0)/avg_ewkzjjweight),
            #validation_data=([data[val_idx*idxVBFonly][:, 3:-6],data[val_idx*idxVBFonly][:, [0,1,2]]],data[val_idx*idxVBFonly][:,-1]==0,
            #data[val_idx*idxVBFonly][:,-3]*( (data[val_idx*idxVBFonly][:, -1]==0)/avg_sweight + (data[val_idx*idxVBFonly][:, -1]>0)/avg_ewkzjjweight)),
            #shuffle=True , batch_size=1024,  epochs=50, verbose=1, callbacks=[reduce_lr, ohistory])    


#val_loss+=history.history['val_loss']
#loss+=history.history['loss']
#acc+=history.history['acc']
#val_acc+=history.history['val_acc']
#lr+=history.history['lr']

#ohistory = OutHistory(name=str("_DYvsH"))

#print data[train_idx*idxSigDY][:,-1]

#history=model_DYvsH.fit([data[train_idx*idxSigDY][:, 3:-6],data[train_idx*idxSigDY][:, [0,1,2]]],data[train_idx*idxSigDY][:,-1]==0, 
            #sample_weight=data[train_idx*idxSigDY][:,-3]*( (data[train_idx*idxSigDY][:, -1]==0)/avg_sweight + (data[train_idx*idxSigDY][:, -1]>0)/avg_DYweight),
            #validation_data=([data[val_idx*idxSigDY][:, 3:-6],data[val_idx*idxSigDY][:, [0,1,2]]],data[val_idx*idxSigDY][:,-1]==0,
            #data[val_idx*idxSigDY][:,-3]*( (data[val_idx*idxSigDY][:, -1]==0)/avg_sweight + (data[val_idx*idxSigDY][:, -1]>0)/avg_DYweight)),
            #shuffle=True , batch_size=1024,  epochs=50, verbose=1, callbacks=[reduce_lr, ohistory])
            
            
#val_loss+=history.history['val_loss']
#loss+=history.history['loss']
#acc+=history.history['acc']
#val_acc+=history.history['val_acc']
#lr+=history.history['lr']

    
#ohistory = OutHistory(name=str("nomass"))

#history=model1.fit(data[train_idx][:, 3:-6],data[train_idx][:,-1]==0,
            #sample_weight=data[train_idx][:,-3]*( (data[train_idx][:, -1]==0)*2/avg_sweight + (data[train_idx][:, -1]>0)/avg_bweight),
            #validation_data=(data[val_idx][:, 3:-6],data[val_idx][:,-1]==0,
            #data[val_idx][:,-3]*( (data[val_idx][:, -1]==0)*2/avg_sweight + (data[val_idx][:, -1]>0)/avg_bweight)),
            #shuffle=True , batch_size=1024,  epochs=50, verbose=1, callbacks=[reduce_lr, ohistory])
            
#val_loss+=history.history['val_loss']
#loss+=history.history['loss']
#acc+=history.history['acc']
#val_acc+=history.history['val_acc']
#lr+=history.history['lr']

#ohistory = OutHistory(name=str("ymass"))

#history=model2.fit(data[train_idx][:, [0,1,2]],data[train_idx][:,-1]==0,
            #sample_weight=data[train_idx][:,-3]*( (data[train_idx][:, -1]==0)*2/avg_sweight + (data[train_idx][:, -1]>0)/avg_bweight),
            #validation_data=(data[val_idx][:, [0,1,2]],data[val_idx][:,-1]==0,
            #data[val_idx][:,-3]*( (data[val_idx][:, -1]==0)*2/avg_sweight + (data[val_idx][:, -1]>0)/avg_bweight)),
            #shuffle=True , batch_size=1024,  epochs=5, verbose=1, callbacks=[reduce_lr, ohistory])
            
                
#val_loss+=history.history['val_loss']
#loss+=history.history['loss']
#acc+=history.history['acc']
#val_acc+=history.history['val_acc']
#lr+=history.history['lr']
    
####freeze weightts here
#for layer in model.layers:
    #if "ymass" in layer.name:
        #layer.trainable=True
    #if "nomass" in layer.name:
        #layer.trainable=False    

####freeze weightts here
#for layer in model.layers:
    #if "ewk" in layer.name:
        #layer.trainable=False
    #if "dy_" in layer.name:
        #layer.trainable=False

model.load_weights("../../prova_tutto_ok18_QGL/MODEL_atEpoch199unfreeze__.h5") #desidered epoch!!
        
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', 'binary_crossentropy'])        

model.summary()

model.save_weights("model_toexport_evt3.h5")
import json
model_json = model.to_json()
with open("model_toexport_evt3.json", "w") as json_file:
    json_file.write(model_json)
    
  
#print "VERY LOUD "
#print -mean[3:], std[3:]**(-1)
#print "AGAIN"
#print -mean[:3], std[:3]**(-1)
  
f=open("helphelp.json","w")

f.write('{\n')
f.write('  "input_sequences": [], \n') 
f.write('  "inputs": [\n')
f.write('    {\n')
f.write('      "name": "node_0", \n') 
f.write('      "variables": [\n')

for i in range(len(mean[3:])):
    f.write('        {\n')
    f.write('          "name": "variable_'+str(i)+'", \n') 
    f.write('          "offset": '+str(-mean[i+3])+', \n')
    f.write('          "scale": '+str(std[i+3]**(-1))+'\n')
    if (i+1 < len(mean[3:]) ):
        f.write('        }, \n')
    else:
        f.write('        }\n')

f.write('      ]\n')
f.write('    }, \n') 
f.write('    {\n')
f.write('      "name": "node_1", \n') 
f.write('      "variables": [\n')

for i in range(len(mean[:3])):
    f.write('        {\n')
    f.write('          "name": "variable_'+str(i)+'", \n') 
    f.write('          "offset": '+str(-mean[i])+', \n')
    f.write('          "scale": '+str(std[i]**(-1))+'\n')
    if (i+1 < len(mean[:3]) ):
        f.write('        }, \n')
    else:
        f.write('        }\n')
    
f.write('      ]\n')
f.write('    }\n') 
f.write('  ], \n') 
f.write('  "outputs": [\n')
f.write('    {\n')
f.write('      "labels": [\n')
f.write('        "out_0"\n')
f.write('      ], \n') 
f.write('      "name": "dense_5_0"\n')
f.write('    }\n')
f.write('  ]\n')
f.write('}\n')



quit()

ohistory = OutHistory(name=str("combine"))
            
history=model.fit([data[train_idx][:, 3:-6],data[train_idx][:, [0,1,2]]],data[train_idx][:,-1]==0, 
            sample_weight=data[train_idx][:,-3]*( (data[train_idx][:, -1]==0)*2/avg_sweight + (data[train_idx][:, -1]>0)/avg_bweight),
            validation_data=([data[val_idx][:, 3:-6],data[val_idx][:, [0,1,2]]],data[val_idx][:,-1]==0,
            data[val_idx][:,-3]*( (data[val_idx][:, -1]==0)*2/avg_sweight + (data[val_idx][:, -1]>0)/avg_bweight)),
            shuffle=True , batch_size=1024,  epochs=50, verbose=1, callbacks=[reduce_lr, ohistory])

val_loss+=history.history['val_loss']
loss+=history.history['loss']
acc+=history.history['acc']
val_acc+=history.history['val_acc']
lr+=history.history['lr']

for layer in model.layers:
    if "ymass" in layer.name:
        layer.trainable=True
    if "nomass" in layer.name:
        layer.trainable=True   

ADAM=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=ADAM,metrics=['accuracy', 'binary_crossentropy'])
ohistory = OutHistory(name=str("unfreeze"))
            
      
history=model.fit([data[train_idx][:, 3:-6],data[train_idx][:, [0,1,2]]],data[train_idx][:,-1]==0, 
            sample_weight=data[train_idx][:,-3]*( (data[train_idx][:, -1]==0)*2/avg_sweight + (data[train_idx][:, -1]>0)/avg_bweight),
            validation_data=([data[val_idx][:, 3:-6],data[val_idx][:, [0,1,2]]],data[val_idx][:,-1]==0,
            data[val_idx][:,-3]*( (data[val_idx][:, -1]==0)*2/avg_sweight + (data[val_idx][:, -1]>0)/avg_bweight)),
            shuffle=True , batch_size=1024,  epochs=50, verbose=1, callbacks=[reduce_lr, ohistory])       

val_loss+=history.history['val_loss']
loss+=history.history['loss']
acc+=history.history['acc']
val_acc+=history.history['val_acc']
lr+=history.history['lr']
   
ohistory = OutHistory(name=str("unfreeze__"))

history=model.fit([data[train_idx][:, 3:-6],data[train_idx][:, [0,1,2]]],data[train_idx][:,-1]==0, 
            sample_weight=data[train_idx][:,-3]*( (data[train_idx][:, -1]==0)*2/avg_sweight + (data[train_idx][:, -1]>0)/avg_bweight),
            validation_data=([data[val_idx][:, 3:-6],data[val_idx][:, [0,1,2]]],data[val_idx][:,-1]==0,
            data[val_idx][:,-3]*( (data[val_idx][:, -1]==0)*2/avg_sweight + (data[val_idx][:, -1]>0)/avg_bweight)),
            shuffle=True , batch_size=10240,  epochs=200, verbose=1, callbacks=[reduce_lr, ohistory])            

val_loss+=history.history['val_loss']
loss+=history.history['loss']
acc+=history.history['acc']
val_acc+=history.history['val_acc']
lr+=history.history['lr']


print history.history.keys()
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss_history.png")
plt.clf()

plt.plot(acc)
plt.plot(val_acc)
plt.title('model acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("acc_history.png")
plt.clf()

plt.plot(lr)
plt.title('model lr')
plt.ylabel('learning rate')
plt.xlabel('epoch')
plt.legend(['learning rate adam'], loc='upper left')
plt.savefig("lr_history.png")
plt.clf()
