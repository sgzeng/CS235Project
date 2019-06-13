#!/usr/bin/env python
# coding: utf-8

# In[1]:


datapath='../../features'
Flods=[0,1,2,3]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import keras
from keras import optimizers
import matplotlib.pyplot as plt
from IPython import display
import time
import os
print(os.getcwd())


# In[80]:


class Network:

    def __init__(self, input_shape, num_classes, x_train, y_train,x_test,y_test):

        self.model = Sequential()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.x_train = x_train
        self.x_test=x_test
        self.y_train = []
        self.y_test=[]
        self.y_clean=[]
        for i in range(0, len(y_train)):
            self.y_train.append(y_train[i])
            self.y_clean.append(y_train[i])
        for i in range(0,len(y_test)):
            self.y_test.append(y_test[i])

        self.y_test=np.array(self.y_test)
        self.y_train=np.array(self.y_train)
        self.y_clean=np.array(self.y_clean)

        k=128
        self.model.add(Dense(k,activation='relu',input_shape=input_shape,
                              kernel_initializer=keras.initializers.RandomNormal(0,0.001)))
        self.model.add(Dense(k/2,activation='relu',
                             kernel_initializer='he_normal'))
        self.model.add(Dense(num_classes,activation='relu',
                             kernel_initializer=keras.initializers.RandomNormal(0,0.001/float(k))))
        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizers.Adadelta(), metrics=["accuracy"])

    def set_weights(self, weight):
        self.model.set_weights(weight)

    def get_weights(self):
        return self.model.get_weights()

    def fit(self, batch_size, epochs,verbose=1,callbacks=[]):
        self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(self.x_test,self.y_test),
            shuffle=True
        )


# In[77]:


from keras import callbacks
import numpy as np
class History(callbacks.Callback):
    def __init__(self):
        self.train_acc=[]
        self.train_loss=[]
        self.test_acc=[]
        self.test_loss=[]
    def on_epoch_end(self,epoch,logs={}):
        self.train_acc.append(logs.get('acc'))
        self.train_loss.append(logs.get('loss'))
        self.test_acc.append(logs.get('val_acc'))
        self.test_loss.append(logs.get('val_loss'))
    def on_train_end(self,logs={}):
        np.save('../result/train_acc',self.train_acc)
        np.save('../result/train_loss',self.train_loss)
        np.save('../result/test_acc',self.test_acc)
        np.save('../result/test_loss',self.test_loss)
        plt.cla()
        plt.plot(self.train_acc,'r',linewidth=2)
        plt.plot(self.test_acc,'b',linewidth=2)
        plt.legend(['train','test'])


# In[45]:


batch_size=128
epoch=200
num_classes=1
for flod in Flods:
    trainfile=open('{0}/cv_{1}_4_{2}_4/cv_train_features_{1}_4_{2}_4.csv'.format(datapath,flod,flod+1))
    testfile=open('{0}/cv_{1}_4_{2}_4/cv_test_features_{1}_4_{2}_4.csv'.format(datapath,flod,flod+1))
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    for line in trainfile:
        line=line.split(',')[1:]
        if line[0]=='safe_type':
            continue
        line=[float(k) for k in line]
        trainX.append(line[1:])
        trainY.append(line[0])
    trainX=np.array(trainX)[:,:]
    meanX=np.mean(trainX)
    trainX-=meanX
    trainY=np.array(trainY).astype(int)
    #trainY[trainY==0]=-1
    print(trainX.shape,trainY.shape)
    
    for line in testfile:
        line=line.split(',')[1:]
        if line[0]=='safe_type':
            continue
        line=[float(k) for k in line]
        testX.append(line[1:])
        testY.append(line[0])
    testX=np.array(testX)[:,:]
    testX-=meanX
    testY=np.array(testY).astype(int)
    print(testX.shape,testY.shape)
    input_shape=(trainX.shape[1],)
    net = Network(input_shape, num_classes, trainX, trainY, testX, testY)
    net.fit(batch_size,epoch,verbose=1)
    print(net.model.evaluate(testX,texsY))


# In[81]:


batch_size=128
epoch=300
trainfile=open('{0}/full/train_features_full.csv'.format(datapath,flod,flod+1))
testfile=open('{0}/full/test_features_full.csv'.format(datapath,flod,flod+1))
trainX=[]
trainY=[]
testX=[]
testY=[]
for line in trainfile:
    line=line.split(',')[1:]
    if line[0]=='safe_type':
        continue
    line=[float(k) for k in line]
    trainX.append(line[1:])
    trainY.append(line[0])
trainX=np.array(trainX)[:,:5]
trainY=np.array(trainY)[:,np.newaxis].astype(int)
print(trainX.shape,trainY.shape)

for line in testfile:
    line=line.split(',')[1:]
    if line[0]=='safe_type':
        continue
    line=[float(k) for k in line]
    testX.append(line[1:])
    testY.append(line[0])
testX=np.array(testX)[:,:5]
testY=np.array(testY)[:,np.newaxis].astype(int)
print(testX.shape,testY.shape)

input_shape=(trainX.shape[1],)
net = Network(input_shape, num_classes, trainX, trainY, testX, testY)


net.fit(batch_size,epoch,verbose=1,callbacks=[History()])
print(net.model.evaluate(testX,testY))


# In[82]:



train_acc=np.load('../result/train_acc.npy')
train_loss=np.load('../result/train_loss.npy')
test_acc=np.load('../result/test_acc.npy')
test_loss=np.load('../result/test_loss.npy')
plt.cla()
plt.plot(train_acc,'r',linewidth=3)
plt.plot(test_acc,'b',linewidth=3)
plt.legend(['train','test'])


# In[87]:


ans=net.model.predict(testX)
print(ans.shape)
ans=ans[:,0]
ans[ans>=0.5]=1
ans[ans<0.5]=0
outfile=open('{0}/full/result_NN.csv'.format(datapath),'w')
for i in range(0,testY.shape[0]):
    outfile.write('{0},{1}\n'.format(int(testY[i][0]),ans[i]))
outfile.close()


# In[95]:


infile=open('{0}/full/result_NN.csv'.format(datapath))
count=0
score=0
for line in infile:
    count+=1
    line=line.replace('\n','').split(',')
    line=np.array([float(k) for k in line]).astype(int)
    if line[0]==line[1]:
        score+=1
    score-=(line[1]-line[0])*line[1]
print(float(score)/count)
infile.close()

