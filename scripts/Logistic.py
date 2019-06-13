#!/usr/bin/env python
# coding: utf-8

# In[7]:


datapath='../../features'
Flods=[0,1,2,3]
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import os
print(os.getcwd())


# In[8]:



def sigmoid(x):
    return 1/(1+np.exp(-x))

def lrloss(w,X,Y,lmbda):

    loss=0
    for i in range(0,X.shape[0]):
        loss+=-np.log(sigmoid(Y[i]*X[i,:].dot(w))+0.0000000001)
    loss+=lmbda*sum(w[1:]**2)/2
    return loss

def lrgrad(w,X,Y,lmbda):
    P=sigmoid(np.multiply(np.matmul(X,w),Y))
    grad=-np.matmul(X.T,np.multiply(1-P,Y))
    grad[1:]+=lmbda*w[1:]
    
    return grad
    
def lrhess(w,X,Y,lmbda):
    P=sigmoid(np.multiply(np.matmul(X,w),Y))
    R=np.diagflat(np.multiply(P,1-P))
    hess=np.matmul(X.T,np.matmul(R,X))
    hess[1:,1:]+=lmbda*np.eye(w.shape[0]-1)
    
    return hess


# In[10]:



def graddesc(w,eta,fn,gradfn, ittfn=None):
    oldf = fn(w)
    df = 1
    while(df>1e-6):
        g = gradfn(w)
        #eta=0.3
        while eta>1e-10:
            neww = w - eta*g
            newf = fn(neww)
            if oldf>newf*1.001:
                break
            eta *= 0.5
        if ittfn is not None:
            ittfn(w,eta,newf)
        df=newf-oldf
        oldf = newf
        
        w = neww
    return w    

def newton(w,fn,gradfn,hessfn,ittfn=None):
    oldf=fn(w)
    df=1
    step_size=1.0
    while(df>1e-6):
        neww = w - np.linalg.solve(hessfn(w),gradfn(w))
        newf = fn(neww)
        df=oldf-newf
        if df<=0:
            step_size*=2
            while(step_size>1e-10):
                neww = w - step_size*gradfn(w)
                newf=fn(neww)
                df=oldf-newf
                if df>1e-6:
                    break
                step_size/=2
        if df>0:
            w=neww
            oldf=newf 
        
    return w
def trainGraddesc(X,Y,lmbda,eta):
    w0 = np.zeros((X.shape[1],1))
    return graddesc(w0,eta,
                   lambda w :lrloss(w,X,Y,lmbda),
                   lambda w :lrgrad(w,X,Y,lmbda))
def trainlr(X,Y,lmbda):
    w0 = np.zeros((X.shape[1],1)) # starting w at zero works well for LR
    return newton(w0,lambda w : lrloss(w,X,Y,lmbda),
                  lambda w : lrgrad(w,X,Y,lmbda),
                  lambda w : lrhess(w,X,Y,lmbda))

def lrerrorrate(X,Y,w):
    return np.sum(Y*np.matmul(X,w)<=0)/Y.shape[0]


# In[98]:


lmbdas=10**np.arange(1,7.1,0.3)
overall_err=[]
for flod in Flods:
    overall_err.append([])
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
    trainX=np.array(trainX)
    trainY=np.array(trainY)[:,np.newaxis].astype(int)
    trainY[trainY==0]=-1
    print(trainX.shape,trainY.shape)
    
    for line in testfile:
        line=line.split(',')[1:]
        if line[0]=='safe_type':
            continue
        line=[float(k) for k in line]
        testX.append(line[1:])
        testY.append(line[0])
    testX=np.array(testX)
    testY=np.array(testY)[:,np.newaxis].astype(int)
    testY[testY==0]=-1
    print(testX.shape,testY.shape)
    for lmbda in lmbdas:
        myw = trainlr(trainX,trainY,lmbda)
        overall_err[-1].append(lrerrorrate(testX,testY,myw))
        print(overall_err[-1][-1])
    plt.cla()
    plt.plot(lmbdas,overall_err[-1],'b',linewidth=2)
    plt.xscale('log')


    plt.xlabel('$\lambda$',fontsize=15)
    plt.ylabel('Error rate',fontsize=15)
    plt.title('Error distribution',fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.show()


# In[91]:


plt.cla()
plt.plot(lmbdas,np.mean(overall_err,0),'b',linewidth=2)
plt.xscale('log')


plt.xlabel('$\lambda$',fontsize=15)
plt.ylabel('Error rate',fontsize=15)
plt.title('Error distribution',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.show()


# In[14]:



lmbda=10**6

trainfile=open('{0}/full/train_features_full.csv'.format(datapath))
testfile=open('{0}/full/test_features_full.csv'.format(datapath))
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
trainY[trainY==0]=-1
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
testY[testY==0]=-1
print(testX.shape,testY.shape)
    
myw = trainlr(trainX,trainY,lmbda)
ans=sigmoid(np.matmul(testX,myw))
threshold=0.5
ans[ans>=threshold]=1
ans[ans<threshold]=-1
err=np.sum(testY*ans<=0)/float(testY.shape[0])
outfile=open('{0}/full/result_lr.csv'.format(datapath),'w')

testY[testY==-1]=0
ans[ans==-1]=0
for i in range(0,testY.shape[0]):
    outfile.write('{0},{1}\n'.format(int(testY[i][0]),ans[i][0]))
print(err)
outfile.close()


# In[15]:


infile=open('{0}/full/result_lr.csv'.format(datapath))
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


# In[ ]:




