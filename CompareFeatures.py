# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:22:51 2019

@author: Admin
"""

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import cv2
from keras import applications
import numpy as np
from keras.applications.resnet50 import preprocess_input
import pandas as pd

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
trainX,testX,trainY,testY = train_test_split(x_train,y_train, test_size=0.2)
imgHistogram = []
imgHarris = []

def histFeatures(imgDS,mode):
    featureDB=[]
    if mode=="DB":
        for i in range (0,imgDS.shape[0]):
            hist = cv2.calcHist(imgDS[i], [0], None, [256], [0, 256])
            featureDB.append(hist)
        return featureDB
    else:
        hist = cv2.calcHist(imgDS, [0], None, [256], [0, 256])
        return hist

        

def shapeFeatures(imgDS,mode):
   featureDB=[]
   if mode=="DB":
        for i in range(0,imgDS.shape[0]):
        #img = trainX[65]
            dst = cv2.cornerHarris(cv2.cvtColor(imgDS[i],cv2.COLOR_BGR2GRAY),2,3,0.04)
            dst = cv2.dilate(dst,None)
            featureDB.append(dst)
        return featureDB
   else:
        dst = cv2.cornerHarris(cv2.cvtColor(imgDS,cv2.COLOR_BGR2GRAY),2,3,0.04)
        dst = cv2.dilate(dst,None)
        return dst
      
      
        
def deepLearnFeatures(imgDS,mode):
        model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        if mode=="DB":
            featureDB =[]
            for i in range(0,imgDS.shape[0]):
                x = imgDS[i]
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)[0]
                featureDB.append(features)
            return featureDB
        else:
            x = imgDS
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)[0]
            
            return features
        
def calcSim(featDB,testDB,simScore,imgClassTrain,imgClassTest):
    imgclasstest=[]
    imgclasstrain=[]
    #for i in range(0,len(testDB)):
    for j in range(0,len(featDB)):
        #imgclasstest.append(imgClassTest[i])
        imgclasstrain.append(imgClassTrain[j])
    #dis = distance.euclidean(featDB[j],query)
        distance = np.linalg.norm(featDB[j]-testDB)
        simScore.append(distance)
        
   # print(len(imgclasstest),len(imgClassTrain.flatten()),len(simScore))        
    df = pd.DataFrame({'TestImg':imgclasstest,'Labels':imgclasstrain,'Score':simScore})
    
    df= df.sort_values(by=['TestImg','Score'])
    print(df.groupby(['TestImg']).apply(lambda x: x.nsmallest(5,'Score')))
    print(df.groupby(['TestImg']).apply(lambda x: x.nlargest(5,'Score')))
    


            
def main():
    simScore1=[]
    simScore2=[]
    simScore3=[]
    imgClassTrain = list(trainY.flatten())
    imgClassTest = list(testY[:10].flatten())
  
    features1 = deepLearnFeatures(trainX[:500],"DB")
    deepFeat = deepLearnFeatures(testX[10],"Query")
    
    features2 = histFeatures(trainX,"DB")
    histFeat = histFeatures(testX[:10],"DB")
    
    features3 = shapeFeatures(trainX,"DB")
    shapeFeat = shapeFeatures(testX[:10],"DB")
    
    calcSim(features1,deepFeat,simScore1,imgClassTrain,imgClassTest)
    calcSim(features2,histFeat,simScore2,imgClassTrain,imgClassTest)
    calcSim(features3,shapeFeat,simScore3,imgClassTrain,imgClassTest)
 

    

