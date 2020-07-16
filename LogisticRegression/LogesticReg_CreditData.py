#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:32:55 2020

@author: muhammadsiraj
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

creditData = pd.read_csv('credit_data.csv')
print(creditData.head())
print(creditData.describe())    
print(creditData.corr())

features = creditData[["income","age","loan"]]
target = creditData.default

feature_train,feature_test,target_train,target_test = train_test_split(features, target,test_size = 0.3) 

model = LogisticRegression()
model.fit(feature_train,target_train )
print("=============================")
prediction  = model.predict(feature_test)

print(confusion_matrix(target_test, prediction))  
print(accuracy_score(target_test, prediction))

#X = [[66155.925095,59.017015,8106.532131]]
#print("my pred is ",model.predict(X))