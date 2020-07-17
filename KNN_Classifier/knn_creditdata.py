#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:20:16 2020

@author: muhammadsiraj
"""


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn import preprocessing

dataset = pd.read_csv('credit_data.csv') 
features = dataset[["income","age","loan"]]
target = dataset.default

features = preprocessing.MinMaxScaler().fit_transform(features)
feature_train,feature_test,target_train,target_test = train_test_split(features, target, test_size = 0.3)
model = KNeighborsClassifier(n_neighbors=20)
model.fit(features, target)    
prediction = model.predict(feature_test)

print(confusion_matrix(target_test, prediction))
print(accuracy_score(target_test, prediction))