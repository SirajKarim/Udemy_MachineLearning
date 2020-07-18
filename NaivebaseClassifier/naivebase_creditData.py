#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:42:01 2020

@author: muhammadsiraj
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('/home/muhammadsiraj/Udemy-ML/LogisticRegression/credit_data.csv')

features = dataset[["income","age","loan"]]
target = dataset.default

feature_train,feature_test,target_test,target_train = train_test_split(features, target, test_size=0.3)
classifier = GaussianNB()
classifier.fit(features, target)

predictions = classifier.predict(feature_test) 


#print(confusion_matrix(target_test, predictions))  #yahan koe bhand ha
#print(accuracy_score(target_test,predictions))      #yahan koe bhand ha