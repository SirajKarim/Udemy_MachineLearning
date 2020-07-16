#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:56:41 2020

@author: muhammadsiraj
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

creditData = pd.read_csv('credit_data.csv')
print(creditData.head())
print(creditData.describe())    
print(creditData.corr())

features = creditData[["income","age","loan"]]
target = creditData.default

model = LogisticRegression()
predicted = cross_validation.cross_val_predict(model,features,target,cv=10)