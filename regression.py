#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:09:36 2019
quandl key: 
@author: jay
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import quandl
import math

quandl.ApiConfig.api_key = ''
df = quandl.get('WIKI/GOOGL')
#print(df.head())

df = df[['Adj. Open', 'Adj. Low', 'Adj. Close', 'Adj. High', 'Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'])*100.0
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'])*100.0

df = df[['HL_PCT', 'PCT_change', 'Adj. Volume', 'Adj. Close']]

#print(df.head())


'''
Change the  forecast column to the thing you want to
be predicting for training the test set.
'''
forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

#This is making 10% as test set.
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
Y = np.array(df['label'])

X = preprocessing.scale(X)

df.dropna(inplace=True)
Y = np.array(df['label'])

#To check length of both column
#print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

clf= LinearRegression()
clf.fit(X_train, Y_train)
confidence = clf.score(X_test, Y_test)
print(confidence)





