#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:09:36 2019
quandl key:
@author: jay
"""
#First Project
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import quandl
import math, datetime

quandl.ApiConfig.api_key = ''
df = quandl.get('WIKI/GOOGL')
#print(df.head())

df = df[['Adj. Open', 'Adj. Low', 'Adj. Close', 'Adj. High', 'Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'])*100.0
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'])*100.0
df = df[['HL_PCT', 'PCT_change', 'Adj. Volume', 'Adj. Close']]

#Change the  forecast column to the thing you want to be predicting for training the test set.
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True) check what this does
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X =X[:-forecast_out:]
df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

clf= LinearRegression()
clf.fit(X_train, Y_train)
confidence = clf.score(X_test, Y_test)
forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

last_date =df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

