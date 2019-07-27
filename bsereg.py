#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import quandl
import math, datetime

df = quandl.get("BSE/BOM539678", authtoken="Mhy8p_3pCoqs2o7cGuF-")

df = df[['Spread H-L', 'Spread C-O', 'Close', 'Open']]

forecast_column = 'Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.02*len(df)))
df['label'] = df[forecast_column].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately= X[-forecast_out:]
X = X[:-forecast_out:]
df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

clf= LinearRegression()
clf.fit(X_train, Y_train)
confidence = clf.score(X_test, Y_test)
print(confidence)
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
    
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

print(df['Forecast'])

