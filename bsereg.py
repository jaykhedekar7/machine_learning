#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
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