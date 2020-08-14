#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:35:32 2020

@author: Neeraj Prajapati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file=pd.read_csv("/home/thispc/Downloads/P14-Part2-Regression/Section 9 - Support Vector Regression (SVR)/Python/Position_Salaries.csv")

x=file.iloc[:,1:-1].values
y=file.iloc[:,-1].values
y = y.reshape(len(y),1)
from sklearn.preprocessing import StandardScaler

xscaler=StandardScaler()
yscaler=StandardScaler()

x=xscaler.fit_transform(x)
y=yscaler.fit_transform(y)

from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(x,y)

plt.scatter(xscaler.inverse_transform(x),yscaler.inverse_transform(y),color='red')
plt.plot(xscaler.inverse_transform(x),yscaler.inverse_transform(svr.predict(x)))
