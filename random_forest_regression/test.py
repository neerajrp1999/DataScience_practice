#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:49:25 2020

@author: Neeraj Prajapati
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file=pd.read_csv('Position_Salaries.csv')
x=file.iloc[:,1:-1].values
y=file.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=0)
model.fit(x,y)

x_x=np.arange(np.min(x),np.max(x),0.1)
x_x=np.reshape(x_x, (len(x_x),1))

plt.scatter(x,y,color='red')
plt.plot(x_x,model.predict(x_x))
plt.show()