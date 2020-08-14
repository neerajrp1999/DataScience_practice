#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:35:32 2020

@author: Neeraj Prajapati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file=pd.read_csv("Position_Salaries.csv")

x=file.iloc[:,1:-1].values
y=file.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()
