#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:58:12 2020

@author: Neeraj Prajapati
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('Position_Salaries.csv')

x=file.iloc[:,1:-1].values
y=file.iloc[:,-1].values
print(x,y)


from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
p=PolynomialFeatures(degree=2)
px=p.fit_transform(x)

print(px)

model2=LinearRegression()
model2.fit(px,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, model1.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) 

fig,(ax1,ax2)=plt.subplots(1,2)
ax1.scatter(x,y,color='red')
ax1.plot(x,model1.predict(x))

ax2.scatter(x,y,color='red')
ax2.plot(X_grid,model2.predict(p.fit_transform(X_grid)))
