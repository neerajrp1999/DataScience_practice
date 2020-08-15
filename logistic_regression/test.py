#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:25:50 2020

@author: Neeraj Prajapati
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('Social_Network_Ads.csv')
x=file.iloc[:,[2,3]].values
y=file.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_pred, y_test)
print(c)
x_set, y_set = x_test, y_test

from matplotlib.colors import ListedColormap

#test
x_1,x_2=np.meshgrid(np.arange(start=x_test[:,0].min()-1,stop=x_test[:,0].max()+1,step=0.001),
                    np.arange(start=x_test[:,1].min()-1,stop=x_test[:,1].max()+1,step=0.001))
plt.contour(x_1,x_2,model.predict(np.array([x_1.ravel(),x_2.ravel()]).T).reshape(x_1.shape),cmap = ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.show()

#train
x_set, y_set = x_train, y_train
x_1,x_2=np.meshgrid(np.arange(start=x_train[:,0].min()-1,stop=x_train[:,0].max()+1,step=0.001),
                    np.arange(start=x_train[:,1].min()-1,stop=x_train[:,1].max()+1,step=0.001))
plt.contour(x_1,x_2,model.predict(np.array([x_1.ravel(),x_2.ravel()]).T).reshape(x_1.shape),cmap = ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.show()

