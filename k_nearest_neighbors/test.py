
"""
Created on Sun Aug 16 14:14:38 2020

@author: thispc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file=pd.read_csv('Social_Network_Ads.csv')
x=file.iloc[:,[2,3]].values
y=file.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_pred,y_test))

#Training set
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

fig,(ax1,ax2)=plt.subplots(1,2)

ax1.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

for i, j in enumerate(np.unique(y_set)):
    ax1.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
ax1.set_title('K-NN (Training set)')
ax1.set_xlabel('Age')
ax1.set_ylabel('Estimated Salary')


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
ax2.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

for i, j in enumerate(np.unique(y_set)):
    ax2.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
ax2.set_title('K-NN (Test set)')
ax2.set_xlabel('Age')
ax2.set_ylabel('Estimated Salary')