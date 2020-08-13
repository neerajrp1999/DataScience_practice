import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file=pd.read_csv('50_Startups.csv')
x=file.iloc[:,:-1].values
y=file.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrought')
x=np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)

y_p=model.pedict(y_train)
print('y_p',y_p)
print('y_test',y_test)

