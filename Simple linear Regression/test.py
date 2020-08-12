import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

file=pd.read_csv('Salary_Data.csv')
x=file.iloc[:,:-1].values
y=file.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

mp.plot(x_train,model.predict(x_train),,color='red')
mp.scatter(x_train,y_train)
mp.title('Salary vs Experience (Training set)')
mp.xlabel('Years of Experience')
mp.ylabel('Salary')
mp.show()

mp.plot(x_test,model.predict(x_test),color='red')
mp.scatter(x_test,y_test)
mp.title('Salary vs Experience (Test set)')
mp.xlabel('Years of Experience')
mp.ylabel('Salary')
mp.show()
