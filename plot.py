import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sb
from sklearn.linear_model import LogisticRegression as model
from sklearn.model_selection import train_test_split
data = pd.read_csv('iris.csv')
# data['variety'].replace({'Setosa':'1','Versicolor':'2','Virginica':'3'}, inplace =True)
print(data)
xtr,xte,ytr,yte = train_test_split(data[['sepal.length','sepal.width','petal.length','petal.width']],data[['variety']],test_size = 0.2)
model.fit(xtr,ytr)
model.predict(xte)