import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
feature= iris.data
label = iris.target
# print(feature[0],label[0])
clf = KNeighborsClassifier()
clf.fit(feature,label)
predict = clf.predict([[12,12,11,1]])
print(predict)
