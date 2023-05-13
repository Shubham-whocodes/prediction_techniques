import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
x= iris["data"][:, 3:]
y= (iris["target"]==2).astype(np.int)
clf = LogisticRegression()
clf.fit(x,y)
ex= clf.predict(([[2.6]]))
print(ex)
