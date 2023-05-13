import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets ,linear_model
from sklearn.metrics import mean_squared_error
dat = datasets.load_diabetes()

datax = dat.data[: ,np.newaxis,2]

dxtrain = datax[:-30]
dxtest = datax[-30:]

dataytrain = dat.target[:-30]
dataytest= dat.target[-30:]

model = linear_model.LinearRegression()
model.fit(dxtrain,dataytrain)
dataypredict = model.predict(dxtest)

print("MSE is = ",mean_squared_error(dataytest,dataypredict))

print("weight "  , model.coef_ )
print("Intercept "  , model.intercept_ )

plt.scatter(dxtest,dataytest)
plt.plot(dxtest,dataypredict)
plt.show()




