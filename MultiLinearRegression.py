# Importing Lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import Data set
DS = pd.read_csv("50_Startups.csv")

state = DS['State']

state = pd.get_dummies(state, drop_first = True)


X_data = DS[['R&D Spend', 'Administration', 'Marketing Spend']]

X = np.hstack((X_data,state))
Y = DS.iloc[:,-1]

x_train, x_test, y_train, y_test= train_test_split(X,Y,test_size=0.2)

LR = LinearRegression()
LR.fit(x_train,y_train)
y_pred = LR.predict(x_train)


plt.scatter(y_train, y_pred, c="r")
plt.plot(y_train, y_pred, c="b")
plt.show()

sns.pairplot(DS)
plt.show()


