# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:46:10 2019

@author: Rahul
"""

#Importing the libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

#Importing the dataset
dataset = pd.read_csv("curated_High_note_data.csv")
X = dataset.iloc[:, [1,2,3,6,8,9,10,11,12,21,22]].values
y = dataset.iloc[:, 7].values

#Splitting the dataset into train set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 0)

#Feature Scaling - The linear regression library will take care of this 
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

#Fitting the multiple regression regressor to the train set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting values for the test set 
y_pred = regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((91154, 1)).astype(int) , values = X , axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5,6,7,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()