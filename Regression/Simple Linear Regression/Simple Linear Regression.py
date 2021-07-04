# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:37:31 2021

@author: TE273350
"""
#Simple Linear Regression

#import necessary libraries
from sklearn import linear_model
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

#List of 30 elements depicting independent variable height
height=[145,163,152,137,175,131,164,148,186,178,156,179,163,152,131,138,166,189,133,178,161,149,189,176,159,162,174,146,139,153]
print(len(height))

#List of 30 elements depicting dependednt variable weight
weight=[51,56,53,48,60,53,71,56,79,68,55,67,62,51,48,53,68,83,54,74,58,52,78,67,53,56,71,50,47,51]
print(len(weight))

#Assumption 1: Determining Normality of data (NORMALITY)
print("Skewness of height: ",stats.skew(height))
print("Kurtosis of height: ",stats.kurtosis(height))
print("Skewness of weight: ",stats.skew(weight))
print("Kurtosis of weight: ",stats.kurtosis(weight))

#Assumption 2: Correlation between dependent, independent variables (LINEARITY)
#Null Hypothesis: There is no significant correlation between weight and height
print("Spearman Correlation: ",stats.stats.spearmanr(height,weight))
print("Pearson Correlation: ",stats.stats.pearsonr(height,weight))

#Creating a scatterplot between height and weight
plt.scatter(weight,height)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter plot between weight and height")
plt.show()

#Converting list into dataframe
heightdf=pd.DataFrame(height)
weightdf=pd.DataFrame(weight)

#Creating a linear regression model
model=linear_model.LinearRegression()

#Fitting the model using fit() function
model.fit(heightdf,weightdf)
print("Adjusted R Squared for the linear regression model: ",model.score(heightdf,weightdf))

#Equation coefficient and intercept
print("Coefficient of independent variables: ",model.coef_)
print("Intercept in model: ",model.intercept_)

#Creating a new dataframe of independent variable height
testheight=pd.DataFrame([172,180,176])

#Predicting the values of weight depending on height
print("Predicted weight for 172cm: ",model.predict(testheight)[0])
print("Predicted weight for 180cm: ",model.predict(testheight)[1])
print("Predicted weight for 176cm: ",model.predict(testheight)[2])