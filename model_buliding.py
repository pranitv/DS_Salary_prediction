# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:57:23 2021

@author: PRANIT
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Data_eda.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)

# choose relevant columns
df_model = df[['avg_Salary','Rating','Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'No_of_Competitors', 'hourly', 'employer provided', 
               'job_state','same_state','Age', 'python_ya', 'spark_ya', 'aws_ya', 'excel_ya','job_simp', 'seniority', 'desc_length']]

# create dummy variable
df_dum = pd.get_dummies(df_model)

# train_test_split
from sklearn.model_selection import train_test_split

x = df_dum.drop('avg_Salary',axis=1)
y = df_dum.avg_Salary.values

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# Mltiple Linear Regression
import statsmodels.api as sm
X_sm = X = sm.add_constant(x)
model = sm.OLS(y,X_sm)
model.fit().summary() 

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
lr.fit(X_train,y_train)
print(np.mean(cross_val_score(lr,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)))


# lasso Regression
lm_l = Lasso(alpha =0.2)
lm_l.fit(X_train,y_train)
print(np.mean(cross_val_score(lm_l,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)))

error = []
alpha = []
for i in range(1,10):
    alpha.append(i/10)
    lm_l = Lasso(alpha = i/10)
    error.append(np.mean(cross_val_score(lm_l,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)))

plt.plot(alpha,error)
error = tuple(zip(alpha,error))
error_df = pd.DataFrame(error,columns=['alpha','error'])

error_df[error_df['error']==max(error_df['error'])]


# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
np.mean(cross_val_score(rf,X_train,y_train,scoring='neg_mean_absolute_error',cv=3))

# tune model using gridsearch cv
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10),'criterion':['mse','mae'],'max_features':['auto','sqrt','log2']}
gs = GridSearchCV(rf, parameters,scoring='neg_mean_absolute_error',cv=3)

gs.fit(X_train,y_train)
# test ensemble
lm_pred = lm_l.predict(X_test)
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
gs_pred = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,lm_pred)
mean_absolute_error(y_test,lr_pred)
mean_absolute_error(y_test,rf_pred)
mean_absolute_error(y_test,gs_pred)

mean_absolute_error(y_test, (lr_pred+gs_pred)/2)

import pickle
filename = 'RandomForest_model.sav'
pickle.dump(rf, open(filename, 'wb'))
