# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:57:14 2019

@author: Arnau
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

datapath = 'C:/Users/Arnau/Documents/GitHub/Reinforcement-learning-to-play-Snake/data.csv'
data = pd.read_csv(datapath)

def prepare_data(ds):
    X_cols = ds.copy()
    X = X_cols.values
    X = X.reshape(len(X_cols),-1)
    
    #We add the dummy x_0
    poly = PolynomialFeatures(1)  
    X = poly.fit_transform(X)
    
    print(X)

    return X

def fit_logreg(X, y):
    '''
    Wraps initialization and training of Logistic regression
    '''
    logreg = LogisticRegression(C=1e20, solver='liblinear', max_iter=500) #
    logreg.fit(X, y)
    
    return logreg

def predict_and_test(model, X_test, y_test):
    '''
'    Predicts using a model received as input and then evaluates the accuracy of the predicted data. 
    As inputs it receives the model, an input dataset X_test and the corresponding targets (ground thruth) y_test
    It returs the classification accuracy.
    '''
    y_hat =model.predict(X_test)
    y_test = np.array(y_test)
    
    correct = 0
    
    for i in range(len(y_test)):
      if y_hat[i] == y_test[i]:
        correct+=1
        
    samples = len(y_test)

    
    return correct/samples

x_train = data.drop(['Action', 'Unnamed: 0'], axis=1)
y_train = data[['Action']]



logreg = fit_logreg(x_train, y_train)
print('*************** Estimated parameters: ***********************')
print('[W_0,W] : [',logreg.intercept_,',', logreg.coef_, ']' )

acc_test = predict_and_test(logreg,x_train, y_train)
print('******************  Testing accuracy *********************')
print('ACC multinomial: ', acc_test)