# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:57:14 2019

@author: Arnau
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

datapath = 'data.csv'
data = pd.read_csv(datapath)

def prepare_data(ds):
    X_cols = ds.copy()
    X = X_cols.values
    X = X.reshape(len(X_cols),-1)
    
    #We add the dummy x_0 and potentially features’ high-order
    poly = PolynomialFeatures(2)  
    X = poly.fit_transform(X)

    return X

def fit_logreg(X, y):
    '''
    Wraps initialization and training of Logistic regression
    '''
    logreg = LogisticRegression(C=1e20, solver='saga',dual=False,n_jobs=-1, max_iter=1400) #
    logreg.fit(X, y)
    
    return logreg

def predict_and_test(model, X_test, y_test):
    '''
'    Predicts using a model received as input and then evaluates the accuracy of the predicted data. 
    As inputs it receives the model, an input dataset X_test and the corresponding targets (ground thruth) y_test
    It returs the classification accuracy.
    '''
    y_hat = np.array(model.predict(X_test))
    y_test = np.array(y_test)[:,0]
    correct = np.sum(y_hat==y_test)
    samples = y_test.size
    return correct/samples

#x_train = data.drop(['Action', 'Unnamed: 0'], axis=1)
x_train = data[['Headx','Heady','Applex','Appley','x+','x-','y+','y-']]
x_train = prepare_data(x_train)

y_train = data[['Action']]



clf = fit_logreg(x_train, y_train)
print('*************** Estimated parameters: ***********************')
print('[W_0,W] : [',clf.intercept_,',', clf.coef_, ']' )

acc_test = predict_and_test(clf,x_train, y_train)
print('******************  Training accuracy *********************')
print('ACC multinomial: ', acc_test)