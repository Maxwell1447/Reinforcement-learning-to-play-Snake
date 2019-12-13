# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:09:21 2019

@author: Arnau
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier

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

def fit_MLP(X, y):
    '''
    Wraps initialization and training of Logistic regression
    '''
    clf = MLPClassifier(activation='relu',solver='adam', max_iter=1000,verbose=True, tol=1e-5) 
    clf.fit(X, y)
    
    return clf

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

x_train = data.drop(['Action', 'Unnamed: 0'], axis=1)
#x_train = data[['Headx','Heady','Applex','Appley','x+','x-','y+','y-']]


y_train = data[['Action']]



clf = fit_MLP(x_train, y_train)
acc_test = predict_and_test(clf,x_train, y_train)
print('******************  Training accuracy *********************')
print('ACC multinomial: ', acc_test)