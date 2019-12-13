# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:25:47 2019

@author: Arnau
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import neighbors

n_neighbors = 1

datapath = 'data.csv'
data = pd.read_csv(datapath)

def prepare_data(ds):
    X_cols = ds.copy()
    X = X_cols.values
    X = X.reshape(len(X_cols),-1)
    
    #We add the dummy x_0 and potentially features’ high-order
    poly = PolynomialFeatures(1)  
    X = poly.fit_transform(X)

    return X

def features_weights_body(df):
    for i in range(20):
        for j in range(20):
            df[str(i)+"&"+str(j)] *= 10**-2

def fit_kNN(X, y):
    '''
    # we create an instance of Neighbours Classifier and fit the data.
    '''
    for weights in ['uniform', 'distance']:
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
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

#x_train = data.drop(['Action', 'Unnamed: 0'], axis=1)
x_train = data[['Headx','Heady','Applex','Appley','x+','x-','y+','y-']]
#x_train = prepare_data(x_train)
#features_weights_body(x_train)

y_train = data[['Action']]



clf = fit_kNN(x_train, y_train)

acc_test = predict_and_test(clf,x_train, y_train)
print('******************  Training *********************')
print('ACC multinomial: ', acc_test)