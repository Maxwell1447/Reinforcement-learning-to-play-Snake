# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:25:47 2019

@author: Arnau
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import neighbors

n_neighbors = 2

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
    y_hat =model.predict(X_test)
    y_test = np.array(y_test)
    
    correct = 0
    
    for i in range(len(y_test)):
      if y_hat[i] == y_test[i]:
        correct+=1
        
    samples = len(y_test)

    
    return correct/samples

x_train = data[['Headx','Heady','Applex','Appley','x+','x-','y+','y-']]
y_train = data[['Action']]



clf = fit_kNN(x_train, y_train)

acc_test = predict_and_test(clf,x_train, y_train)
print('******************  Testing accuracy *********************')
print('ACC multinomial: ', acc_test)