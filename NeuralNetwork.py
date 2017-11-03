# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:35:26 2017

@author: pig84
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection

def load_train_test_data(train_ratio):
    df = pd.read_csv('train.csv')
    y = df['label']
    X = df.drop(['label'], axis = 1)
    return sklearn.model_selection.train_test_split(X, y, train_size = train_ratio, random_state = 0)

def build_network():
    network = {}
    network['w1'] = np.array([[0, 0, 0], [0, 0, 0]])
    network['b1'] = np.array([0])
    network['w2'] = np.array([[0, 0, 0], [0, 0, 0]])
    network['b2'] = np.array([0])
    return network

def main():
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio = 0.8)
    network = build_network()
    

main()    