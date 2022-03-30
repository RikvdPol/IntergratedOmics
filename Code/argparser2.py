# -*- coding: utf-8 -*-

import pandas as pd

def arg_parser():
    '''arg parser'''
    import argparse
    parser = argparse.ArgumentParser(description='Use -i to specify the input file, use -o to specify the output file, use -t to specify the type of required ML method, use -h to see the help')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-t', '--type', help='Type of ML method', required=True)
    args = parser.parse_args()
    return args

filepath = arg_parser().input
output = arg_parser().output
ML_type = arg_parser().type

with open(filepath, 'r') as f:
    data = f.read()

def LinearRegression():
        from sklearn import linear_model
        data = pd.read_csv(filepath)
        regr = linear_model.LinearRegression()
        X = ['x1', 'x2', 'x3'] #independent variables
        y = 'biomarkers' #dependent variable
        regr.fit(X, y)

def switch(ML_type):
    if ML_type == 'LinearRegression':
        return LinearRegression()
        '''
    elif ML_type == 'LogisticRegression':
        return LogisticRegression()
    elif ML_type == 'KNN':
        return KNN()
    elif ML_type == 'DecisionTree':
        return DecisionTree()
    elif ML_type == 'RandomForest':
        return RandomForest()
    elif ML_type == 'SVM':
        return SVM()
    elif ML_type == 'NaiveBayes':
        return NaiveBayes()
    elif ML_type == 'XGBoost':
        return XGBoost()
    elif ML_type == 'LGBM':
        return LGBM()
        '''
    else:
        print('Wrong ML type')

switch(ML_type)





