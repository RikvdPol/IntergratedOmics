# -*- coding: utf-8 -*-

import sys

def LinearRegression(data):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        desired_variables = sys.argv()
        X = ['x1', 'x2', 'x3'] #independent variables
        y = 'biomarkers' #dependent variable
        #regr.fit(X, y)