# -*- coding: utf-8 -*-

def LinearRegression(data):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        X = ['x1', 'x2', 'x3'] #independent variables
        y = 'biomarkers' #dependent variable
        regr.fit(X, y)