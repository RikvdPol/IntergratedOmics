
# First XGBoost model for Pima Indians dataset
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data

class XG:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None

    def extract_labels(self):
        self.file = self.file.dropna(axis=0)
        self.labels = self.file[self.labelname]
        X = self.file[['Age', 'Gender']] #independent variables
        Y = self.file['BMI']  #dependent variables

        Y = Y.astype(int)

        X = X.to_numpy()
        Y = Y.to_numpy()
        return X, Y

    def define_model(self, X, Y, seed = 7, testsize = 0.33):
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        # fit model no training data
        model = XGBRegressor()
        model.fit(X_train, y_train)
        return model, X_train, X_test, y_train, y_test


    def predict(self, model, X_train, X_test, y_train, y_test):
        # make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]

    def evaluate(self, y_test, predictions):
                # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
