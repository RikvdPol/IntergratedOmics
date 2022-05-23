
# First XGBoost model for Pima Indians dataset
import xgboost
import Logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
import sys, os
# load data


class XG:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None

    def extract_labels(self):
        sep = os.path.sep
        self.file = self.file.dropna(axis=0)
        logs = Logging.Logging()
        try:
            self.labels = self.file[self.labelname]
            msg = (f"Columnname {self.labelname} used as predictor label")
            logs.create_logs(self.__class__.__name__, msg)

        except KeyError as e:
            msg = f"{e}: Labelname {self.labelname} not present in data"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)

        self.file = self.file.drop([self.labelname, "Pseudo", "Antibody_batch"], axis=1)

    def split_data(self, test_size=0.3, random_state=None):
        """
        Split the data intro training and test data, and the labels intro training and test labels.
        The split is decided by the percentage of data to be used for testing, provided by a number between
        0 and 1. By defeault this number is set to 0.3, which means 30% of the data provided will be used
        for testing, and 70% will be used for training. In addition, the split function expects a random state.
        Providing a number will result in a specific random state, meaning that the same data is used for training
        and testing every single time if a number is provided, otherwise this will be random everytime. 
        This is random because is should not matter to the model which data it gets, it should always be comparable
        to previous instances.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.file,
                                                            self.labels,
                                                            test_size=test_size,
                                                            random_state=random_state)


        return X_train, X_test, y_train, y_test

    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        # fit model no training data
        model = XGBRegressor()
        return model, cv

    def train_model(self, X_train, y_train):
        elastic_model = XGBRegressor().fit(X_train, y_train)
        return elastic_model

    def predict(self, model, X_test):
        # make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        return predictions

    def evaluate_model(self, model, cv):
        scores = cross_val_score(model, self.file, self.labels, scoring=make_scorer(mean_squared_error), cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        return scores
