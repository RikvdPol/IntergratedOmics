from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import sys
import os
import Logging

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Abstractalgorithm:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        # self.labels = None

    # def extract_labels(self):
    #     sep = os.path.sep
    #     self.file = self.file.dropna(axis=0)
    #     logs = Logging.Logging()
    #     try:
    #         self.labels = self.file[self.labelname]
    #         msg = (f"Columnname {self.labelname} used as predictor label")
    #         logs.create_logs(self.__class__.__name__, msg)
    #
    #     except KeyError as e:
    #         msg = f"{e}: Labelname {self.labelname} not present in data"
    #         logs.create_logs(self.__class__.__name__, msg)
    #         sys.exit(0)
    #
    #     self.file = self.file.drop([self.labelname, "Pseudo", "Antibody_batch"], axis=1)

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
                                                            self.labelname,
                                                            test_size=test_size,
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test

    def predict(self, model, X_test):
        predictions = model.predict(X_test)
        return predictions

    def evaluate_model(self, model, cv):
        # metric = Metrics.Metrics()
        scores = cross_val_score(model, self.file, self.labelname, scoring=make_scorer(mean_absolute_error), cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        return scores

    def define_model(self, alpha=1.0, n_splits=10, n_repeats=3, random_state=None):
        pass

    def train_model(self, clf, X_train, y_train):
        model = clf.fit(X_train, y_train)
        return model


        