
# First XGBoost model for Pima Indians dataset
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
# load data


class XG:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None


    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        model = XGBRegressor(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return model, cv

    def evaluate_model(self, model, cv):
        # metric = Metrics.Metrics()
        scores = cross_val_score(model, self.file, self.labels, scoring=make_scorer(mean_squared_error), cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        return scores

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


    def XG_boost(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None
        #dataset = pd.read_csv('Data/Covariates.csv', delimiter="\s+")
        # split data into X and y
        dataset = file

        dataset.dropna(inplace = True)
        X = dataset[['Age', 'Gender']]
        Y = dataset['BMI']

        Y = Y.astype(int)

        X = X.to_numpy()
        Y = Y.to_numpy()

        # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        # fit model no training data
        model = XGBRegressor()
        model.fit(X_train, y_train)
        # make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
