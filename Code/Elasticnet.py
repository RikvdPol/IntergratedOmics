from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
import numpy as np
import logging
import sys
import os


class Elasticnet:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None

    def extract_labels(self):
        sep = os.path.sep
        self.file = self.file.dropna(axis=0)
        try:
            self.labels = self.file[self.labelname]

        except KeyError as e:
            logging.basicConfig(filename=f'..{sep}Logfiles{sep}ElasticNet.log',
                                filemode='a+',
                                format='%(asctime)s %(message)s',
                                force=True)
            logging.warning(f'{e}: Labelname {self.labelname} not present in data')
            sys.exit(0)

        self.file = self.file.drop([self.labelname, "Pseudo", "Antibody_batch"], axis=1)
        # print(self.labels)

    def split_data(self, test_size=0.3, random_state=None):
        X_train, X_test, y_train, y_test = train_test_split(self.file,
                                                            self.labels,
                                                            test_size=0.25,
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test

    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return model, cv

    def train_model(self, X_train, y_train):
        elastic_model = ElasticNet().fit(X_train, y_train)
        return elastic_model

    def predict(self, elastic_model, X_train):
        predictions = elastic_model.predict(X_train)
        print(predictions[:10])

    def evaluate_model(self, model, cv):
        scores = cross_val_score(model, self.file, self.labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

