from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
import numpy as np


class Elasticnet:
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname
        self.labels = None

    def extract_labels(self):
        self.file = self.file.dropna(axis=0)
        self.labels = self.file[self.labelname]
        self.file = self.file.drop([self.labelname, "Pseudo", "Antibody_batch"], axis=1)
        # print(self.labels)

    def split_data(self):
        pass

    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
        return model, cv

    def train_model(self):
        pass

    def predict(self):
        pass

    def evaluate_model(self, model, cv):
        scores = cross_val_score(model, self.file, self.labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # print(scores)

