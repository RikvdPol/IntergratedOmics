from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from Abstractalgorithm import Abstractalgorithm
import numpy as np

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Elasticnet(Abstractalgorithm):
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        # model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return cv

    def train_model(self, clf, X_train, y_train):
        model = clf.fit(X_train, y_train)
        return model

    def tune_hyperparameters(self):
        params = {"alpha": np.arange(0.1, 1.1, 0.1),
                  "l1_ratio": np.arange(0.1, 1.1, 0.1),
                  "max_iter": np.arange(100, 2100, 100),
                  "tol": np.arange(1e-6, 1e-2, 1e-1)}
        clf = GridSearchCV(
            estimator=ElasticNet(),
            param_grid=params,
            cv=10,
            n_jobs=30,
            verbose=1
        )
        return clf

