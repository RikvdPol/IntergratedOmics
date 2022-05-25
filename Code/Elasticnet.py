from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score, make_scorer
from sklearn.linear_model import ElasticNet
from Abstractalgorithm import Abstractalgorithm

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"

class Elasticnet(Abstractalgorithm):
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return model, cv

    def train_model(self, X_train, y_train):
        elastic_model = ElasticNet().fit(X_train, y_train)
        return elastic_model

