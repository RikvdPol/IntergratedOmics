from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from AlgorithmBaseClass import AlgorithmBaseClass

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Elasticnet(AlgorithmBaseClass):
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return clf, cv
