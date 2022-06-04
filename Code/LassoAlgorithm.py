from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from Abstractalgorithm import Abstractalgorithm

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class LassoAlgorithm(Abstractalgorithm):
    def define_model(self, alpha=1.0, n_splits=10, n_repeats=3, random_state=None):
        model = Lasso(alpha=alpha)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return model, cv

    def train_model(self, clf, X_train, y_train):
        model = clf.fit(X_train, y_train)
        return model
