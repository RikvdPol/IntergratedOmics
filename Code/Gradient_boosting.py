'''gradient boosting algorithm'''
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from Abstractalgorithm import Abstractalgorithm

__author__ = "Ruben Otter"
__license__ = "MIT"
__email__ = "rubenotter@hotmail.com"
__status__ = "Version 1.0"

class Gradientboost(Abstractalgorithm):
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        model = GradientBoostingRegressor(alpha=alpha, l1_ratio=l1_ratio)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return model, cv

    def train_model(self, X_train, y_train):
        gradient_model = GradientBoostingRegressor().fit(X_train, y_train)
        return gradient_model
