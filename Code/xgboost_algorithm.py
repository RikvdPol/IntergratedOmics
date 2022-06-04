
# First XGBoost model for Pima Indians dataset
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from Abstractalgorithm import Abstractalgorithm


class XG(Abstractalgorithm):
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        # fit model no training data
        model = XGBRegressor()
        return model, cv

    def train_model(self, clf, X_train, y_train):
        elastic_model = clf.fit(X_train, y_train)
        return elastic_model

