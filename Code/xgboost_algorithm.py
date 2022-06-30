from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from AlgorithmBaseClass import AlgorithmBaseClass
import sys
import Logging


class XG(AlgorithmBaseClass):
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        logs = Logging.Logging()
        try:
            clf = XGBRegressor()
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            msg = f"Model {type(clf)} definition succeeded"
            logs.create_logs(self.__class__.__name__, msg)
            return clf, cv

        except BaseException as e:
            msg = f"{e}: Model definition failed"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)
