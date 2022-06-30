import sys
import Logging
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from AlgorithmBaseClass import AlgorithmBaseClass

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Elasticnet(AlgorithmBaseClass):
    """
    Performs the elasticnet machine learning algorithm. Inherits from the AlgorithmBaseClass.
    """
    def define_model(self, alpha=1.0, l1_ratio=0.5, n_splits=10, n_repeats=3, random_state=None):
        """
        Tries to define an ElasticNet model. Other functionality is inherited from the base class.
        """
        logs = Logging.Logging()
        try:
            clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            msg = f"Model {clf} definition succeeded"
            logs.create_logs(self.__class__.__name__, msg)
            return clf, cv

        except BaseException as e:
            msg = f"{e}: Model definition failed"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)
