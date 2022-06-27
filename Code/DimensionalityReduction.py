import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pca import pca
import Logging
import sys

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class DimensionalityReduction:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = None

    def construction(self, components=10):
        logs = Logging.Logging()
        try:
            model = pca(n_components=components, normalize=True)
            self.results = model.fit_transform(self.X)
            msg = (f"PCA succesfully performed")
            logs.create_logs(self.__class__.__name__, msg)
            return model
        except BaseException as e:
            msg = (f"Principal Component Analysis could not be performed because of error {e}")
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)


    def construct_scree_plot(self, model):
        logs = Logging.Logging()
        try:
            model.plot(figsize=(10, 8))
            msg = (f"Scree plot successfully created")
            logs.create_logs(self.__class__.__name__, msg)
        except BaseException as e:
            msg = (f"Scree plot could be created because of error {e}")
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)
            

    def construct_bi_plot(self, model):
        logs = Logging.Logging()
        try:
            model.biplot(n_feat=3, legend=False, figsize=(10, 8), label=True)
            msg = (f"Biplot successfully created")
            logs.create_logs(self.__class__.__name__, msg)
        except BaseException as e:
            msg = (f"Biplot could be created because of error {e}")
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)

    def get_loadings(self):
        return self.results["loadings"]

    def get_pcs(self):
        return self.results["PC"]
