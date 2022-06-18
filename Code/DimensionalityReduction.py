import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pca import pca


class DimensionalityReduction:
    def __init__(self, X, y):
        # self.data = data
        self.X = X
        self.y = y
        self.results = None

    # def preparation(self, label="Gender"):
    #     data = self.data.dropna()
    #     self.y = data[label]
    #     self.X = data.drop(columns=["Pseudo", "Antibody_batch"])

    def construction(self, components=10):
        model = pca(n_components=components, normalize=True)
        self.results = model.fit_transform(self.X)
        return model

    def construct_scree_plot(self, model):
        model.plot(figsize=(10, 8))

    def construct_bi_plot(self, model):
        model.biplot(n_feat=3, legend=True, figsize=(10, 8), y=self.y, label=True)

    def get_loadings(self):
        return self.results["loadings"]

    def get_pcs(self):
        return self.results["PC"]
