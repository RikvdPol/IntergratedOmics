from pca import pca
import Logging
import sys

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class DimensionalityReduction:
    """
    Perform Principal Component Analysis on the provided data. Plots the scree and biplot. The loadings
    and principal components can be gathered by using their respective get functions.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = None

    def construction(self, components=10):
        """
        Set up the construction of the principal component analysis. No plotting is performed.
        """
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
        """
        Construct the scree plot of the provided data. The scree plot shows how many principal components
        are needed to explain variance in the data. The cumulative explained variance is shows as a black line.
        """
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
        """
        Construct the biplot of the principal component analysis. The biplot is a combination of the loadings and the scoring plot.
        Loadings show the importance of each feature in the component, and the scores is the point where that observation project onto
        the direction.
        """
        logs = Logging.Logging()
        try:
            model.biplot(n_feat=3, legend=False, figsize=(10, 8), label=False, alpha_transparency=0.15, cmap="Reds")
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
