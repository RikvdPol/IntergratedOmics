import matplotlib.pyplot as plt
import Logging
import sys
import os
from pathlib import Path

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Visualisations:
    """
    Plot visualisations based on the data provided.
    parameters:
        scores = dictionary containing the name of a machine learning model as a key and a list of metric scores as a value
        cv = crossfold validation values(generated in the algorithm class)
        metric = the metric used to calculate the scores dictionary
    """
    def __init__(self, scores, cv, metric="mean_squared_error"):
        self.scores = scores
        self.cv = cv
        self.metric = metric

    def boxplot(self):
        """
        Plot a boxplot for each machine learning algorithm in the scores dictionary
        """
        logs = Logging.Logging()
        try: 
            labels, data = self.scores.keys(), self.scores.values()
            msg = f"Algorithm name and data successfully extracted"
            logs.create_logs(self.__class__.__name__, msg)

        except AttributeError as e:
            msg = f"{e}: Data provided not in dictionary"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)

        if not os.path.exists("Visuals"):
            os.makedirs("Visuals")

        path = Path(f"Visuals/AlgorithmBoxplot.png")
        plt.title(f"""Comparison of RepeatedKFold with n_split={int(self.cv.get_n_splits() / self.cv.n_repeats)}
        and n_repeats={self.cv.n_repeats} for a total of {self.cv.get_n_splits()} metric results""")
        plt.boxplot(data)
        plt.ylabel(self.metric)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.savefig(path)
