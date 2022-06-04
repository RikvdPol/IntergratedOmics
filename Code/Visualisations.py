import matplotlib.pyplot as plt
import Logging
# import plotly.express as px
import sys

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Visualisations:
    def __init__(self, data, cv, metric="mean_squared_error"):
        self.data = data
        self.cv = cv
        self.metric = metric

    def boxplot(self):
        logs = Logging.Logging()
        try: 
            labels, data = self.data.keys(), self.data.values()
            msg = f"Algorithm name and data succesfully extracted"
            logs.create_logs(self.__class__.__name__, msg)

        except AttributeError as e:
            msg = f"{e}: Data provided not in dicionary"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)

        plt.title(f"""Comparisson of RepeatedKFold with n_split={int(self.cv.get_n_splits() / self.cv.n_repeats)}
        and n_repeats={self.cv.n_repeats} for a total of {self.cv.get_n_splits()} metric results""")
        plt.boxplot(data)
        plt.ylabel(self.metric)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.savefig("../Visuals/Sampleboxplot.png")
