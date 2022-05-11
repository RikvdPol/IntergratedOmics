import matplotlib.pyplot as plt

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"

class Visualisations:
    def __init__(self, data):
        self.data = data

    def boxplot(self): 
        labels, data = self.data.keys(), self.data.values()

        plt.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.savefig("../Visuals/Sampleboxplot.png")

