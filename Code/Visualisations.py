import matplotlib.pyplot as plt

class Visualisations:
    def __init__(self):
        pass

    def boxplot(self, data):
        plt.boxplot(data)
        plt.show()
        # fig1, ax1 = plt.subplots()
        # ax1.set_title('Algorithm Scores')
        # ax1.boxplot(data)
        # ax1.show()
