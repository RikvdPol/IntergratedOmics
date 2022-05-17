import pandas as pd
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class test_PCA:
    def __init__(self):
        # self.data = pd.read_csv("../Data/Covariates.csv", sep="\t", header=0)
        # self.labels = self.data["BMI"]
        # self.file = self.data.drop(["BMI", "Pseudo", "Antibody_batch"], axis=1)
        self.df = None
    
    def load_data(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        # load dataset into Pandas DataFrame
        self.df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
        print(self.df)


    def standardise(self):
        features = ['sepal length', 'sepal width', 'petal length', 'petal width']
        # Separating out the features
        x = self.df.loc[:, features].values
        # Separating out the target
        self.y = self.df.loc[:,['target']].values
        # Standardizing the features
        self.x = StandardScaler().fit_transform(x)
        # print(x)

    def projection(self):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.x)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, self.df[['target']]], axis = 1)
        self.visuals(finalDf)


    def visuals(self, finalDf):
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , s = 50)
        ax.legend(targets)
        ax.grid()
        fig.savefig("test.png")


def main():
    pca = test_PCA()
    pca.load_data()
    pca.standardise()
    pca.projection()



if __name__ == "__main__":
    sys.exit(main())


    

