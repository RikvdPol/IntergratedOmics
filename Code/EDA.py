import pandas as pd
import numpy as np

cova = pd.read_csv("Covariates.csv", sep="\t")
meta = pd.read_csv("MetaPhlan3.csv", sep="\t")
link = pd.read_csv("Olink.csv", sep="\t")
phip = pd.read_csv("PhipSeq.csv", sep="\t")
plasma = pd.read_csv("PlasmaMetabolomics.csv", sep="\t")


class EDA:
    def reader(self):
        self.cova = pd.read_csv("Covariates.csv", sep="\t")
        self.meta = pd.read_csv("MetaPhlan3.csv", sep="\t")
        self.link = pd.read_csv("Olink.csv", sep="\t")
        self.phip = pd.read_csv("PhipSeq.csv", sep="\t")
        self.plasma = pd.read_csv("PlasmaMetabolomics.csv", sep="\t")

    def dimensions(self):
        print("Covariates rows and columns: \n", self.cova.shape, "\n")

        print("MetaPhlan rows and columns: \n ", self.meta.shape, "\n")

        print("Olink rows and columns: \n", self.link.shape, "\n")

        print("PhipSeq rows and columns: \n", self.phip.shape, "\n")

        print("PlasmaMetabolomic rows and columns: \n", self.plasma.shape, "\n")

    def get_covariates(self):
        print(self.cova)

    def get_meta(self):
        print(self.meta)

    def get_link(self):
        print(self.link)

    def get_phip(self):
        print(self.phip)

    def get_plasma(self):
        print(self.plasma)




eda = EDA()
eda.reader()
# eda.dimensions()

eda.get_covariates()
# eda.get_meta()
# eda.get_link()
# eda.get_phip()
# eda.get_plasma()