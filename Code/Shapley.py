"Author: Hicham Jemil"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn import model_selection, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import shap



class Shap():
    def __init__(self, X_train, model):
        self.X_train = X_train
        self.model = model
        
    def shap_test(self):
        "Provides the user both local and global shap values based on the Game theory"
        shap.initjs()
        X_sampled = self.X_train.sample(100, random_state=10)  #in order to improve computing efficiency, a random v is chosen
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sampled)
        print("## Contributing features to diviate from the base value")
        print("Features in red contribute to a higher prediction")
        print("Features in blue contribute to a lower prediction")
        display(shap.force_plot(explainer.expected_value, shap_values[0,:], X_sampled.iloc[0,:]))
        
        print("\n## Contributing effect of a single feature vs the model output")
        print("Shap values represents a feature's responsability for a change in a selected output")
        print("Vertical dispersion represents the interaction vs the other features")
        display(shap.force_plot(explainer.expected_value, shap_values, X_train))
        
        print("\n## Mean absolute contribution for each feature")
        display(shap.summary_plot(shap_values, X_sampled))
        
        print("\n## Mean absolute contribution for each feature")
        display(shap.summary_plot(shap_values, X_sampled, plot_type="bar"))
        

def script_shap(X_train, model):
    Shap(X_train, model).shap_test()
    


