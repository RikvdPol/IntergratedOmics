"Author: Hicham Jemil"

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
# from sklearn import model_selection, ensemble
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold, cross_val_score, train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import matplotlib.pyplot as plt
import shap
from IPython.display import display
from shap import maskers
from shap import summary_plot
from sklearn.linear_model import ElasticNet


class Shap():
    def __init__(self, X_train, model):
        self.X_train = X_train
        self.model = model
        
    def shap_test(self):
        "Provides the user both local and global shap values based on the Game theory"
        shap.initjs()
        X_sampled = self.X_train.sample(100, random_state=10)
        explainer = None

        if "ElasticNet" in str(type(self.model)):
            masker = maskers.Independent(data=self.X_train)
            explainer = shap.explainers.Linear(self.model, masker=masker)
        elif "XGBRegressor" in str(type(self.model)):
            explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(X_sampled)
        print("## Contributing features to diviate from the base value")
        print("Features in red contribute to a higher prediction")
        print("Features in blue contribute to a lower prediction")
        # fig1 = shap.force_plot(explainer.expected_value, shap_values[0, :], X_sampled.iloc[0, :], matplotlib=True, show=False)

        print("\n## Contributing effect of a single feature vs the model output")
        print("Shap values represents a feature's responsability for a change in a selected output")
        print("Vertical dispersion represents the interaction vs the other features")
        # shap.force_plot(explainer.expected_value, shap_values, self.X_train, matplotlib=True)

        print("\n## Mean absolute contribution for each feature")
        # fig2 = summary_plot(shap_values, X_sampled, show=False, plot_size=None)

        print("\n## Mean absolute contribution for each feature")
        # fig3 = summary_plot(shap_values, X_sampled, plot_type="bar", show=False, plot_size=None)


        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        summary_plot(shap_values, X_sampled, show=False, plot_size=None)
        plt.subplot(1,2,2)
        summary_plot(shap_values, X_sampled, plot_type="bar", show=False, plot_size=None)
        plt.tight_layout()
        plt.show()

def script_shapley(X_train, model):
    Shap(X_train, model).shap_test()
    


