"""Author: Hicham Jemil"""

import matplotlib.pyplot as plt
import shap
from shap import maskers
from shap import summary_plot


class Shap:
    def __init__(self, x_train, model):
        self.X_train = x_train
        self.model = model
        
    def shap_test(self):
        """Provides the user both local and global shap values based on the Game theory"""
        shap.initjs()
        x_sampled = self.X_train.sample(100, random_state=10)
        explainer = None

        if "ElasticNet" in str(type(self.model)):
            masker = maskers.Independent(data=self.X_train)
            explainer = shap.explainers.Linear(self.model, masker=masker)
        elif "XGBRegressor" in str(type(self.model)):
            explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(x_sampled)
        print("## Contributing features to deviate from the base value")
        print("Features in red contribute to a higher prediction")
        print("Features in blue contribute to a lower prediction")
        shap.force_plot(explainer.expected_value, shap_values[0, :], x_sampled.iloc[0, :], matplotlib=True, show=False)

        print("\n## Contributing effect of a single feature vs the model output")
        print("Shap values represents a feature's responsibility for a change in a selected output")
        print("Vertical dispersion represents the interaction vs the other features")

        print("\n## Mean absolute contribution for each feature")

        print("\n## Mean absolute contribution for each feature")

        plt.figure(figsize=(25, 30))
        plt.subplot(1, 2, 1)
        summary_plot(shap_values, x_sampled, show=False, plot_size=None)
        plt.subplot(1, 2, 2)
        summary_plot(shap_values, x_sampled, plot_type="bar", show=False, plot_size=None)
        plt.tight_layout()
        plt.show()


def script_shapley(x_train, model):
    Shap(x_train, model).shap_test()
    


