import shap
import pandas as pd
from xgboost import XGBRegressor

df = pd.read_csv('Covariates.csv', sep='\t')
X = df[['Age', 'Gender']]
y = df[['BMI']]


# train an XGBoost model
# X, y = shap.datasets.boston()
model = XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])



# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])


# visualize all the training set predictions
shap.plots.force(shap_values)

class Shap:
    def __init__(self, fit_model, X) -> None:
        self.model = fit_model
        self.X = X
        self.explainer = shap.Explainer(model)
        self.shap_values = explainer(X)

    def plot_beeswarm():
        shap.plots.beeswarm(shap_values)

    def plot_bar():
        shap.plots.bar(shap_values)


# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:,"RM"], color=shap_values)


# summarize the effects of all the features
shap.plots.beeswarm(shap_values)


shap.plots.bar(shap_values)