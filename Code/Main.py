import sys
import Elasticnet
import xgboost_algorithm
import Preprocess
import Advice
import DimensionalityReduction
import Shapley
import Visualisations

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


def main():
    ## Snippet to run from Preprocess till advice:
    end_data, target, features, df_bins = Preprocess.script_preprocessing()
    end_result = Advice.script_recommendation(end_data, features)

    # Prepare the scores dictionary. Will later be used to store and plot the metric scores.
    scores_dict = {}

    models_dict = {'elasticnet': Elasticnet.Elasticnet, 'xgboost': xgboost_algorithm.XG}
    # Perform principal component analysis
    reductions = DimensionalityReduction.DimensionalityReduction(end_data, target)
    mod = reductions.construction()
    reductions.construct_scree_plot(mod)
    reductions.construct_bi_plot(mod)

    # Run the implemented algorithms and the SHAP module for each algorithm
    for model in models_dict:
        algorithm = models_dict[model](features, target)
        # gather the training and testing data
        x_train, x_test, y_train, y_test = algorithm.split_data()
        # Gather the defined classifier and crossfold validation objects
        clf, cv = algorithm.define_model()
        clf = algorithm.train_model(clf, x_train, y_train)
        shap = Shapley.Shap(x_train, clf)
        shap.shap_test()
        scores = algorithm.evaluate_model(clf, cv)
        scores_dict[model] = scores

    # Run the visual module. cv will always be the same, so it does not matter it contains the last one from the loop
    visuals = Visualisations.Visualisations(scores_dict, cv)
    visuals.boxplot()


if __name__ == "__main__":
    sys.exit(main())
