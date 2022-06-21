import argparse
from copy import copy
import Reader
import Elasticnet
import sys
import Visualisations
import Metrics
import Logging
import xgboost_algorithm
import Gradient_boosting
import Preprocess
import pandas as pd
import Advice
import DimensionalityReduction
import Shapley


def main():


    # The Preprocess module requests a datafile. THis is for now only in a pickle extension. Run this snippet to store the combined dataset in a pickle format:
    df_ID = pd.read_csv('Data/Covariates.csv', index_col=0, sep="\t")   # Example file is used
    df_micro = pd.read_csv('Data/MetaPhlan3.csv', index_col=0, sep="\t")
    df_test = pd.concat([df_ID, df_micro], axis=1)
    df_test.drop("X1172", axis=1, inplace=True)
    df_test.dropna(inplace=True)
    df_test.to_pickle("microbiome_df.pkl")


    ## Snippet to run from Preprocess till advice:
    end_data, target, features, df_bins = Preprocess.script_preprocessing()   # Use microbiome_df.pkl as file
    #Target contains the label column, features contain the data
    # print(features)
    end_result = Advice.script_recommendation(end_data, features)

    scores_dict = {}
    models_dict = {'elasticnet': Elasticnet.Elasticnet, 'xgboost': xgboost_algorithm.XG,
                   'gradientboost': Gradient_boosting.Gradientboost}


    reductions = DimensionalityReduction.DimensionalityReduction(end_data, target)
    mod = reductions.construction()
    reductions.construct_scree_plot(mod)
    reductions.construct_bi_plot(mod)


    for model in models_dict:

        algorithm = models_dict[model](features, target)
        X_train, X_test, y_train, y_test = algorithm.split_data()
        clf, cv = algorithm.define_model()
        shap = Shapley.Shap(X_train, clf)
        shap.shap_test()
    #     elastic_model = algorithm.train_model(clf, X_train, y_train)
    #     predictions = algorithm.predict(elastic_model, X_test)

    #     scores = algorithm.evaluate_model(clf, cv)
    #     metrics = Metrics.Metrics(y_test, predictions)
    #     scores_dict[model] = scores

    # visuals = Visualisations.Visualisations(scores_dict, cv)
    # visuals.boxplot()

if __name__ == "__main__":
    sys.exit(main())
