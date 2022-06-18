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


def main():


    # The Preprocess module requests a datafile. THis is for now only in a pickle extension. Run this snippet to store the combined dataset in a pickle format:
    df_ID = pd.read_csv('../Data/Covariates.csv', index_col=0, sep="\t")   # Example file is used
    df_micro = pd.read_csv('../Data/MetaPhlan3.csv', index_col=0, sep="\t")
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
        elastic_model = algorithm.train_model(clf, X_train, y_train)
        predictions = algorithm.predict(elastic_model, X_test)

        scores = algorithm.evaluate_model(clf, cv)
        metrics = Metrics.Metrics(y_test, predictions)
        scores_dict[model] = scores

    visuals = Visualisations.Visualisations(scores_dict, cv)
    visuals.boxplot()

    # Parse commandline arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-f", help="File to be read", type=str, required=True)
    # args = parser.parse_args()
    # models_dict = {'elasticnet': Elasticnet.Elasticnet, 'xgboost': xgboost_algorithm.XG, 'gradientboost': Gradient_boosting.Gradientboost}
    # scores_dict = {}
    # Read the data provided via the commandline
    # read = Reader.Reader(args.f, 0)
    # data = read.reader()
    # end_data, target, features, df_bins = Preprocessing.script_preprocessing(df_test)
    #
    # for model in models_dict:
    #     print(models_dict[model])
    #     algorithm = models_dict[model](end_data, target)
    #     X_train, X_test, y_train, y_test = algorithm.split_data()
    #     algorithm.extract_labels()
    #     elastic_model = algorithm.train_model(X_train, y_train)
    #     predictions = algorithm.predict(elastic_model, X_test)
    #     model, cv = algorithm.define_model()
    #     scores = algorithm.evaluate_model(model, cv)
    #
    #     metrics = Metrics.Metrics(y_test, predictions)
    #     r2, mse, mae, rmse = metrics.get_scores()
    #     print("Input file: %s" % args.f)
    #
    #     scores_dict[model] = scores
    #
    # visuals = Visualisations.Visualisations(scores_dict, cv)
    # visuals.boxplot()


if __name__ == "__main__":
    sys.exit(main())
