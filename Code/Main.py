import Elasticnet
import sys
import Visualisations
import Metrics
import xgboost_algorithm
import Preprocess
import Advice
import DimensionalityReduction
import Shapley


def main():
    ## Snippet to run from Preprocess till advice:
    end_data, target, features, df_bins = Preprocess.script_preprocessing()
    end_result = Advice.script_recommendation(end_data, features)

    scores_dict = {}
    models_dict = {'elasticnet': Elasticnet.Elasticnet, 'xgboost': xgboost_algorithm.XG}



    reductions = DimensionalityReduction.DimensionalityReduction(end_data, target)
    mod = reductions.construction()
    reductions.construct_scree_plot(mod)
    reductions.construct_bi_plot(mod)


    for model in models_dict:

        algorithm = models_dict[model](features, target)
        X_train, X_test, y_train, y_test = algorithm.split_data()
        clf, cv = algorithm.define_model()
        clf = algorithm.train_model(clf, X_train, y_train)
        shap = Shapley.Shap(X_train, clf)
        shap.shap_test()
        # predictions = algorithm.predict(clf, X_test)

        # scores = algorithm.evaluate_model(clf, cv)
        # metrics = Metrics.Metrics(y_test, predictions)
        # scores_dict[model] = scores

    # visuals = Visualisations.Visualisations(scores_dict, cv)
    # visuals.boxplot()


if __name__ == "__main__":
    sys.exit(main())
