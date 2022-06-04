import argparse
import Reader
import Elasticnet
import sys
import Visualisations
import Metrics
import Logging
import xgboost_algorithm
import LassoAlgorithm
# import Preprocessing
# import Recommendation


def main():
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    parser.add_argument("-head", help="Confirm if a header exists in the file", type=int, required=True)
    args = parser.parse_args()

    # Read the data provided via the commandline
    read = Reader.Reader(args.f, args.head)
    data = read.reader()


# TODO upper en lower inputs
# TODO let user define which labels are of interest

    algorithm = Elasticnet.Elasticnet(data, 'BMI')
    algorithm.extract_labels()
    X_train, X_test, y_train, y_test = algorithm.split_data()
    clf = algorithm.tune_hyperparameters()
    clf_model = algorithm.train_model(clf, X_train, y_train)
    predictions = algorithm.predict(clf_model, X_test)
    print(predictions - y_test)
    cv = algorithm.define_model()
    scores = algorithm.evaluate_model(clf_model, cv)
    metrics = Metrics.Metrics(y_test, predictions)
    r2 = metrics.r_squared()
    mse = metrics.mean_squared_error()
    mae = metrics.mean_absolute_error()
    rmse = metrics.root_mean_squared_error()
    print(scores)
    print("Input file: %s" % args.f)
            
    # algorithm = xgboost_algorithm.XG(data, 'BMI')
    # algorithm.extract_labels()
    # X_train, X_test, y_train, y_test = algorithm.split_data()
    # elastic_model = algorithm.train_model(X_train, y_train)
    # predictions = algorithm.predict(elastic_model, X_test)
    # model, cv = algorithm.define_model()
    # scores1 = algorithm.evaluate_model(model, cv)
    # metrics = Metrics.Metrics(y_test, predictions)
    # r2 = metrics.r_squared()
    # mse = metrics.mean_squared_error()
    # mae = metrics.mean_absolute_error()
    # rmse = metrics.root_mean_squared_error()
    # print("Input file: %s" % args.f)
    #
    # algorithm = LassoAlgorithm.LassoAlgorithm(data, 'BMI')
    # algorithm.extract_labels()
    # X_train, X_test, y_train, y_test = algorithm.split_data()
    # elastic_model = algorithm.train_model(X_train, y_train)
    # predictions = algorithm.predict(elastic_model, X_test)
    # model, cv = algorithm.define_model()
    # scores2 = algorithm.evaluate_model(model, cv)
    # metrics = Metrics.Metrics(y_test, predictions)
    # r2 = metrics.r_squared()
    # mse = metrics.mean_squared_error()
    # mae = metrics.mean_absolute_error()
    # rmse = metrics.root_mean_squared_error()
    # print("Trained Lasso")
    # print("Input file: %s" % args.f)
    #
    # scores2 = {"elasticnet": scores, "XG": scores1, "Lasso": scores2}
    # visuals = Visualisations.Visualisations(scores2, cv)
    # visuals.boxplot()


if __name__ == "__main__":
    sys.exit(main())

