import argparse
import Reader
import Elasticnet
import sys
import Visualisations
import Metrics


def main():
    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    parser.add_argument("-head", help="Confirm if a header exists in the file", type=int, required=True)
    parser.add_argument("-output", help="Output file", type=str, required=True)
    args = parser.parse_args()

    #Read the data provided via the commandline
    read = Reader.Reader(args.f, args.head)
    data = read.reader()

    #Run elasticnet
    algorithm = Elasticnet.Elasticnet(data, "BMI")
    algorithm.extract_labels()
    X_train, X_test, y_train, y_test = algorithm.split_data()
    elastic_model = algorithm.train_model(X_train, y_train)
    predictions = algorithm.predict(elastic_model, X_test)
    model, cv = algorithm.define_model()
    scores = algorithm.evaluate_model(model, cv)

    metrics = Metrics.Metrics()
    r2 = metrics.r_squared(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    rmse = metrics.root_mean_squared_error(y_test, predictions)
    print("Input file: %s" % args.f)

    visuals = Visualisations.Visualisations()
    visuals.boxplot(scores)

if __name__ == "__main__":
    sys.exit(main())
