import argparse
import Reader
import Elasticnet
import sys
import Visualisations


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
    # model, cv = algorithm.define_model()
    # elastic_model = algorithm.evaluate_model(model, cv)
    algorithm.predict(elastic_model, X_test, y_test)
    print("Input file: %s" % args.f)


if __name__ == "__main__":
    sys.exit(main())
