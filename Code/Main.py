import argparse
from copy import copy
import Reader
import Elasticnet
import sys
import Visualisations
import Metrics
import Logging
import xgboost_algorithm
# import Preprocessing
# import Recommendation

models_dict = {'elasticnet': Elasticnet.Elasticnet, 'xgboost': xgboost_algorithm.XG}
scores_dict = {}

def main():
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    parser.add_argument("-head", help="Confirm if a header exists in the file", type=int, required=True)
    parser.add_argument("-ML_type", help="Output file", type=str, required=True)
    args = parser.parse_args()

    # Read the data provided via the commandline
    read = Reader.Reader(args.f, args.head)
    data = read.reader()
    X_train, X_test, y_train, y_test = algorithm.split_data()

    for model in models_dict:
        print(models_dict[model])
        algorithm = models_dict[model](data, 'BMI')
        algorithm.extract_labels()
        elastic_model = algorithm.train_model(X_train, y_train)
        predictions = algorithm.predict(elastic_model, X_test)
        model, cv = algorithm.define_model()
        scores = algorithm.evaluate_model(model, cv)
        
        metrics = Metrics.Metrics(y_test, predictions)
        r2 = metrics.r_squared()
        mse = metrics.mean_squared_error()
        mae = metrics.mean_absolute_error()
        rmse = metrics.root_mean_squared_error()
        print("Input file: %s" % args.f)
        
        scores_dict[model] = scores

visuals = Visualisations.Visualisations(scores_dict, cv)
visuals.boxplot()
            
if __name__ == "__main__":
    sys.exit(main())
