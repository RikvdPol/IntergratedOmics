
   
import argparse
from copy import copy
import Reader
import Elasticnet
import sys
import Visualisations
import Metrics
import Logging
import xgboost_algorithm
from sklearn.ensemble import RandomForestRegressor
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
    X_train, X_test, y_train, y_test = rf.split_data()

    
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    metrics = Metrics.Metrics(y_test, predictions)
    r2, mse, mae, rmse = metrics.get_scores()
    print("Input file: %s" % args.f)
    

visuals = Visualisations.Visualisations(scores_dict, cv)
visuals.boxplot()
            
if _name_ == "_main_":
    sys.exit(main())