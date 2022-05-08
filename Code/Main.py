import argparse
import Reader
import Elasticnet
import xgboost_algorithm
import sys


def main():
    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    parser.add_argument("-head", help="Confirm if a header exists in the file", type=int, required=True)
    parser.add_argument("-ML_type", help="Output file", type=str, required=True)
    args = parser.parse_args()

    #Read the data provided via the commandline
    read = Reader.Reader()
    data = read.reader(args.f)

#TODO upper en lower inputs
#TODO let user define which labels are of interest

    if args.ML_type == "Elasticnet":
        try: 
            algorithm = Elasticnet.Elasticnet(data, 'BMI')
        except:
            print("Error: Elasticnet failed")
            sys.exit(1)

    elif args.ML_type == "XG":
        try:
            pass
        except:
            print("Error: XG failed")
            sys.exit(1)

        #algorithm = xgboost_algorithm.XG(data, 'BMI')
        #algorithm.XG_boost(data, 'BMI')

    elif args.ML_type == "etcetera":
        print("etcetera")
    
    else:
        print("Please specify a valid ML type")     #raise exception?
        sys.exit()



    #Run elasticnet
    '''
    algorithm = Elasticnet.Elasticnet(data, "BMI")
    algorithm.extract_labels()
    model, cv = algorithm.define_model()
    algorithm.evaluate_model(model, cv)
    print("Input file: %s" % args.f)
    '''


if __name__ == "__main__":
    sys.exit(main())
