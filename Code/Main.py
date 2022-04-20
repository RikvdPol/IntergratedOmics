import argparse
import Reader
import ElasticNet




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    parser.add_argument("-head", help="Confirm if a header exists in the file", type=int, required=True)
    parser.add_argument("-output", help="Output file", type=str, required=True)
    args = parser.parse_args()
    read = Reader.Reader()
    data = read.reader(args.f)
    algorithm = ElasticNet.ElasticNet(data, "BMI")
    algorithm.extract_labels()
    print("Input file: %s" % args.f)
