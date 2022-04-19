import argparse
import Reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    parser.add_argument("-head", help="Confirm if a header exists in the file", type=int, required=True)
    parser.add_argument("-output", help="Output file", type=str, required=True)
    args = parser.parse_args()
    read = Reader.Reader()
    read.reader(args.f)
    print("Input file: %s" % args.f)
