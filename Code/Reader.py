import pandas as pd
import numpy as np
import argparse
import logging

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "WIP"


class Reader:
    """"
    Simple reader class that reads a file.
    """
    def __init__(self):
        pass

    def reader(self, file):
        """"
        Read a file and store in variable.
        """
        with open(file) as f:
            data = f.read()

        for line in data:
            print(line)

    def test(self):
        print("Hoi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to be read", type=str, required=True)
    args = parser.parse_args()
    read = Reader()
    read.reader(args.f)
    read.test()
    print("Input file: %s" % args.f)
