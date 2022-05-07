import pandas as pd
import os
import logging
import sys

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Reader:
    """
    Reads an input file and store the file in a pandas dataframe for later use.
    """

    def __init__(self, file, header):
        self.file = file
        self.header = header

    def reader(self):
        """"
        Read a file and print the first five rows.
        """
        sep = os.path.sep
        if not os.path.exists(f'..{sep}Logfiles'):
            os.makedirs(f'..{sep}Logfiles')
        try:
            data = pd.read_csv(self.file, sep="\t", header=self.header)
            logging.basicConfig(filename=f'..{sep}Logfiles{sep}reader.log',
                                filemode='a+',
                                format='%(asctime)s %(message)s',
                                force=True)
            logging.warning(f'Reading of {self.file} successful')
            return data
        except FileNotFoundError as e:
            logging.basicConfig(filename=f'..{sep}Logfiles{sep}reader.log',
                                filemode='a+',
                                format='%(asctime)s %(message)s',
                                force=True)
            logging.warning(f'{e}: Reading of {self.file} unsuccessful')
            sys.exit(0)


