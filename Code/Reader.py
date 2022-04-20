import pandas as pd
import os
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
        Read a file and print the first five rows.
        """
        sep = os.path.sep
        try:
            data = pd.read_csv(file, sep="\t")
            logging.basicConfig(filename=f'..{sep}Logfiles{sep}reader.log', filemode='a+',
                                format='%(asctime)s %(message)s')
            logging.warning(f'Reading of {file} successful')
            return data
        except FileNotFoundError as e:
            logging.basicConfig(filename=f'..{sep}Logfiles{sep}reader.log', filemode='a+',
                                format='%(asctime)s %(message)s')
            logging.warning(f'{e}: Reading of {file} unsuccessful')
