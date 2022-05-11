import pandas as pd
import os
import Logging
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

        logs = Logging.Logging()
        try:
            data = pd.read_csv(self.file, sep="\t", header=self.header)
            msg = (f'Reading of {self.file} successful')
            logs.create_logs(self.__class__.__name__, msg)

            return data
    
        except FileNotFoundError as e:
            msg = f"{e}: Reading of {self.file} unsuccessful"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)


