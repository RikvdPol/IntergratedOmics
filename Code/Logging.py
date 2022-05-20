import logging
import os
from pathlib import Path


__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Logging:
    def __int__(self):
        # self.sep = os.path.sep
        # self.sep = "/"
        pass


    
    def create_logs(self, classname, msg):
        # sep = os.path.sep
        path = Path(f"../Logfiles/{classname}.log")
        logging.basicConfig(filename=path,
                                filemode='a+',
                                format='%(asctime)s %(message)s',
                                force=True)
        logging.warning(msg)