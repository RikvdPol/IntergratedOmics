import logging
from pathlib import Path

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Logging:
    def __int__(self):
        pass

    def create_logs(self, classname, msg):
        path = Path(f"Logfiles/{classname}.log")
        print(msg)
        logging.basicConfig(filename=path,
                            filemode='a+',
                            format='%(asctime)s %(message)s',
                            force=True)
        logging.warning(msg)
