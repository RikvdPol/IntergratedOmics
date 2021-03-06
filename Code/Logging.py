import logging
import os
from pathlib import Path

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Logging:
    """
    Allows for logging. Using this class in each module, it becomes possible to log the events happening during
    the run of the pipeline.
    """

    def create_logs(self, classname, msg):
        """
        Creates the logiles of each module in the Logfiles folder.
        """
        if not os.path.exists(f'Logfiles'):
            os.makedirs(f'Logfiles')

        path = Path(f"Logfiles/{classname}.log")
        logging.basicConfig(filename=path,
                            filemode='a+',
                            format='%(asctime)s %(message)s',
                            force=True)
        logging.warning(msg)
