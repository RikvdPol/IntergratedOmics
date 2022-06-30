from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import Logging

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Metrics:
    """
    Perform different kinds of metrics used to identify the accuracy of a machine learning algorithm.
    parameters:
        y_test: Array of true labels
        predictions: Array of labels predicted by the machine learning model
    """
    def __init__(self, y_test, predictions):
        self.y_test = y_test
        self.predictions = predictions

    def r_squared(self):
        """
        R-squared is a statistical measure that represents the 
        proportion of the variance for a dependent variable thats explained 
        by an independent variable or variables in a regression model. 
        Whereas correlation explains the strength of the relationship between an 
        independent and dependent variable, R-squared explains to what extent the 
        variance of one variable explains the variance of the second variable. 
        So, if the R2 of a model is 0.50, then approximately half of the observed 
        variation can be explained by the models inputs.
        """

        logs = Logging.Logging()
        try:
            r2 = r2_score(self.y_test, self.predictions)
            msg = f"Calculated model R2 value"
            logs.create_logs(self.__class__.__name__, msg)
            return r2

        except BaseException as e:
            msg = f"{e}: R2 value could not be calculated"
            logs.create_logs(self.__class__.__name__, msg)

    def mean_squared_error(self):
        """
        Calculates the mean squared error. The mean squared error calculates the sum
        of all predictions squared, and than divided it by the number of observations.
        Returns a single value that represents model accuracy. The lower the mean squared
        error, the better the model performs.
        """

        logs = Logging.Logging()
        try:
            mse = mean_squared_error(self.y_test, self.predictions)
            msg = f"Calculated model mse value"
            logs.create_logs(self.__class__.__name__, msg)
            return mse

        except BaseException as e:
            msg = f"{e}: mse value could not be calculated"
            logs.create_logs(self.__class__.__name__, msg)

    def mean_absolute_error(self):
        """
        Calculates the mean absolute error. Rather than squaring the predictions in order
        to get rid of the negative sign, this method simply takes the absolute value and 
        calculates the mean. Returns a single value that represents model accuracy. The 
        lower the mean absolute error, the better the model performs.
        """
        logs = Logging.Logging()
        try:
            mae = mean_absolute_error(self.y_test, self.predictions)
            msg = f"Calculated model mae value"
            logs.create_logs(self.__class__.__name__, msg)
            return mae

        except BaseException as e:
            msg = f"{e}: mse value could not be calculated"
            logs.create_logs(self.__class__.__name__, msg)

    def root_mean_squared_error(self):
        """
        Calculates the root mean squared error. This metric is very similar to the mean squared error.
        It consists of only one additional step. In order to get rid of the squaring of the data, the result
        of the root of the mean squared error is used.
        """
        logs = Logging.Logging()
        try:
            rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
            msg = f"Calculated model rmse value"
            logs.create_logs(self.__class__.__name__, msg)
            return rmse

        except BaseException as e:
            msg = f"{e}: mse value could not be calculated"
            logs.create_logs(self.__class__.__name__, msg)



