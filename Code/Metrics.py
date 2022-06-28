from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import numpy as np

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class Metrics:
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
        r2 = r2_score(self.y_test, self.predictions)
        print(f"Model R2: {r2}")
        return r2

    def mean_squared_error(self):
        """
        Calculates the mean squared error. The mean squared error calculates the sum
        of all predictions squared, and than divided it by the number of observations.
        Returns a single value that represents model accuracy. The lower the mean squared
        error, the better the model performs.
        """
        mse = mean_squared_error(self.y_test, self.predictions)
        print(f"Model Mean Squared Error: {mse}")
        return mse

    def mean_absolute_error(self):
        """
        Calculates the mean absolute error. Rather than squaring the predictions in order
        to get rid of the negative sign, this method simply takes the absolute value and 
        calculates the mean. Returns a single value that represents model accuracy. The 
        lower the mean absolute error, the better the model performs.
        """
        mae = mean_absolute_error(self.y_test, self.predictions)
        print(f"Model Mean Absulute Error: {mae}")
        return mae

    def root_mean_squared_error(self):
        """
        Calculates the root mean squared error. This metric is very similar to the mean squared error.
        It consists of only one additional step. In order to get rid of the squaring of the data, the result
        of the root of the mean squared error is used.
        """
        rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        print(f"Model Root Mean Squared Error: {rmse}")
        return rmse

    def repeatedKfold(self, model, cv):
        """
        scores = cross_val_score(model, self.file, self.labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        return scores
        """

    def get_scores(self):
        r2 = self.r_squared()
        mse = self.mean_squared_error()
        mae = self.mean_absolute_error()
        rmse = self.root_mean_squared_error()
        return r2, mse, mae, rmse
