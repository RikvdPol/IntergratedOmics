from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import numpy as np

class Metrics:
    def __init__(self):
        pass

    def r_squared(self, y_test, predictions):
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
        r2 = r2_score(y_test,predictions)
        print(f"Model R2: {r2}")

    def mean_squared_error(self, y_test, predictions):
        mse = mean_squared_error(y_test,predictions)
        print(f"Model Mean Squared Error: {mse}")
        return mse

    def mean_absolute_error(self, y_test, predictions):
        mae = mean_absolute_error(y_test, predictions)
        print(f"Model Mean Absulute Error: {mae}")
        return mae

    
    def root_mean_squared_error(self, y_test, predictions):
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Model Root Mean Squared Error: {rmse}")
        return rmse