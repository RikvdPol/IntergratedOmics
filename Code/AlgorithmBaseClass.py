from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import Logging
import sys

__author__ = "Rik van de Pol"
__license__ = "MIT"
__email__ = "rikvdpol93@gmail.com"
__status__ = "Version 1.0"


class AlgorithmBaseClass:
    """
    Forms the base class for all algorithms. All algorithms have very similar functionality. Therefore, the base class
    has been constructed. The algorithm inherits this class and only the define_algorithm method needs to be changed.
    In the future, new algorithms can be added by simply inheriting this class and defining a new method in the
    define_model method of the newly created algorithm class.
    """
    def __init__(self, file, labelname):
        self.file = file
        self.labelname = labelname

    def split_data(self, test_size=0.3, random_state=42):
        """
        Split the data intro training and test data, and the labels intro training and test labels.
        The split is decided by the percentage of data to be used for testing, provided by a number between
        0 and 1. By defeault this number is set to 0.3, which means 30% of the data provided will be used
        for testing, and 70% will be used for training. In addition, the split function expects a random state.
        Providing a number will result in a specific random state, meaning that the same data is used for training
        and testing every single time if a number is provided, otherwise this will be random everytime. 
        This is random because is should not matter to the model which data it gets, it should always be comparable
        to previous instances.
        """
        logs = Logging.Logging()
        try:
            x_train, x_test, y_train, y_test = train_test_split(self.file,
                                                                self.labelname,
                                                                test_size=test_size,
                                                                random_state=random_state)

            msg = f"Data succesfully split into training and test data"
            logs.create_logs(self.__class__.__name__, msg)
            return x_train, x_test, y_train, y_test
    
        except BaseException as e:
            msg = f"{e}: splitting of data unsuccessful"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)

    def predict(self, model, x_test):
        """
        Makes predictions on the test dataset by using the trained model.
        """
        logs = Logging.Logging()
        try:
            predictions = model.predict(x_test)
            msg = f"Predictions could not be made"
            logs.create_logs(self.__class__.__name__, msg)
            return predictions

        except BaseException as e:
            msg = f"{e}: Could not make predictions"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)

    def evaluate_model(self, model, cv):
        """
        Evaluate the model using the cross_val_score function. The model is evaluated several times and the results of the
        cost function are saved. This will later be used for the plotting of the boxplots.
        """
        scores = cross_val_score(model, self.file, self.labelname, scoring=make_scorer(mean_absolute_error), cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        return scores

    def define_model(self, alpha=1.0, n_splits=10, n_repeats=3, random_state=None):
        """
        Define a machine learning algorithm. Only this method needs to be changed in the algorithm class provided
        it inherits this class.
        """
        pass

    def train_model(self, clf, x_train, y_train):
        """
        Train the previously defined model by providing it with training data and the corresponding labels.
        """
        model = clf.fit(x_train, y_train)
        return model


        