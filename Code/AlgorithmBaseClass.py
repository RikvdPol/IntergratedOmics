from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np


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
        self.labels = None

    def split_data(self, test_size=0.3, random_state=None):
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
        X_train, X_test, y_train, y_test = train_test_split(self.file,
                                                            self.labels,
                                                            test_size=test_size,
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test

    def predict(self, model, X_test):
        """
        Makes predictions on the test dataset by using the trained model.
        """
        predictions = model.predict(X_test)
        return predictions

    def evaluate_model(self, model, cv):
        """
        Evaluate the model using the cross_val_score function. The model is evaluated several times and the results of the
        cost function are saved. This will later be used for the plotting of the boxplots.
        """
        scores = cross_val_score(model, self.file, self.labels, scoring=make_scorer(mean_absolute_error), cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        return scores

    def define_model(self, alpha=1.0, n_splits=10, n_repeats=3, random_state=None):
        """
        Define a machine learning algorithm. Only this method needs to be changed in the algorithm class provided
        it inherits this class.
        """
        pass

    def train_model(self, clf, X_train, y_train):
        """
        Train the previously defined model by providing it with training data and the corresponding labels.
        """
        model = clf.fit(X_train, y_train)
        return model


        