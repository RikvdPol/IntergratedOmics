
# First XGBoost model for Pima Indians dataset
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data

dataset = pd.read_csv('Code/Data/Covariates.csv', delimiter="\s+")
# split data into X and y
print(dataset.BMI)

dataset.dropna(inplace = True)
X = dataset[['Age', 'Gender']]
Y = dataset['BMI']

Y = Y.astype(int)

X = X.to_numpy()
Y = Y.to_numpy()

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBRegressor()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
