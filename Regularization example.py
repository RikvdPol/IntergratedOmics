

''''
Regularization is a technique to combat the overfitting issue in machine learning. 
Overfitting, also known as High Variance, refers to a model that learns the training data too well but fail to generalize to new data.


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()

# Load data into a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Convert datatype to float
df = df.astype(float)
# append "target" and name it "label"
df['label'] = iris.target
# Use string label instead
df['label'] = df.label.replace(dict(enumerate(iris.target_names)))


# label -> one-hot encoding
label = pd.get_dummies(df['label'], prefix='label')
df = pd.concat([df, label], axis=1)
# drop old label
df.drop(['label'], axis=1, inplace=True)


# Creating X and y
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
# Convert DataFrame into np array
X = np.asarray(X)
y = df[['label_setosa', 'label_versicolor', 'label_virginica']]
# Convert DataFrame into np array
y = np.asarray(y)


# '''split the dataset into a training set (80%)and a test set (20%) using train_test_split() from sklearn library.

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.20
)

# Build an unregularized neural network model

#create a function called create_model() to return a Sequential model.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def create_model(): 
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model



model = create_model()
model.summary()


#Training a model
#Use Adam (adam) optimization algorithm as the optimizer
#Use categorical cross-entropy loss function (categorical_crossentropy) for our multiple-class classification problem
#For simplicity, use accuracy as our evaluation metrics to evaluate the model during training and testing.


model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)


#After that, we can call model.fit() to fit our model to the training data

history = model.fit(
    X_train, 
    y_train, 
    epochs=200, 
    validation_split=0.25, 
    batch_size=40, 
    verbose=2
)



#Model Evaluation:
#Plot the progress on loss and accuracy metrics
#Test our model against data that has never been used for training. 
#This is where the test dataset X_test that we set aside earlier come to play.

create a function plot_metric() for plotting metrics.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


# #By running plot_metric(history, 'accuracy') to plot the progress on accuracy.
# #By running plot_metric(history, 'loss') to plot the progress on loss.



#To evaluate the model on the test set
# Evaluate the model on the test set
model.evaluate(X_test, y_test, verbose=2)


#Adding L2 regularization and Dropout
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2


#Then, we create a function called create_regularized_model() and it will return a model similar to the one we built before. But, this time we will add L2 regularization and Dropout layers, so this function takes 2 arguments: a L2 regularization factor and a Dropout rate.

#just add L2 regularization in all layers except the output layer .
#also add Dropout layer between every two dense layers.
def create_regularized_model(factor, rate):
    model = Sequential([
        Dense(64, kernel_regularizer=l2(factor), activation="relu", input_shape=(4,)),
        Dropout(rate),
        Dense(128, kernel_regularizer=l2(factor), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=l2(factor), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=l2(factor), activation="relu"),
        Dropout(rate),
        Dense(64, kernel_regularizer=l2(factor), activation="relu"),
        Dropout(rate),
        Dense(64, kernel_regularizer=l2(factor), activation="relu"),
        Dropout(rate),
        Dense(64, kernel_regularizer=l2(factor), activation="relu"),
        Dropout(rate),
        Dense(3, activation='softmax')
    ])
    return model

model = create_regularized_model(1e-5, 0.3)
model.summary()

#Training
# First configure model using model.compile()
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
# Then, train the model with fit()
history = model.fit(
    X_train, 
    y_train, 
    epochs=200, 
    validation_split=0.25, 
    batch_size=40, 
    verbose=2
)

# Model Evaluation
plot_metric(history, 'loss')
#to evaluate the model on the test set
model.evaluate(X_test, y_test, verbose=2)

