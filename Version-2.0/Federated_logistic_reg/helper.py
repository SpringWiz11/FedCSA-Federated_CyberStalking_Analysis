# helper.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import List
from flwr.common import NDArrays

def load_dataset(client_id: int):
    df = pd.read_csv('/home/kishan/Documents/projects/machinelearning_cyberstalking_research/dataset.csv')
    df['text'] = df['text'].str.replace('[^a-zA-Z\s]', '').str.lower()
    df = df.drop(df.columns[2], axis=1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    np.random.seed(42)
    random_choose = np.random.choice(X.index, (len(X) % 2), replace=False)
    X = X.drop(random_choose)
    y = y.drop(random_choose)

    X_split, y_split = np.split(X, 2), np.split(y, 2)
    X1, y1 = X_split[0], y_split[0]
    X2, y2 = X_split[1], y_split[1]
    # X3, y3 = X_split[2], y_split[2]

    X_train, y_train, X_test, y_test = [], [], [], []
    train_size = 0.8

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=train_size, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=train_size, random_state=42)
    # X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size=train_size, random_state=42)

    X_train.append(X1_train)
    X_train.append(X2_train)
    # X_train.append(X3_train)

    y_train.append(y1_train)
    y_train.append(y2_train)
    # y_train.append(y3_train)

    X_test.append(X1_test)
    X_test.append(X2_test)
    # X_test.append(X3_test)

    y_test.append(y1_test)
    y_test.append(y2_test)
    # y_test.append(y3_test)

    return X_train[client_id], y_train[client_id], X_test[client_id], y_test[client_id]


def get_params(model: LogisticRegression) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params

def set_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Sets the parameters of a sklearn LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But the server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # churn dataset has 2 classes
    n_features = 1  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))