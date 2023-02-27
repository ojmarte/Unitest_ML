import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import sys
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from machine_learning import (train_test_and_evaluate_model, tune_logistic_regression)

def test_train_test_and_evaluate_model():
    # Load example dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Test train_test_and_evaluate_model function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val = train_test_and_evaluate_model(
            ML_lib='sklearn.linear_model',
            package_name=None,
            algorithm_name='LogisticRegression',
            cv_split=5,
            X=X,
            y=y,
            test_size=0.2,
            random_state=2000
        )
    
    # Convert conf_matrix from ndarray to pd.DataFrame
    conf_matrix = pd.DataFrame(conf_matrix)

    # Assertions
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert isinstance(model, LogisticRegression)
    assert isinstance(acc, float)
    assert isinstance(conf_matrix, pd.DataFrame)

def test_tune_logistic_regression():
    # Load example dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Test tune_logistic_regression function
    X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val = tune_logistic_regression(
        X=X,
        y=y,
        test_size=0.2,
        random_state=123,
        cv_split=5,
        hyper_params={"max_iter": 20000}
    )

    # Assertions
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert isinstance(model, LogisticRegression)
    assert isinstance(acc, float)
    assert isinstance(conf_matrix, np.ndarray)
    assert isinstance(cross_val, np.ndarray)
    assert len(cross_val) == 5

    # Check accuracy
    accuracy = model.score(X_test, y_test)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

    # Convert conf_matrix from ndarray to pd.DataFrame
    conf_matrix = pd.DataFrame(conf_matrix)

    # Check conf_matrix
    assert isinstance(conf_matrix, pd.DataFrame)
