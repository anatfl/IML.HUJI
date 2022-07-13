from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated
        model.
        When called, the scoring function receives the true- and predicted
        values for each sample and potentially additional arguments.
        The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    idx = np.random.permutation(y.shape[0])
    X = X[idx]
    y = y[idx]

    X_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)
    valid_score, train_score = np.zeros(cv), np.zeros(cv)

    for i in range(cv):
        # separate train & validation sets:
        x_valid = X_split[i]
        y_valid = y_split[i]
        x_train = np.concatenate(X_split[:i] + X_split[i+1:])
        y_train = np.concatenate(y_split[:i] + y_split[i+1:])

        # fit train set:
        estimator.fit(x_train, y_train)

        # predict for train and validation sets:
        train_pred = estimator.predict(x_train)
        val_pred = estimator.predict(x_valid)

        # calculate error:
        train_score[i] = scoring(y_train, train_pred)
        valid_score[i] = scoring(y_valid, val_pred)

    return np.average(train_score), np.average(valid_score)

