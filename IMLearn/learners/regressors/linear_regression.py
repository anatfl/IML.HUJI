from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics import loss_functions


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """


    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def add_inter_column(self, X: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to add first column to avoid intercept

        Returns
        -------
        new_X : ndarray of shape (n_samples, n+1_features)
                Input data with first column of 1's
        """
        intercept_column = np.ones((len(X), 1))
        new_x = np.concatenate((intercept_column, X), axis=1)
        return new_x

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of
        `self.include_intercept_`
        """
        if self.include_intercept_:
            new_x = self.add_inter_column(X)
            w_hat = pinv(new_x) @ y
            self.coefs_ = w_hat
        else:
            w_hat = pinv(X) @ y
            self.coefs_ = w_hat

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            new_x = self.add_inter_column(X)
            y_hat = new_x @ self.coefs_
            return y_hat

        else:
            y_hat = X @ self.coefs_
            return y_hat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_prediction = self._predict(X)
        mse = loss_functions.mean_square_error(y, y_prediction)
        return mse
