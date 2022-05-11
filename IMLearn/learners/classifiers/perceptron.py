from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while
            fitting to given data Callable function should receive as input
            a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] =
                 default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while
            fitting to given data Callable function should receive as input
            a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None
        self.fitted_ = False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long
        as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of
        `self.fit_intercept_`
        """
        # raise NotImplementedError()
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]
        self.coefs_ = np.zeros(X.shape[1])

        self.fitted_ = True
        iteration = 0
        change = True
        for iter in range(self.max_iter_):
            for i in range(y.shape[0]):
                if y[i] * (X[i] @ self.coefs_) <= 0:
                    addition = y[i] * X[i]
                    self.coefs_ += addition
                    self.callback_(self, X[i], y[i])
                    break


        # while iteration < self.max_iter_ and change:
        #     for (xi, yi) in zip(X, y):
        #         res = yi @ np.dot(xi, self.coefs_)
        #         if res <= 0:
        #             self.coefs_ += yi @ xi
        #             self.callback_(self, xi, yi)
        #             iteration += 1
        #             break
        #     change = False
        # self.fitted_ = True

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
        if not self.fitted_:
            raise ValueError("estimator must br fitted before predict")
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]

        y_hat = []
        for xi in X:
            yi_hat = np.dot(xi, self.coefs_)
            if yi_hat > 0:
                y_hat.append(1)
            else:
                y_hat.append(-1)

        return np.array(y_hat)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        loss = misclassification_error(y, y_pred)
        return loss
