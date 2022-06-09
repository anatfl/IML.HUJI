import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.fitted_ = False
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.weights_ = np.zeros(self.iterations_)
        n = y.shape[0]
        self.models_ = [None] * self.iterations_
        # D(0) - uniform distribution:
        self.D_ = np.ones(n) / n
        for t in range(0, self.iterations_):
            # find base learner and fit:
            self.models_[t] = self.wl_()
            self.models_[t].fit(X, self.D_ * y)
            # predict y hat for evaluating epsilon_t:
            y_hat = self.models_[t].predict(X)
            # evaluating epsilon_t (np.abs(y_hat - y) / 2 is 1 for error):
            epsilon_t = np.sum((np.abs(y_hat - y) / 2) * self.D_)
            # evaluating Wt by the equation we learned:
            self.weights_[t] = 0.5 * np.log(1.0 / epsilon_t - 1)
            # update sample weights:
            exp_elem = np.exp((-1) * self.weights_[t] * y * y_hat)
            self.D_ = self.D_ * exp_elem
            # normalize sample weights:
            self.D_ = self.D_ / np.sum(self.D_)

    def _predict(self, X):
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
        # initializing:
        y_hat = np.zeros(X.shape[0])

        # evaluating by the equation we learned:
        for t in range(0, self.iterations_):
            y_t = self.models_[t].predict(X)
            y_hat += y_t * self.weights_[t]

        return np.sign(y_hat)

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
        y_hat = self._predict(X)
        mce = misclassification_error(y, y_hat)
        return mce

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        original_iter = self.iterations_
        self.iterations_ = T
        y_hat = self._predict(X)
        self.iterations_ = original_iter
        return y_hat

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        original_iter = self.iterations_
        self.iterations_ = T
        mce = self._loss(X, y)
        self.iterations_ = original_iter
        return mce
