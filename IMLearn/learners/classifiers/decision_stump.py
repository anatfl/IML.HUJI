from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART
    algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature
        is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = np.inf
        for sign, i in product([-1, 1], range(X.shape[1])):
            threshold, loss = self._find_threshold(X[:, i], y, sign)
            if loss < min_loss:
                self.sign_, self.threshold_, self.j_ = sign, threshold, i
                min_loss = loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign`
        whereas values which equal to or above the threshold are predicted
        as `sign`
        """
        y_hat = np.array([self.sign_ if xj >= self.threshold_ else -self.sign_
                          for xj in X[:, self.j_]])
        return y_hat

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to
        perform a split The threshold is found according to the value
        minimizing the misclassification error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are
        predicted as `-sign` whereas values which equal to or above the
        threshold are predicted as `sign`
        """
        sort_idx = np.argsort(values)
        y, x = labels[sort_idx], values[sort_idx]
        sorted_threshold = np.concatenate([[-np.inf],
                                           (x[1:] + x[:-1])/2, [np.inf]])
        min_threshold_loss = np.abs(np.sum(y[np.sign(y) == sign]))
        losses_lst = np.append(min_threshold_loss, min_threshold_loss -
                               np.cumsum(y * sign))
        min_loss_idx = np.argmin(losses_lst)
        return sorted_threshold[min_loss_idx], losses_lst[min_loss_idx]

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
