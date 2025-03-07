from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in
        `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in
        `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()

        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None,\
                                                                      None,\
                                                                      None, \
                                                                      None,\
                                                                      None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector,
        same covariance matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)

        mu = [np.mean(X[y == i], axis=0) for i in self.classes_]
        self.mu_ = np.array(mu)

        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))
        self.pi_ = np.array(counts / X.shape[0])
        for i in range(len(self.classes_)):
            x = X[y == self.classes_[i]]
            self.cov_ += self.pi_[i] * np.cov(x, rowvar=False)

        self._cov_inv = inv(self.cov_)

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
        likelihood = self.likelihood(X)
        y_pred = self.classes_[np.argmax(likelihood, axis=1)]
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`likelihood` function")

        likelihood_matrix = np.zeros(shape=(X.shape[0], len(self.classes_)))
        d = len(self.cov_)
        det_cov = det(self.cov_)
        for i in range(X.shape[0]):
            xi_l = []
            for j in range(len(self.classes_)):
                e_power = (X[i] - self.mu_[j]) @ self._cov_inv @ \
                          (X[i] - self.mu_[j])
                left_factor = np.power(2 * np.pi, d) * det_cov
                pdf = 1 / (np.sqrt(left_factor))
                res = pdf * np.exp(-0.5 * e_power) * self.pi_[j]
                xi_l.append(res)
            likelihood_matrix[i] = np.array(xi_l)

        return likelihood_matrix

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
        mce = misclassification_error(y, y_pred)
        return mce
