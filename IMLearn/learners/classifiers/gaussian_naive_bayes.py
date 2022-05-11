from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes.
            To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in
            `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_, self.fitted_ = \
            None, None, None, None, False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / X.shape[0]

        # initializing:
        self.mu_ = np.zeros(shape=(len(self.classes_), X.shape[1]))
        self.vars_ = np.zeros(shape=(len(self.classes_), X.shape[1]))

        for j in range(len(self.classes_)):
            self.mu_[j] = np.mean(X[y == self.classes_[j]], axis=0)
            self.vars_[j] = np.var(X[y == self.classes_[j]], axis=0,
                                   ddof=1)
        self.fitted_ = True

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

        # like LDA but now each class has it's cov:
        likelihood_matrix = np.zeros(shape=(X.shape[0], len(self.classes_)))
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        d = X.shape[1]
        for i in range(X.shape[0]):
            xi_l = []
            for j in range(len(self.classes_)):
                np.fill_diagonal(cov, self.vars_[j])
                e_power = (X[i] - self.mu_[j]) @ np.linalg.inv(cov) @ \
                          (X[i] - self.mu_[j])
                left_factor = np.power(2 * np.pi, d) * np.linalg.det(cov)
                pdf = np.power(left_factor, -0.5) * np.exp(-0.5 * e_power)
                result = pdf * self.pi_[j]
                xi_l.append(result)
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
