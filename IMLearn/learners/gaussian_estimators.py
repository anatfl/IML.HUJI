from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased
            estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
            been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in
            `UnivariateGaussian.fit` function.

        var_: float
            Estimated variance initialized as None. To be set in
            `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated
        estimation (where estimator is either biased or unbiased). Then sets
        `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        if self.biased_ == "false":
            self.var_ = X.var(ddof=1)

        else:
            self.var_ = X.var(ddof=0)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`pdf` function")
        pdf_lst = []
        for xi in X:
            exp_power = ((xi - self.mu_) ** 2) / (2 * self.var_)
            denominator = (np.sqrt(2 * np.pi * self.var_)) * np.exp(exp_power)
            xi_pdf = 1 / denominator
            pdf_lst.append(xi_pdf)
        pdf_arr = np.array(pdf_lst)
        return pdf_arr

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified
        Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        n = len(X)
        first_elem = -0.5 * n * np.log(2 * np.pi)
        second_elem = 0.5 * n * np.log(sigma)
        third_elem = 0.5 * sigma * np.sum(np.power(X-mu, 2))
        return first_elem - second_elem - third_elem


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
             been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in
            `MultivariateGaussian.fit` function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in
            `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated
        estimation. Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False, bias=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`pdf` function")
        else:
            d = len(self.cov_)
            e_power = (X - self.mu_) @ inv(self.cov_) * (X - self.mu_)
            left_factor = np.power(2 * np.pi, d) * det(self.cov_)
            res = np.power(left_factor, -1/2) * np.exp(-1/2 * e_power)
            return res

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
        model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given
            parameters of Gaussian
        """
        m, d = X.shape
        first_elem = -0.5 * m*d * np.log(2 * np.pi)
        second_elem = 0.5 * m * slogdet(cov)[1]
        third_elem = 0.5 * np.sum((X-mu) @ inv(cov) * (X-mu))
        return first_elem - second_elem - third_elem