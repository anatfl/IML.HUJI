from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def cross_val_q1(n_samples, noise):
    """

    Parameters
    ----------
    n_samples
    noise

    Returns
    -------

    """
    # sampling data:
    X = np.random.uniform(-1.2, 2, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = f(X)
    epsilon = np.random.normal(0, noise, n_samples)
    y_noisy = y + epsilon
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y_noisy)

    # splitting with train portion 2/3 and sorting both train and test:
    train_x, train_y, test_x, test_y = split_train_test(X_df,
                                                        y_series,
                                                        2 / 3)

    train_x = train_x.to_numpy().flatten()
    test_x = test_x.to_numpy().flatten()
    train_y = train_y.to_numpy().flatten()
    test_y = test_y.to_numpy().flatten()

    # scatter plot of true model (noiseless), train and test sets of
    # different colors:
    fig = go.Figure([go.Scatter(x=X, y=y,
                                mode='markers', name="noiseless"),
                     go.Scatter(x=train_x, y=train_y,
                                mode='markers', name="train"),
                     go.Scatter(x=test_x, y=test_y,
                                mode='markers', name="test")],
                    layout=
                    go.Layout(title=r"Model of {} samples and noise level: {},"
                                    r" training and Test Sets as different "
                                    r"colors".format(n_samples, noise),
                              xaxis=dict(title="features"),
                              yaxis=dict(title="labels")))
    fig.show()

    return train_x, train_y, test_x, test_y, X, y_noisy


def cross_val_q2(X, y):
    avg_train_err, avg_valid_err = np.zeros(11), np.zeros(11)

    for k in range(11):
        pol_obj = PolynomialFitting(k)
        avg_train_err[k], avg_valid_err[k] = \
            cross_validate(pol_obj, X, y, mean_square_error, cv=5)

    fig = go.Figure([go.Scatter(x=list(range(11)), y=avg_train_err,
                                mode='lines+markers', name="avg train err"),
                     go.Scatter(x=list(range(11)), y=avg_valid_err,
                                mode='lines+markers', name="avg "
                                                           "validation err")],
                    layout=go.Layout(
                        title=r"Average training error and Validation error as"
                        r" a function of degree in Polynomial Fitting",
                        xaxis=dict(title="Degree of Polynomial"),
                        yaxis=dict(title="Errors"),
                    showlegend=True))
    fig.show()

    return avg_valid_err


def cross_val_q3(train_x, train_y, test_x, test_y, avg_cross_val_err, noise):
    k_star = np.argmin(avg_cross_val_err)
    pol_obj = PolynomialFitting(k_star)
    pol_obj.fit(train_x, train_y)
    test_err = round(pol_obj.loss(test_x, test_y), 2)
    print("noise: {}".format(noise))
    print("best k: {}".format(k_star))
    print("CV loss: {}".format(round(avg_cross_val_err[k_star], 2)))
    print("test error with k star: {}".format(test_err))


def ridge_q7(train_x, train_y, l_range, num_evaluations):
    ridge_train_err = np.zeros(num_evaluations)
    ridge_valid_err = np.zeros(num_evaluations)
    i = 0
    for k in l_range:
        ridge_obj = RidgeRegression(k)
        ridge_train_err[i], ridge_valid_err[i] = \
            cross_validate(ridge_obj, train_x, train_y, mean_square_error)
        i += 1

    return ridge_train_err, ridge_valid_err


def lasso_q7(train_x, train_y, l_range, num_evaluations):
    lasso_train_err = np.zeros(num_evaluations)
    lasso_valid_err = np.zeros(num_evaluations)
    i = 0
    for k in l_range:
        lasso_obj = Lasso(alpha=k)
        lasso_train_err[i], lasso_valid_err[i] = \
            cross_validate(lasso_obj, train_x, train_y, mean_square_error)
        i += 1
    return lasso_train_err, lasso_valid_err


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select
    the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2)
    # + eps for eps Gaussian noise and split into training-
    # and testing portions
    train_x, train_y, test_x, test_y, X, y = cross_val_q1(n_samples, noise)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    avg_cross_val_error = cross_val_q2(train_x, train_y)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and
    # report test error
    cross_val_q3(train_x, train_y, test_x, test_y, avg_cross_val_error, noise)


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
    fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the
        algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing
    # portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y = X[:50, :], y[:50]
    test_x, test_y = X[50:, :], y[50:]

    # Question 7 - Perform CV for different values of the regularization
    # parameter for Ridge and Lasso regressions
    num_evaluations = 500
    l_range = np.linspace(0.001, 5, num=num_evaluations)

    ridge_train_err, ridge_valid_err = ridge_q7(train_x, train_y, l_range,
                                                num_evaluations)

    lasso_train_err, lasso_valid_err = lasso_q7(train_x, train_y, l_range,
                                                num_evaluations)

    plots_fig = make_subplots(1, 2, subplot_titles=["Ridge Method",
                                                    "Lasso Method"])
    plots_fig.update_layout(title=r"The Average training error and validation "
                                  r"error as a function of the lambda "
                                  r"parameter value in Regularization Methods")

    plots_fig.add_traces([go.Scatter(x=l_range, y=ridge_train_err,
                                     mode='lines+markers', name="train"),
                          go.Scatter(x=l_range, y=ridge_valid_err,
                                     mode='lines+markers', name="CV")]
                         , rows=1, cols=1)

    plots_fig.add_traces([go.Scatter(x=l_range, y=lasso_train_err,
                                     mode='lines+markers', name="train"),
                          go.Scatter(x=l_range, y=lasso_valid_err,
                                     mode='lines+markers', name="CV")],
                         rows=1, cols=2)

    plots_fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least
    # Squares model
    ridge_param = np.argmin(ridge_valid_err)
    lasso_param = np.argmin(lasso_valid_err)
    best_ridge_obj = RidgeRegression(l_range[ridge_param])
    best_lasso_object = Lasso(alpha=l_range[lasso_param])
    lin_reg_obj = LinearRegression()

    best_ridge_obj.fit(train_x, train_y)
    best_lasso_object.fit(train_x, train_y)
    lin_reg_obj.fit(train_x, train_y)

    ridge_test_err = round(best_ridge_obj.loss(test_x, test_y), 2)
    print("ridge_test_err: {}".format(ridge_test_err))
    print("best lambda: {}".format(l_range[ridge_param]))

    lasso_pred = best_lasso_object.predict(test_x)
    lasso_mse = round(mean_square_error(lasso_pred, test_y), 2)
    print("lasso mse: {}".format(lasso_mse))
    print("best lambda: {}".format(l_range[lasso_param]))

    lin_reg_test_err = round(lin_reg_obj.loss(test_x, test_y), 2)
    print("lin_reg_test_err: {}".format(lin_reg_test_err))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
