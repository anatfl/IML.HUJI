from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import os as os
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.fillna(0)
    for feature in ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_living',
                    'sqft_lot', 'floors', 'price']:
        df = df[df[feature] > 0]
    df = df.drop(columns='date')
    df = pd.get_dummies(data=df, prefix='zipcode', columns=['zipcode'])
    df['is_renovated'] = df.apply(lambda row: 1 if row['yr_renovated'] > 0
                                  else 0, axis=1)
    df = df.drop(columns='yr_renovated')
    df['isnt_old'] = df.apply(lambda row: 1 if ((2022 - row['yr_built']) < 30
                                                or (row['is_renovated']
                                                    == 1)) else 0, axis=1)
    response_vector = df['price']
    design_mat = df.drop(columns=['price', 'id'])
    return design_mat, response_vector


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".")\
        -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # raise NotImplementedError()
    y_std = np.std(y)
    for i in X.columns:
        corr = np.cov(X[i], y)[0][1] / (np.std(X[i]) * y_std)
        fig = go.Figure([go.Scatter(x=X[i], y=y,
                                    name=r"Correlation Between feature {0} "
                                         r"and Prices is {1}".format(i, corr),
                                    mode='markers',
                                    marker=dict(color="LightSkyBlue"),
                                    showlegend=False)], layout=dict(
                                    title=r"Correlation Between feature {0} "
                                          r"and Prices is {1}".format(i, corr)))
        fig.update_xaxes(title=i)
        fig.update_yaxes(title="price")
        pio.write_image(fig=fig, file=output_path + r"\{}.png".format(i))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    design_matrix, y_response = load_data('..\datasets\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(design_matrix, y_response, r"..\plots_of_ex2")

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall
    # training data For every percentage p in 10%, 11%, ..., 100%, repeat the
    # following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
