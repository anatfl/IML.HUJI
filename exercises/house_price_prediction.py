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
    # replace all blanks with 0:
    df = df.fillna(0)

    # features which must br positive:
    for feature in ['sqft_living', 'sqft_above', 'sqft_lot', 'floors',
                    'price']:
        df = df[df[feature] > 0]

    # remove the odds:
    df = df[df['bedrooms'] < 16]
    df = df[df["sqft_lot"] < 1120000]
    df = df[df["sqft_lot15"] < 800000]

    # inserting better new columns:
    df['is_renovated'] = df.apply(lambda row: 1 if row['yr_renovated'] > 0
    else 0, axis=1)

    df['isnt_old'] = df.apply(lambda row: 1 if ((2022 - row['yr_built']) < 20
                                                or (row['is_renovated']
                                                    == 1)) else 0, axis=1)

    # remove irrelevant columns to evaluate:
    df = df.drop(columns=['date', 'lat', 'long', 'yr_renovated'])

    # casting categorical features:
    df = pd.get_dummies(data=df, prefix='zipcode', columns=['zipcode'])

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
                                          r"and Prices is {1}".format(i,
                                                                      corr)))
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
    train_samples, train_y, test_samples, test_y = \
        split_train_test(design_matrix, y_response)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data For every percentage p in 10%, 11%, ..., 100%, repeat the
    # following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)
    mean_lst = []
    var_lst = []
    for percent in range(10, 101):
        loss_lst = []
        for counter in range(10):
            train_p_samples = train_samples.sample(frac=percent/100)
            train_p_y = train_y[train_p_samples.index]
            p_fit = LinearRegression().fit(train_p_samples, train_p_y)
            p_loss = p_fit.loss(test_samples, test_y)
            loss_lst.append(p_loss)
        p_loss_lst = np.array(loss_lst)
        mean_lst.append(np.mean(p_loss_lst))
        var_lst.append(np.std(p_loss_lst))

    mean_lst = np.array(mean_lst)
    var_lst = np.array(var_lst)

    fig = go.Figure([go.Scatter(x=list(range(10, 101)), y=mean_lst,
                                mode='markers+lines',
                                name='mean loss'),
                     go.Scatter(x=list(range(10, 101)),
                                y=mean_lst-(2 * var_lst),
                                fill=None, mode="lines",
                                line=dict(color="lightgrey"),
                                name='error ribbon'),
                     go.Scatter(x=list(range(10, 101)),
                                y=mean_lst + 2 * var_lst,
                                fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"),
                                name='error ribbon')],
                    layout=go.Layout(title=r"$\text{average loss as "
                                           r"function of training size }$",
                                     xaxis=dict(title="x% of train set"),
                                     yaxis=dict(title="loss of test set"),
                                     height=500))
    fig.show()

