import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    # replace all blanks with 0:
    df = df.fillna(0)
    # remove the odds:
    df = df[df['Temp'] > -70]
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    design_matrix = load_data('..\datasets\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = design_matrix.loc[design_matrix['Country'] == 'Israel']
    fig1 = px.scatter(israel_df, x="DayOfYear", y="Temp",
                      color=israel_df["Year"].astype(str),
                      title="Average daily temperature change as function of "
                            "DayOfYear")
    fig1.show()

    std_israel = israel_df.groupby('Month').agg('std')
    std_israel = std_israel.reset_index()
    fig2 = px.bar(std_israel,
                  labels=dict(x="Month", y="std of daily temp"),
                  x='Month', y='Temp',
                  title="standard deviation of daily temperatures change as "
                  "function of Month")
    fig2.show()

    # Question 3 - Exploring differences between countries
    c_m_group = design_matrix.groupby(['Country', 'Month'])
    c_m_data = c_m_group['Temp'].agg(['mean', 'std'])
    c_m_data = c_m_data.reset_index()
    fig3 = px.line(c_m_data, x='Month', y='mean', error_y='std',
                   color='Country', title="Average and standard deviation of "
                                          "temperatures according to 'Country'"
                                          " and 'Month'")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    loss_k_lst = []
    train_x, train_y, test_x, test_y = split_train_test(israel_df['DayOfYear'],
                                                        israel_df['Temp'])
    for k in range(1, 11):
        k_obj = PolynomialFitting(k)
        k_obj.fit(train_x, train_y)
        k_loss = k_obj.loss(test_x, test_y)
        loss_k_lst.append(round(k_loss, 2))

    for i, loss in enumerate(loss_k_lst):
        print("loss of test set for k={} is: {}".format(i+1, loss))

    fig4 = px.bar(x=range(1, 11), y=loss_k_lst,
                  title="Test error (loss) for each value of k",
                  labels=dict(x="k value", y="test loss"))

    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    k_5_obj = PolynomialFitting(5)
    k_5_obj.fit(israel_df['DayOfYear'], israel_df['Temp'])

    df_other = design_matrix[design_matrix['Country'] !=
                             'Israel'].groupby('Country')
    other_c = []
    loss_5_lst = []

    for curr_country, group in df_other:
        other_c.append(curr_country)
        c_loss = k_5_obj.loss(group['DayOfYear'], group['Temp'])
        loss_5_lst.append(c_loss)

    fig5 = px.bar(x=other_c, y=loss_5_lst,
                  labels=dict(x='countries',
                              y='loss value'),
                  title='Evaluating Israel fitted model on different countries'
                        '(model error)')
    fig5.show()
