import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers.decision_stump import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape 
    (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) =\
        generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ad_obj = AdaBoost(DecisionStump, n_learners)
    ad_obj.fit(train_X, train_y)
    train_err = []
    test_err = []
    num_of_learners = np.arange(1, n_learners)
    for t in num_of_learners:
        train_err.append(ad_obj.partial_loss(train_X, train_y, t))
        test_err.append(ad_obj.partial_loss(test_X, test_y, t))
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=num_of_learners, y=train_err, mode="lines",
                              name=r'train samples'))
    fig1.add_trace(go.Scatter(x=num_of_learners, y=test_err, mode="lines",
                              name=r'test samples'))
    fig1.update_layout(title="(1) Adaboost error on train and test as function"
                             " of number of learners",
                       xaxis_title="number of learners",
                       yaxis_title="error")
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    limits = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[
         train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=2, cols=2,
                         subplot_titles=[rf"{num} models" for num in T],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(
            lambda x: ad_obj.partial_predict(x, t),
            limits[0], limits[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color=test_y.astype(int),
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(
        title=r"(2) Decision Boundaries Of Models according to number "
              r"of models",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)

    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_loss_num = np.argmin([ad_obj.partial_loss(test_X, test_y, k) for
                              k in np.arange(1, n_learners)]) + 1
    y_hat = ad_obj.partial_predict(test_X, min_loss_num)
    fig3 = go.Figure()
    fig3.add_traces([decision_surface(
        lambda x: ad_obj.partial_predict(x, min_loss_num),
        limits[0], limits[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                   showlegend=False,
                   marker=dict(color=test_y.astype(int),
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))])

    fig3.update_layout(
        title=rf"(3) Decision Surface Of ensemble with minimal error,"
              rf" ensemble size: {min_loss_num}, "
              rf"accuracy: {accuracy(test_y, y_hat):.4f}",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)

    fig3.show()

    # Question 4: Decision surface with weighted samples
    fig4 = go.Figure()
    fig4.add_traces([decision_surface(ad_obj.predict, limits[0], limits[1],
                                      showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                mode="markers",
                                showlegend=False,
                                marker=
                                dict(color=train_y.astype(int),
                                     colorscale=[custom[0], custom[-1]],
                                     line=dict(color="black", width=1),
                                     size=ad_obj.D_/np.max(ad_obj.D_) * 10))])

    fig4.update_layout(
        title=r"(4) Decision Surface Of ensemble with size 250 and"
        r" weighted samples",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0.0)
    fit_and_evaluate_adaboost(0.4)
