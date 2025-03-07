from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy

PATH = "..\\datasets\\"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first
    2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis) as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data, true_y = load_dataset(PATH + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(p: Perceptron, X: np.ndarray, y: int):
            losses.append(p._loss(data,true_y))

        p_obj = Perceptron(callback=callback)
        p_obj.fit(data, true_y)

        # Plot figure of loss as function of fitting iteration
        x_seq = np.arange(0, len(losses))
        px.line(x=x_seq, y=losses, title="perceptron algorithm's training loss"
                                         "values as a function of the training"
                                         " iterations - {0}".format(n),
                labels=dict(x="iteration num", y="loss")).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified
    covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 \
        else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # load dataset
        X, true_y = load_dataset(PATH + f)

        # Fit models and predict over training set
        lda_obj = LDA()
        bayes_obj = GaussianNaiveBayes()

        lda_obj.fit(X, true_y)
        bayes_obj.fit(X, true_y)

        lda_pred = lda_obj.predict(X)
        bayes_pred = bayes_obj.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions on the right. Plot title
        # should specify dataset used and subplot titles should specify
        # algorithm and accuracy
        # Create subplots
        classifiers = ["Naive Gaussian Bayes", "LDA"]
        title_1 = f"{f} - Classifier is: {classifiers[0]}, Accuracy is: " \
                  f"{accuracy(true_y, bayes_pred):.4f}"
        title_2 = f"{f} - Classifier is: {classifiers[1]}, Accuracy is: " \
                  f"{accuracy(true_y, lda_pred):.4f}"

        # creating subplots #
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(title_1, title_2))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(row=1, col=1,
                      trace=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                       marker=
                                       go.scatter.Marker(color=bayes_pred,
                                                         symbol=true_y)))
        fig.add_trace(row=1, col=2,
                      trace=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                       marker=
                                       go.scatter.Marker(color=lda_pred,
                                                         symbol=true_y)))

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(row=1, col=1, trace=go.Scatter(x=bayes_obj.mu_[:, 0],
                                                     y=bayes_obj.mu_[:, 1],
                                                     mode='markers',
                                                     showlegend=False,
                                                     marker=dict(color='black',
                                                                 symbol='x',
                                                                 size=10)))

        fig.add_trace(row=1, col=2, trace=go.Scatter(x=lda_obj.mu_[:, 0],
                                                     y=lda_obj.mu_[:, 1],
                                                     mode='markers',
                                                     showlegend=False,
                                                     marker=dict(color='black',
                                                                 symbol='x',
                                                                 size=10)))

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda_obj.classes_)):
            cov = np.zeros(shape=(X.shape[1], X.shape[1]))
            np.fill_diagonal(cov, bayes_obj.vars_[i])
            fig.add_trace(row=1, col=1, trace=get_ellipse(bayes_obj.mu_[i],
                                                          cov))
            fig.add_trace(row=1, col=2,
                          trace=get_ellipse(lda_obj.mu_[i], lda_obj.cov_))

        fig.update_layout(showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()


