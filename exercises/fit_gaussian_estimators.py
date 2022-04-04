from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    samples = np.random.normal(mu, var, size=1000)
    uni_obj = UnivariateGaussian()
    uni_obj.fit(samples)
    print((uni_obj.mu_, uni_obj.var_))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    m_lst = np.linspace(10, 1000, 100).astype(int)
    for m in m_lst:
        estimated_mean.append(np.abs(np.mean(samples[:m]) - mu))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=m_lst, y=estimated_mean, mode='markers+lines',
                              name=r'$\widehat\mu$'))

    fig1.update_layout(title="(2) Estimation of distance from Expectation As "
                             "Function Of Number Of Samples",
                       xaxis_title="m - number of samples",
                       yaxis_title="sample distances from real expectation")
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_arr = uni_obj.pdf(samples)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=samples, y=pdf_arr, mode='markers',
                              name=r'$\pdf\mu$'))

    fig2.update_layout(title="(3) pdf of samples with "
                             "distribution of N(10,1)",
                       xaxis_title="value of samples",
                       yaxis_title="pdf value of samples")
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples_multi = np.random.multivariate_normal(mu, cov, size=1000)
    multi_obj = MultivariateGaussian()
    multi_obj.fit(samples_multi)
    true_mu = multi_obj.mu_
    true_cov = multi_obj.cov_
    print(true_mu)
    print(true_cov)

    # Question 5 - Likelihood evaluation
    f_1_opt = np.linspace(-10, 10, 200)
    f_3_opt = np.linspace(-10, 10, 200)
    mesh_arr = np.array(np.meshgrid(f_1_opt, f_3_opt))

    # creates all optional pairs of f1 & f3:
    comb_opt = mesh_arr.T.reshape(-1, 2)
    log_like_lst = np.ndarray(shape=len(comb_opt),)
    for i in range(len(comb_opt)):
        mu_arr = np.array([comb_opt[i][0], 0, comb_opt[i][1], 0])
        log_like_lst[i] = MultivariateGaussian.log_likelihood(mu_arr, true_cov,
                                                              samples_multi)

    log_title = "(5) log-likelihood of samples with expectation mu=" \
                "[f1,0,f3,0]^T and the true covariance matrix"
    fig3 = go.Figure(go.Heatmap(x=comb_opt.T[1], y=comb_opt.T[0],
                                z=log_like_lst),
                     layout=go.Layout(title=log_title))
    fig3.update_xaxes(title="f1")
    fig3.update_yaxes(title="f3")
    fig3.show()

    # Question 6 - Maximum likelihood
    max_likelihood_index = np.argmax(log_like_lst)
    max_likelihood_pair = comb_opt[max_likelihood_index]
    # the pair is [-0.05025126 3.96984925]
    # so [-0.050 3.969]


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
