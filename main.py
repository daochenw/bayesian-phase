# program to generate figure 1 and figure 2 of the paper arXiv: 1508.00869v1
# "Efficient Bayesian Phase Estimation"
from math import cos, pi, sqrt, log, ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize


def update(mu, sigma, e, m_cct, theta_cct, m, k=1):
    # this is almost exactly the pseudo code in the appendix: Algorithm 1
    x = np.random.normal(mu, sigma, m)
    u = np.random.uniform(0, 1, m)

    p_ex = (1 + (1 - 2 * e) * np.cos(m_cct * (x + theta_cct))) / 2
    ft = p_ex >= k * u
    n = sum(ft)
    x_inc = sum(np.cos(x)[ft]) / n
    y_inc = sum(np.sin(x)[ft]) / n

    mu_ = np.angle([x_inc + y_inc * 1j])[0]
    # the st.dev formula given in the paper has an incorrectly extra sqrt; it is removed here:
    sigma_ = sqrt(log(max(1., 1 / (x_inc ** 2 + y_inc ** 2))))

    return mu_, sigma_


def optimal_phi(cct, mu, sigma):
    m_cct, theta_cct = cct[0], cct[1]

    # analytical solution (checked equal direct calculation)
    if True:
        e = np.array([1, -1])

        temp_0 = e * (np.exp(2j * mu * m_cct) - np.exp(-2j * theta_cct * m_cct))

        temp_1 = e * (np.exp(2j * mu * m_cct) + np.exp(-2j * theta_cct * m_cct))

        temp_2 = 2 * np.exp(m_cct * (1j * (mu - theta_cct) + sigma ** 2 * m_cct / 2))

        phi_opt = mu + (1j * sigma ** 2 * m_cct) * temp_0 / (temp_1 + temp_2)

        # x_opt should be real itself (checked), this just removes any small imaginary part due to imprecision

        phi_opt = phi_opt.real

    # direct calculation of optimal_phi = mean of posterior distribution; this breaks down when
    else:
        def dist(phi, d):
            return 0.5 * (1 + d * cos(m_cct * (phi + theta_cct))) * \
                   (1 / sqrt(2 * pi * sigma ** 2)) * np.exp(-0.5 * ((phi - mu) / sigma) ** 2)

        norm0, norm1 = quad(dist, -np.inf, np.inf, args=(1,))[0], quad(dist, -np.inf, np.inf, args=(-1,))[0]

        def mean_int(phi, d):
            return phi * dist(phi, d)

        phi_opt = np.array([quad(mean_int, -np.inf, np.inf, args=(1,))[0] / norm0,
                            quad(mean_int, -np.inf, np.inf, args=(-1,))[0] / norm1])

    return phi_opt


def bayes_risk_integrand(phi, phi_opt, cct, mu, sigma):
    m_cct, theta_cct = cct[0], cct[1]

    integrand = (phi - phi_opt) ** 2 * 0.5 * (1 + np.array([1, -1]) * cos(m_cct * (phi + theta_cct))) * \
                (1 / sqrt(2 * pi * sigma ** 2)) * np.exp(-0.5 * ((phi - mu) / sigma) ** 2)

    integrand = sum(integrand)

    return integrand


def bayes_risk(cct, mu, sigma):
    # analytical solution when theta_cct = 0, i.e. no inversion (checked equal direct calculation)
    if True:
        m_cct, theta_cct = cct[0], cct[1]

        integral = sigma ** 2 * (1 + m_cct ** 2 * sigma ** 2 * np.sin((mu + theta_cct) * m_cct) ** 2 /
                                 (-np.exp(m_cct ** 2 * sigma ** 2) + np.cos((mu + theta_cct) * m_cct) ** 2))

    else:
        phi_opt = optimal_phi(cct, mu, sigma)

        integral = quad(bayes_risk_integrand, -np.inf, np.inf, args=(phi_opt, cct, mu, sigma))[0]

    return integral


def optimal_experiment(mu, sigma):
    res = minimize(bayes_risk, np.array([1.25 / sigma, -(mu + sigma)]),
                   args=(mu, sigma),
                   method='Nelder-Mead')

    # cct = res.x
    # print("the suggested M: ", 1.25 / sigma, "theta_1: ", -mu - sigma, "and theta_2: ", -mu + sigma)
    # print("my optimal M: ", cct[0], "and theta: ", cct[1])

    return res.x


def run(phi_true, n, mu_0=0, sigma_0=pi, m=200, mode_exp=0):
    output = np.zeros([n, 3])
    m_cct_total = 0
    for i in range(n):
        if sigma_0 == 0:
            output[i:n] = output[i]
            return output

        # particle guess heuristic for m_cct and theta_cct according to eqn (4)
        if mode_exp == 0:
            m_cct = ceil(1.25 / sigma_0)
            theta_cct = np.random.normal(mu_0, sigma_0)

        # local Bayes risk minimiser
        else:
            opt = optimal_experiment(mu_0, sigma_0)
            m_cct = opt[0]
            theta_cct = opt[1]

        m_cct_total = m_cct_total + m_cct
        p = (1 - cos(m_cct * (phi_true + theta_cct))) / 2
        e = np.random.binomial(1, p)
        mu, sigma = update(mu_0, sigma_0, e, m_cct, theta_cct, m)
        output[i, 0], output[i, 1], output[i, 2] = mu, m_cct_total, sigma
        mu_0, sigma_0 = mu, sigma
    return output


def performance(n_exp, n_true=100, mode_exp=0, mode_true=0):
    # mode=0 for Fig. 1, and mode=1 for Fig. 2
    # n_true is the number of random initial choices ,of the true eigenphase (over which the median error is taken)
    # n_exp is the number of experiments performed to try to estimate each true eigenphase

    # the algorithm seems quite sensitive to how well the initial guess is within the true_phi
    if mode_true == 0:
        np.random.seed(0)
        x = np.random.rand(n_true) * 2 * pi - pi
    else:
        t = 10000
        np.random.seed(0)
        x = 2 * pi * np.random.randint(0, t, n_true) / t - pi

    # define a comparison array:
    # layer 0: predicted phi, 1: number of U applications, 2: error
    comp = np.zeros([n_exp, n_true, 3])

    for j in range(n_true):
        print(j)
        temp = run(x[j], n_exp, mode_exp=mode_exp)
        comp[:, j, 0] = temp[:, 0]
        comp[:, j, 1] = temp[:, 1]

    comp[:, :, 2] = np.absolute(comp[:, :, 0] - x)

    # taking the median error and number of U applications
    comp_med = np.median(comp[:, :, 1:3], axis=1)
    b = np.array([range(n_exp)])
    comp_med = np.concatenate((b.T, comp_med), axis=1)

    return comp, comp_med


def bayes_risk_plot(theta_cct, mu, sigma):
    m_cct = np.arange(0, 300, 0.1)
    risk = np.zeros(m_cct.size)

    for i in range(m_cct.size):
        print(i)
        cct = np.array([m_cct[i], theta_cct])
        risk[i] = bayes_risk(cct, mu, sigma)

    plt.plot(m_cct, risk)
    plt.show(block=True)


if True:
    data_1 = performance(50, mode_exp=0, mode_true=0)[1]
    data_2 = performance(50, mode_exp=0, mode_true=1)[1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.semilogy(data_1[:, 0], data_1[:, 2], basey=10)
    ax.semilogy(data_2[:, 0], data_2[:, 2], basey=10)
    ax.set_xlabel('Experiment number')
    ax.set_ylabel('Median error')
    ax = fig.add_subplot(1, 2, 2)
    ax.loglog(data_1[:, 1], data_1[:, 2], basex=10, basey=10, label='Heuristic')
    ax.loglog(data_2[:, 1], data_2[:, 2], basex=10, basey=10, label='Optimised')
    ax.set_xlabel('Median Applications of U')
    ax.legend(loc=0)
    plt.show()
