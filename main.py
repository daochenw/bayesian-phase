# program to generate figure 1 and figure 2 of the paper arXiv: 1508.00869v1
# "Efficient Bayesian Phase Estimation"
from math import sin, cos, pi, sqrt, log, ceil, e as eu
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize


def update(mu, sigma, e, m_cct, theta_cct, m, k=1):
    # this is almost exactly the pseudo code in the appendix: Algorithm 1
    (x_inc, y_inc, n_a) = (0, 0, 0)
    x = np.random.normal(mu, sigma, m)
    u = np.random.uniform(0, 1, m)
    for i in range(m):
        p__ex = (1 + (1-2*e) * cos(m_cct * (x[i] + theta_cct))) / 2
        if p__ex >= k*u[i]:
            x_inc = x_inc + cos(x[i])
            y_inc = y_inc + sin(x[i])
            n_a = n_a + 1

    x_inc = x_inc/n_a
    y_inc = y_inc/n_a

    mu_ = np.angle([x_inc+y_inc*1j])[0]
    # the st.dev formula given in the paper has an incorrectly extra sqrt; it is removed here:
    sigma_ = sqrt(log(max(1., 1/(x_inc**2+y_inc**2))))

    return mu_, sigma_


def bayes_risk_integrand(phi, cct, mu, sigma):

    m_cct, theta_cct = cct[0], cct[1]

    m_cct = round(m_cct)

    # analytical solution for best estimator of phi
    e = np.array([0, 1])

    temp_0 = (1 - 2 * e) * (eu ** (2j * mu * m_cct) - eu ** (-2j * theta_cct * m_cct))

    temp_1 = (1 - 2 * e) * (eu ** (2j * mu * m_cct) + eu ** (-2j * theta_cct * m_cct))

    temp_2 = 2 * eu ** (m_cct * (1j * (mu - theta_cct) + sigma ** 2 * m_cct / 2))

    phi_opt = mu + (1j * sigma ** 2 * m_cct) * temp_0 / (temp_1 + temp_2)

    # x_opt should be real itself (checked), this just removes any small imaginary part due to imprecision
    phi_opt = phi_opt.real

    # put the optimal estimator into the Bayes risk integrand
    # the phi_opt is getting recalculated when it doesn't need to be

    integrand = (phi-phi_opt)**2*0.5*(1+np.array([1, -1])*cos(m_cct*(phi+theta_cct))) * \
                (sqrt(2*pi*sigma**2))**(-1)*eu**(-((phi-mu)/sigma)**2)

    integrand = sum(integrand)

    return integrand


def bayes_risk(cct, mu, sigma):

    integral = quad(bayes_risk_integrand, -np.inf, np.inf, args=(cct, mu, sigma))

    # print("integral is", integral, "error is", err)
    # print("% error is", abs(err)/integral)

    return integral[0]


def optimal_experiment(mu, sigma):

    #print("I'm minimizing")
    res = minimize(bayes_risk, np.array([np.ceil(1.25/sigma), -(mu+sigma)]),
                   args=(mu, sigma),
                   method='Nelder-Mead')

    #print("I've done the minimizing")
    # bounds = ((0, None), (None, None))

    cct = res.x
    cct[0] = round(cct[0])

    #print("the suggested M: ", np.ceil(1.25/sigma), "theta_1: ", -mu-sigma, "and theta_2: ", -mu+sigma)
    #print("my optimal M: ", cct[0], "and theta: ", cct[1])
    #print(1/sigma)

    # sense check
    th_risk = (1-1/eu)*sigma**2
    risk = bayes_risk(cct, mu, sigma)

    # assert risk >= th_risk

    # print("% error = ", (risk-th_risk)/th_risk)

    return res.x


def run(phi_true, n, mu_0=0, sigma_0=pi, m=200):
    output = np.zeros([n, 3])
    m_cct_total = 0
    for i in range(n):
        if sigma_0 == 0:
            output[i:n] = output[i]
            return output

        # particle guess heuristic for m_cct and theta_cct according to eqn (4)
        if False:
            m_cct = ceil(1.25/sigma_0)
            theta_cct = np.random.normal(mu_0, sigma_0)

        # local Bayes risk minimiser
        if True:
            opt = optimal_experiment(mu_0, sigma_0)
            m_cct = opt[0]
            theta_cct = opt[1]

        m_cct_total = m_cct_total + m_cct
        p = (1-cos(m_cct*(phi_true+theta_cct)))/2
        e = np.random.binomial(1, p)
        mu, sigma = update(mu_0, sigma_0, e, m_cct, theta_cct, m)
        output[i, 0], output[i, 1], output[i, 2] = mu, m_cct_total, sigma
        mu_0, sigma_0 = mu, sigma
    return output


def performance(n_exp, n_true=20, mode=0):
    # mode=0 for Fig. 1, and mode=1 for Fig. 2
    # n_true is the number of random initial choices of the true eigenphase (over which the median error is taken)
    # n_exp is the number of experiments performed to try to estimate each true eigenphase

    # the algorithm seems quite sensitive to how well the initial guess is within the true_phi
    if mode == 0:
        x = np.random.rand(n_true) * 2 * pi - pi
    else:
        t = 10000
        x = 2*pi*np.random.randint(0, t, n_true)/t-pi

    # define a comparison array:
    # layer 0: predicted phi, 1: number of U applications, 2: error
    comp = np.zeros([n_exp, n_true, 3])

    for j in range(n_true):
        print(j)
        temp = run(x[j], n_exp)
        comp[:, j, 0] = temp[:, 0]
        comp[:, j, 1] = temp[:, 1]

    comp[:, :, 2] = np.absolute(comp[:, :, 0] - x)

    # taking the median error and number of U applications
    comp_med = np.median(comp[:, :, 1:3], axis=1)
    b = np.array([range(n_exp)])
    comp_med = np.concatenate((b.T, comp_med), axis=1)

    return comp, comp_med

if True:
    data_1 = performance(20, mode=0)[1]
    data_2 = performance(20, mode=1)[1]
    plt.subplot(121)
    plt.semilogy(data_1[:, 0], data_1[:, 2], basey=10)
    plt.title('Figure 1')
    plt.xlabel('Experiment number')
    plt.ylabel('Median error')
    plt.subplot(122)
    plt.loglog(data_2[:, 1], data_2[:, 2], basex=10, basey=10)
    plt.title('Figure 2')
    plt.xlabel('Median Applications of U')
    plt.show(block=True)
