import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np


def read_data(filter_name, filename: str = 'PS1_PS1MD_PSc370330.snana.dat'):
    """
    General function to read .snana.dat files
    :param filename:
    :return: times (x), flux values (y), and flux errors (yerr)
    """
    data = np.genfromtxt(filename, dtype=None, skip_header=17, skip_footer=1, usecols=(1, 2, 4, 5),
                         encoding=None)

    x = list()
    y = list()
    yerr = list()

    for entree in data:
        if entree[1] == filter_name:
            x.append(entree[0])
            y.append(entree[2])
            yerr.append(entree[3])
    return np.array(x), np.array(y), np.array(yerr)


def flux_equation(time, amplitude, beta, gamma, t0, tau_rise, tau_fall):
    """
    Equation of Flux for a transient

    :param time:
    :param amplitude: Amplitude (flux)
    :param beta: Plateau Slope (days^-1)
    :param gamma: Plateau Duration (days)
    :param t0: Reference Epoch (days)
    :param tau_rise: Rise Time (days)
    :param tau_fall: Fall Time (days)
    :return: Flux over time
    """
    return np.nan_to_num(amplitude * ((1 - beta * np.minimum(time - t0, gamma)) *
                         np.exp(-(np.maximum(time - t0, gamma) - gamma) / tau_fall)) /
                         (1 + np.exp(-(time - t0) / tau_rise)))


def metric_log_likelihood(flux_vars, scale_vars, x, y, yerr):
    """
    Computes the log likelihood of a model

    :param flux_vars: See flux_equation() parameters
    :param x: Times (days)
    :param y: (flux)
    :param yerr: (flux)
    :return: Log Likelihood
    """
    amplitude, beta = flux_vars
    scale_a, scale_p = scale_vars
    amplitude *= scale_a
    beta *= scale_p

    gamma, t0, tau_rise, tau_fall, s_n = 8.92935484e+00, 5.60039124e+04, 3.87462396e+00, 1.89186260e+01, 1.90379142e-01

    sigma2 = s_n ** 2 + yerr ** 2
    return -0.5 * (np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / sigma2 +
                          np.log(sigma2)))


def log_prior(flux_vars, fobs_max):
    # TODO: How to set range for s_n?
    # Sets Priors for all flux_vars variables
    amplitude, beta = flux_vars

    # Uniform Priors
    if ((np.log(1) < np.log(amplitude)) and (np.log(amplitude) < np.log(100 * fobs_max)) and
            (0 < beta) and (beta < 0.01)):
        return 0
    return -np.inf


def log_prior_2(scale_vars):
    scale_a, scale_p = scale_vars
    mu1 = 1
    sigma1 = 0.5
    mu2 = 1
    sigma2 = 0.5
    return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_a - mu1) ** 2 / sigma1 ** 2 + \
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (scale_p - mu2) ** 2 / sigma2 ** 2


def log_probability(flux_vars, x, y, yerr, best=[0, 0], first=True):
    # [3.17657446e+02 4.94039667e-03 8.92935484e+00 5.60039124e+04
    #  3.87462396e+00 1.89186260e+01 1.90379142e-01]

    # Sums log prior and likelihood
    fobs_max = np.max(y)  # Maximum observed flux

    # Calls Priors
    if first:
        lp = log_prior(flux_vars, fobs_max)
        if not np.isfinite(lp):
            return -np.inf
        return lp + metric_log_likelihood(flux_vars, [1, 1], x, y, yerr)
    else:
        lp = log_prior_2(flux_vars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + metric_log_likelihood(best, flux_vars, x, y, yerr)


def mcmc(x, y, yerr, x2, y2, yerr2):
    """
    Runs emcee given x, y, yerr and plots the best fit and flux_vars corner plot

    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 2, 150
    p0 = [200, 0.001]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]
    # Real
    real = [8.92935484e+00, 5.60039124e+04, 3.87462396e+00, 1.89186260e+01]

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, 5000)

    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)

    # Model vs data
    plt.plot(x, flux_equation(x, *best, *real), label='Best Fit')
    plt.errorbar(x, y, yerr, ls='none')
    plt.legend()
    plt.show()

    # Corner
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    labels = ["Amplitude", "Plateau Slope"]
    corner.corner(flat_samples, labels=labels)
    plt.show()

    # Fit for scale
    p1 = [1, 1]
    pos1 = [p1 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    sampler2 = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x2, y2, yerr2, best, False))
    sampler2.run_mcmc(pos1, 5000)
    samples2 = sampler2.get_chain(flat=True)
    best2 = samples2[np.argmax(sampler.get_log_prob())]
    print('best2', best2)

    # Model vs data
    plt.plot(x2, flux_equation(x, best[0] * best2[0], best[1] * best2[1], *real), label='Best Fit')
    plt.errorbar(x2, y2, yerr2, ls='none')
    plt.legend()
    plt.show()

    # Corner
    flat_samples2 = sampler2.get_chain(discard=500, thin=10, flat=True)
    labels = ["Amplitude Scaler", "Plateau Slope Scaler"]
    corner.corner(flat_samples2, labels=labels)
    plt.show()


if __name__ == '__main__':
    xr, yr, yerr_r = read_data('r')
    xg, yg, yerr_g = read_data('g')
    mcmc(xr, yr, yerr_r, xg, yg, yerr_g)
