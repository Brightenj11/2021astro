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


def metric_log_likelihood(flux_vars, x, y, yerr):
    """
    Computes the log likelihood of a model

    :param flux_vars: See flux_equation() parameters
    :param x: Times (days)
    :param y: (flux)
    :param yerr: (flux)
    :return: Log Likelihood
    """
    amplitude, beta, scale_a, scale_p = flux_vars
    # Set other parameters as constants
    gamma, t0, tau_rise, tau_fall, s_n = 8.92935484e+0, 5.60039124e+04, 3.87462396e+00, 1.89186260e+01, 1.90379142e-01

    sigma2 = s_n ** 2 + yerr ** 2
    return -0.5 * (np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / sigma2 +
                          np.log(sigma2)))


def log_prior(flux_vars, fobs_max):
    # TODO: How to set range for s_n?
    # For amplitude and plateau
    # Sets Priors for all flux_vars variables
    amplitude, beta, scale_a, scale_p = flux_vars

    # Uniform Priors
    if ((np.log(1) < np.log(amplitude)) and (np.log(amplitude) < np.log(100 * fobs_max)) and
            (0 < beta) and (beta < 0.01)):
        return 0
    return -np.inf


def log_prior_2(flux_vars):
    # For scaling variables
    amplitude, beta, scale_a, scale_p = flux_vars

    if not((0 < scale_a) and (0 < scale_p)):
        return -np.inf

    mu1 = 1
    sigma1 = 0.5
    mu2 = 1
    sigma2 = 0.5
    return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_a - mu1) ** 2 / sigma1 ** 2 + \
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (scale_p - mu2) ** 2 / sigma2 ** 2


def log_probability(flux_vars, x, y, yerr, x2, y2, yerr2):
    # Best fit parameters for red band - [3.17657446e+02 4.94039667e-03 8.92935484e+00 5.60039124e+04,
    # 3.87462396e+00 1.89186260e+01 1.90379142e-01]

    # Sums log prior and likelihood
    fobs_max = np.max(y)  # Maximum observed flux

    # Calls Priors
    lp_r = log_prior(flux_vars, fobs_max)
    lp_g = log_prior_2(flux_vars)

    if not np.isfinite(lp_r) and not np.isfinite(lp_g):
        return -np.inf

    # Scale variables appropriately
    temp_vars = flux_vars[0] * flux_vars[2], flux_vars[1] * flux_vars[3], flux_vars[2], flux_vars[3]
    return lp_r + lp_g + metric_log_likelihood(flux_vars, x, y, yerr) + metric_log_likelihood(temp_vars, x2, y2, yerr2)


def mcmc(x, y, yerr, x2, y2, yerr2):
    """
    Runs emcee given x, y, yerr and plots the best fit and flux_vars corner plot

    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 4, 150
    p0 = [200, 0.001, 1, 1]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    # Real Values
    real = [8.92935484e+0, 5.60039124e+04, 3.87462396e+00, 1.89186260e+01]

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y, yerr, x2, y2, yerr2))
    sampler.run_mcmc(pos, 5000)

    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)

    # Model Fits vs Data
    plt.plot(x, flux_equation(x, best[0], best[1], *real), label='Best Fit Red', color='r')
    plt.errorbar(x, y, yerr, ls='none', label='Red Data', color='r')

    plt.plot(x2, flux_equation(x2, best[0] * best[2], best[1] * best[3], *real), label='Best Fit Green', color='g')
    plt.errorbar(x2, y2, yerr2, ls='none', label='Green Data', color='g')
    plt.xlim([55800, 56400])
    plt.legend()
    plt.show()

    # Corner
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    labels = ["Amplitude", "Plateau Slope", "Amplitude Scaler", "Plateau Scaler"]
    corner.corner(flat_samples, labels=labels)
    plt.show()


if __name__ == '__main__':
    xr, yr, yerr_r = read_data('r')
    xg, yg, yerr_g = read_data('g')
    mcmc(xr, yr, yerr_r, xg, yg, yerr_g)
