import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


def read_data(filter_name, filename: str = 'PS1_PS1MD_PSc370330.snana.dat'):
    """
    General function to read .snana.dat files
    :param filter_name: Gets data from this filter character
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


def log_likelihood(flux_vars, x, y, yerr):
    """
    Computes the log likelihood of a model

    :param flux_vars: See flux_equation() parameters
    :param x: Times (days)
    :param y: (flux)
    :param yerr: (flux)
    :return: Log Likelihood
    """
    # TODO: get rid of second row (scale stuff)
    log_amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n, \
        scale_a, scale_b, scale_g, scale_tr, scale_tf, s_ng = flux_vars

    amplitude = 10. ** log_amplitude
    sigma2 = s_n ** 2 + yerr ** 2
    return -0.5 * (np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / sigma2 +
                          np.log(sigma2)))


def log_prior(flux_vars, x, y, yerr, fobs_max):
    # TODO: get rid of second row
    """Sets Priors for initial flux_vars variables"""
    log_amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n, \
        scale_a, scale_b, scale_g, scale_tr, scale_tf, s_ng = flux_vars

    if not ((np.log10(1) < log_amplitude) and (log_amplitude < np.log10(100 * fobs_max)) and
            (0 < beta) and (beta < 0.01) and (0 < gamma) and
            (x[np.argmax(y)] - 100 < t0) and (t0 < x[np.argmax(y)] + 300) and
            (0.01 < tau_rise) and (tau_rise < 50) and
            (1 < tau_fall) and (tau_fall < 300) and
            (0 < s_n) and (s_n < 3 * np.std(yerr))):
        return -np.inf

    # Gaussian Prior for plateau duration (gamma)
    mu1 = 5
    sigma1 = 25
    mu2 = 60
    sigma2 = 900
    return 2 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (gamma - mu1) ** 2 / sigma1 ** 2 + \
        1 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (gamma - mu2) ** 2 / sigma2 ** 2


def log_prior_scale(flux_vars, yerr):
    # TODO: Get rid of first row
    """Priors for scaling variables"""
    amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n, \
        scale_a, scale_b, scale_g, scale_tr, scale_tf, s_n2 = flux_vars

    if not((0 < scale_a) and (0 < scale_b) and (0 < scale_g) and (0 < scale_tr) and (0 < scale_tf) and (0 < s_n2)
           and (s_n2 < 3 * np.std(yerr))):
        return -np.inf

    # TODO: Do I need different mu/sigma for each scaling variable?
    # Gaussian Priors for all scaling variables
    mu1 = 1
    sigma1 = 0.5
    return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_a - mu1) ** 2 / sigma1 ** 2 + \
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_b - mu1) ** 2 / sigma1 ** 2 + \
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_g - mu1) ** 2 / sigma1 ** 2 + \
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_tr - mu1) ** 2 / sigma1 ** 2 + \
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (scale_tf - mu1) ** 2 / sigma1 ** 2


def log_probability(flux_vars, x, y, yerr, x2, y2, yerr2):
    # TODO: Rename flux_vars to red_vars. rename temp_vars to green_vars
    """Sums log prior and likelihood"""
    fobs_max = np.max(y)  # Maximum observed flux

    # Calls Priors
    lp_r = log_prior(flux_vars, x, y, yerr, fobs_max)
    lp_g = log_prior_scale(flux_vars, yerr)
    if not np.isfinite(lp_r) and not np.isfinite(lp_g):
        return -np.inf

    # TODO: Put this into a function?? Add variable names inside
    # Scale variables appropriately
    temp_vars = np.zeros(13)
    temp_vars[0] = 10. ** flux_vars[0] * flux_vars[7]
    for i in range(1, 3):
        temp_vars[i] = flux_vars[i] * flux_vars[7 + i]
    temp_vars[3] = flux_vars[3]
    for k in range(4, 6):
        temp_vars[k] = flux_vars[6 + k]
    temp_vars[6] = flux_vars[-1]

    return lp_r + lp_g + log_likelihood(flux_vars, x, y, yerr) + log_likelihood(temp_vars, x2, y2, yerr2)


def mcmc(x, y, yerr, x2, y2, yerr2):
    """
    Runs emcee given x, y, yerr and plots the best fit and flux_vars corner plot given data (x, y, yerr, x2, y2, yerr2)
    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 13, 100
    p0 = [np.log10(200), 0.001, 100, x[np.argmax(y)], 5, 10, 20, 1, 1, 1, 1, 1, 10]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y, yerr, x2, y2, yerr2))
    sampler.run_mcmc(pos, 7500)

    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)
    '''
    # Plot Single Walker
    plt.plot(sampler.get_chain()[:, 0:, 2][:, 0])
    plt.show()

    plt.plot(sampler.get_chain()[:, :, 0], alpha=0.3)
    plt.show()
    '''
     
    # Model Fits vs Data
    plt.plot(x, flux_equation(x, 10 ** best[0], *best[1:6]), label='Best Fit Red', color='r')
    plt.errorbar(x, y, yerr, ls='none', label='Red Data', color='r')

    # Green data
    green_vars = np.zeros(13)
    green_vars[0] = 10. ** best[0] * best[7]
    for i in range(1, 3):
        green_vars[i] = best[i] * best[7 + i]
    green_vars[3] = best[3]
    for k in range(4, 6):
        green_vars[k] = best[6 + k]
    green_vars[6] = best[-1]

    plt.plot(x2, flux_equation(x2, *green_vars[:6]), label='Best Fit Green', color='g')
    plt.errorbar(x2, y2, yerr2, ls='none', label='Green Data', color='g')
    plt.xlim([55800, 56400])
    plt.legend()
    plt.show()

    plt.plot(x2, flux_equation(x2, *green_vars[:6]), label='Best Fit Green', color='g')
    plt.errorbar(x2, y2, yerr2, ls='none', label='Green Data', color='g')
    plt.legend()
    plt.show()

    # Corner
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    labels = ["Amplitude", "Plateau Slope", "Plateau Duration", "Reference Epoch", "Rise Time", "Fall Time", "Scatter",
              "Amplitude Scaler", "Plateau Scaler", "Duration Scaler", "Rise Scaler", "Fall Scaler", "Scatter2"]
    corner.corner(flat_samples, labels=labels)
    plt.show()


if __name__ == '__main__':
    xr, yr, yerr_r = read_data('r')
    xg, yg, yerr_g = read_data('g')
    mcmc(xr, yr, yerr_r, xg, yg, yerr_g)
