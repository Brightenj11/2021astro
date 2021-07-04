import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np


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


def scale_variables(flux_var, filter_from, filter_to):
    """
    Scale flux_equation variables from one band to another band using flux_var

    :param flux_var: Numpy array of variables for all bands
    :param filter_from: band to scale from
    :param filter_to: band to scale to
    :return: size 7 numpy array that holds the scaled variables in the filter_to band
    """
    # Default Values going from filter r to filter g
    base = 0
    to = 7
    # Extra array used when scaling from 'i' band
    helper_scale = np.ones(5)
    if filter_from == 'i':
        helper_scale = flux_var[13:18]
    if filter_to == 'i':
        to = 13
    if filter_to == 'z':
        to = 19

    # Set Variables
    amplitude, amplitude_scale = flux_var[base] * helper_scale[0], flux_var[to]
    beta, gamma, beta_scale, gamma_scale = flux_var[base + 1] * helper_scale[1], flux_var[base + 2] * helper_scale[2], \
        flux_var[to + 1], flux_var[to + 2]
    t0 = flux_var[3]
    tau_rise, tau_fall, tau_rise_scale, tau_fall_scale = flux_var[base + 4] * helper_scale[3], flux_var[base + 5] * \
        helper_scale[4], flux_var[to + 3], flux_var[to + 4]
    scatter = flux_var[to + 5]

    # Scale Variables
    temp_var = np.zeros(7)
    temp_var[0] = 10. ** amplitude * amplitude_scale
    temp_var[1], temp_var[2] = beta * beta_scale, gamma * gamma_scale
    temp_var[3] = t0
    temp_var[4], temp_var[5] = tau_rise * tau_rise_scale, tau_fall * tau_fall_scale
    temp_var[6] = scatter
    return temp_var


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
    log_amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars

    amplitude = 10. ** log_amplitude
    sigma2 = s_n ** 2 + yerr ** 2
    return -0.5 * (np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / sigma2 +
                          np.log(sigma2)))


def log_prior(flux_vars, x, y, yerr, fobs_max):
    """Sets Priors for initial flux_vars variables"""
    log_amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars

    if not ((np.log10(1) < log_amplitude) and (log_amplitude < np.log10(100 * fobs_max)) and
            (0 < beta) and (beta < 0.01) and (0 < gamma) and
            (x[np.argmax(y)] - 100 < t0) and (t0 < x[np.argmax(y)] + 300) and
            (0.01 < tau_rise) and (tau_rise < 50) and
            (1 < tau_fall) and (tau_fall < 300) and
            (0 < s_n) and (s_n < 3 * np.std(yerr))):
        return -np.inf

    # Gaussian Prior for plateau duration (gamma)
    mu1 = 5.
    sigma1 = 25.
    mu2 = 60.
    sigma2 = 900.
    return 2 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (gamma - mu1) ** 2 / sigma1 ** 2 + \
        1 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (gamma - mu2) ** 2 / sigma2 ** 2


def log_prior_scale(flux_vars, yerr):
    """Priors for scaling variables"""
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


def log_probability(flux_vars, x, y, yerr, x2, y2, yerr2, x3, y3, yerr3, x4, y4, yerr4):
    """Sums log prior and likelihood"""
    fobs_max = np.max(y)  # Maximum observed flux

    # Calls Priors
    lp_r = log_prior(flux_vars[:7], x, y, yerr, fobs_max)
    lp_g = log_prior_scale(flux_vars[7:13], yerr)
    lp_i = log_prior_scale(flux_vars[13:19], yerr)
    lp_z = log_prior_scale(flux_vars[19:], yerr)

    if not (np.isfinite(lp_r) and np.isfinite(lp_g) and np.isfinite(lp_i) and np.isfinite(lp_z)):
        return -np.inf

    # Scale variables
    vars_r = flux_vars[:7]
    vars_g = scale_variables(flux_vars, filter_from='r', filter_to='g')
    vars_i = scale_variables(flux_vars, filter_from='r', filter_to='i')
    vars_z = scale_variables(flux_vars, filter_from='i', filter_to='z')

    return lp_r + lp_g + lp_i + lp_z + log_likelihood(vars_r, x, y, yerr) + log_likelihood(vars_g, x2, y2, yerr2) +\
        log_likelihood(vars_i, x3, y3, yerr3) + log_likelihood(vars_z, x4, y4, yerr4)


def mcmc(x, y, yerr, x2, y2, yerr2, x3, y3, yerr3, x4, y4, yerr4):
    """
    Runs emcee given x, y, yerr and plots the best fit and flux_vars corner plot given data (x, y, yerr, ...)
    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 25, 100
    p0 = [np.log10(200), 0.001, 100, x[np.argmax(y)], 5, 10, 20, 1.1, 1.1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 10,
          1, 1, 1, 1, 1, 10]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y, yerr, x2, y2, yerr2,
                                                                                 x3, y3, yerr3, x4, y4, yerr4))
    sampler.run_mcmc(pos, 7500)

    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)

    # TODO: Make a plotting function
    # Model Fits vs Data
    plt.plot(x, flux_equation(x, 10. ** best[0], *best[1:6]), label='Best Fit R', color='r')
    plt.errorbar(x, y, yerr, ls='none', label='R Data', color='r')

    # G band data
    g_vars = scale_variables(best, filter_from='r', filter_to='g')
    plt.plot(x2, flux_equation(x2, *g_vars[:6]), label='Best Fit G', color='g')
    plt.errorbar(x2, y2, yerr2, ls='none', label='G Data', color='g')

    # I band data
    i_vars = scale_variables(best, filter_from='r', filter_to='i')
    plt.plot(x3, flux_equation(x3, *i_vars[:6]), label='Best Fit I', color='b')
    plt.errorbar(x3, y3, yerr3, ls='none', label='I Data', color='b')

    # Z band data
    z_vars = scale_variables(best, filter_from='i', filter_to='z')
    plt.plot(x4, flux_equation(x4, *z_vars[:6]), label='Best Fit Z', color='y')
    plt.errorbar(x4, y4, yerr4, ls='none', label='Z Data', color='y')

    plt.xlim([55900, 56100])
    plt.ylim([-50, 400])
    plt.legend()
    plt.show()

    plt.plot(x2, flux_equation(x2, *g_vars[:6]), label='Best Fit G', color='g')
    plt.errorbar(x2, y2, yerr2, ls='none', label='G Data', color='g')
    plt.legend()
    plt.show()

    # I band data
    i_vars = scale_variables(best, filter_from='r', filter_to='i')
    plt.plot(x3, flux_equation(x3, *i_vars[:6]), label='Best Fit I', color='b')
    plt.errorbar(x3, y3, yerr3, ls='none', label='I Data', color='b')
    plt.legend()
    plt.show()

    # Z band data
    z_vars = scale_variables(best, filter_from='i', filter_to='z')
    plt.plot(x4, flux_equation(x4, *z_vars[:6]), label='Best Fit Z', color='y')
    plt.errorbar(x4, y4, yerr4, ls='none', label='Z Data', color='y')
    plt.ylim([-50, 400])
    plt.legend()
    plt.show()

    # Corner
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    labels = ["Amplitude", "Plateau Slope", "Plateau Duration", "Reference Epoch", "Rise Time", "Fall Time", "Scatter",
              "G_amplitude", "G_plateau", "G_duration", "G_rise", "G_fall", "G_scatter",
              "I_amplitude", "I_plateau", "I_duration", "I_rise", "I_fall", "I_scatter",
              "Z_amplitude", "Z_plateau", "Z_duration", "Z_rise", "Z_fall", "Z_scatter"]
    corner.corner(flat_samples, labels=labels)
    plt.show()


if __name__ == '__main__':
    xr, yr, yerr_r = read_data('r')
    xg, yg, yerr_g = read_data('g')
    xi, yi, yerr_i = read_data('i')
    xz, yz, yerr_z = read_data('z')
    mcmc(xr, yr, yerr_r, xg, yg, yerr_g, xi, yi, yerr_i, xz, yz, yerr_z)
