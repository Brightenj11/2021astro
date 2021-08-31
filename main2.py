import os
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Distance


def read_data(filter_name, filename: str = 'PS1_PS1MD_PSc370330.snana.dat'):
    """
    General function to read .snana.dat files and get specific filter data

    :param filter_name: Gets data from this filter character
    :param filename:
    :return: times (x), flux values (y), and flux errors (yerr)
    """
    data = np.genfromtxt(os.getcwd() + '/ps1_sne_zenodo/' + filename, dtype=None, skip_header=17,
                         skip_footer=1, usecols=(1, 2, 4, 5), encoding=None)

    x = list()
    y = list()
    yerr = list()

    for entree in data:
        if entree[1] == filter_name:
            # if entree[2] / entree[3] > 3.:
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
    temp_var[0] = amplitude + np.log10(amplitude_scale)  # Amplitude is in log space
    temp_var[1], temp_var[2] = beta * beta_scale, gamma * gamma_scale
    temp_var[3] = t0
    temp_var[4], temp_var[5] = tau_rise * tau_rise_scale, tau_fall * tau_fall_scale
    temp_var[6] = scatter
    return temp_var


def plot_data_vs_fit(x, y, yerr, flux_vars, filter_from, filter_to):
    """Plots a specific filter's fit with its data"""
    # Set plot color
    plot_color = 'b'
    if filter_to == 'g' or filter_to == 'r':
        plot_color = filter_to
    if filter_to == 'z':
        plot_color = 'y'
    # Scale Variables
    scaled_vars = scale_variables(flux_vars, filter_from=filter_from, filter_to=filter_to)
    # Plot fit then data
    plt.plot(x, flux_equation(x, 10. ** scaled_vars[0], *scaled_vars[1:6]), label='Best Fit ' + filter_to,
             color=plot_color)
    plt.errorbar(x, y, yerr, ls='none', label=filter_to + ' Data', color=plot_color)


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
    return np.nan_to_num(amplitude * ((1. - beta * np.minimum(time - t0, gamma)) *
                         np.exp(-(np.maximum(time - t0, gamma) - gamma) / tau_fall)) /
                         (1. + np.exp(-(time - t0) / tau_rise)))


def sanders(t, t0, log_a, log_b1, log_b2, log_bdn, log_bdc, t1, tp, t2, td, m_p):
    a = np.exp(log_a)
    b1 = np.exp(log_b1)
    b2 = np.exp(log_b2)
    bdn = np.exp(log_bdn)
    bdc = np.exp(log_bdc)

    m_1 = m_p / np.exp(b1 * tp)
    m_2 = m_p / np.exp(b2 * t2)
    m_d = m_2 / np.exp(bdn * td)

    if t < t0:
        return 0.0
    elif t < t0 + t1:
        return m_1 * ((t - t0) / t1) ** a
    elif t < t0 + t1 + tp:
        return m_1 * np.exp(b1 * (t - (t1 + t0)))
    elif t < t0 + t1 + tp + t2:
        return m_p * np.exp(-b2 * (t - (tp + t1 + t0)))
    elif t < t0 + t1 + tp + t2 + td:
        return m_2 * np.exp(-bdn * (t - (t2 + tp + t1 + t0)))
    elif t0 + t1 + tp + t2 + td <= t:
        return m_d * np.exp(-bdc * (t - (td + t2 + tp + t1 + t0)))
    else:
        raise ValueError('time t is either negative or somehow did not fit into the conditional blocks')


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
    sigma2 = s_n ** 2. + yerr ** 2.
    return -0.5 * (np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / sigma2 +
                          np.log(sigma2)))


def log_prior(flux_vars, x, y, yerr, fobs_max):
    """Sets Priors for initial flux_vars variables"""
    log_amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars

    if not ((np.log10(1.) < log_amplitude) and (log_amplitude < np.log10(100. * fobs_max)) and
            (0. < beta) and (beta < 0.01) and (0. < gamma) and
            (x[np.argmax(y)] - 100. < t0) and (t0 < x[np.argmax(y)] + 300.) and
            (0.01 < tau_rise) and (tau_rise < 50.) and
            (1. < tau_fall) and (tau_fall < 300.) and
            (0. < s_n) and (s_n < 3. * np.std(yerr))):
        return -np.inf

    # Gaussian Prior for plateau duration (gamma)
    mu1 = 5.
    sigma1 = 25.
    mu2 = 60.
    sigma2 = 900.
    return 2. / 3 * np.log(1. / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (gamma - mu1) ** 2 / sigma1 ** 2 + \
        1. / 3 * np.log(1. / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (gamma - mu2) ** 2 / sigma2 ** 2


def log_prior_scale(flux_vars, yerr):
    """Priors for scaling variables"""
    scale_a, scale_b, scale_g, scale_tr, scale_tf, s_n2 = flux_vars

    if not((0. < scale_a) and (0. < scale_b) and (0. < scale_g) and (0. < scale_tr) and (0. < scale_tf) and (0. < s_n2)
           and (s_n2 < 3. * np.std(yerr))):
        return -np.inf

    # TODO: Should we change different mu/sigma for each scaling variable
    # Gaussian Priors for all scaling variables
    mu1 = 1.
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
    Runs emcee given data and plots the best fit and flux_vars corner plot given data (x, y, yerr, ...)
    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 25, 100
    p0 = [np.log10(500), 0.001, 100, x[np.argmax(y)], 5, 10, 2 * np.std(yerr), 1, 1, 1, 1, 1, 2 * np.std(yerr),
          1, 1, 1, 1, 1, 2 * np.std(yerr), 1, 1, 1, 1, 1, 2 * np.std(yerr)]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y, yerr, x2, y2, yerr2,
                                                                                 x3, y3, yerr3, x4, y4, yerr4))
    sampler.run_mcmc(pos, 5000)

    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)

    # Plot All Band Fits vs ALl data
    plt.plot(x, flux_equation(x, 10. ** best[0], *best[1:6]), label='Best Fit R', color='r')
    plt.errorbar(x, y, yerr, ls='none', label='R Data', color='r')

    plot_data_vs_fit(x2, y2, yerr2, best, filter_from='r', filter_to='g')
    plot_data_vs_fit(x3, y3, yerr3, best, filter_from='r', filter_to='i')
    plot_data_vs_fit(x4, y4, yerr4, best, filter_from='i', filter_to='z')
    plt.legend()
    plt.show()

    # R band fit
    plt.plot(x, flux_equation(x, 10. ** best[0], *best[1:6]), label='Best Fit R', color='r')
    plt.errorbar(x, y, yerr, ls='none', label='R Data', color='r')
    plt.legend()
    plt.show()

    # G band fit
    plot_data_vs_fit(x2, y2, yerr2, best, filter_from='r', filter_to='g')
    plt.legend()
    plt.show()

    # I band fit
    plot_data_vs_fit(x3, y3, yerr3, best, filter_from='r', filter_to='i')
    plt.legend()
    plt.show()

    # Z band fit
    plot_data_vs_fit(x4, y4, yerr4, best, filter_from='i', filter_to='z')
    plt.legend()
    plt.show()

    # Corner Plot
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    labels = ["Amplitude", "Plateau Slope", "Plateau Duration", "Reference Epoch", "Rise Time", "Fall Time", "Scatter",
              "G_amplitude", "G_plateau", "G_duration", "G_rise", "G_fall", "G_scatter",
              "I_amplitude", "I_plateau", "I_duration", "I_rise", "I_fall", "I_scatter",
              "Z_amplitude", "Z_plateau", "Z_duration", "Z_rise", "Z_fall", "Z_scatter"]
    figure = corner.corner(flat_samples, label_kwargs={"fontsize": 6})
    # TODO: adjust plotting labels
    # figure.subplots_adjust(right=1.2, top=1.2)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='both', which='major', pad=0.25)
    # figure.savefig("../../Documents/2021astro/corner_griz.png", dpi=200, pad_inches=0.3, bbox_inches='tight')
    plt.show()


def convert_flux(flux_arr):
    return -2.5 * np.log10(flux_arr) + 27.5


def convert_lum(arr):
    return -2.5 * np.log10(10 ** 7 * arr)


if __name__ == '__main__':
    file_name = 'PS1_PS1MD_PSc500332.snana.dat'
    xr, yr, yerr_r = read_data('r', filename=file_name)
    xg, yg, yerr_g = read_data('g', filename=file_name)
    xi, yi, yerr_i = read_data('i', filename=file_name)
    xz, yz, yerr_z = read_data('z', filename=file_name)
    # mcmc(xr, yr, yerr_r, xg, yg, yerr_g, xi, yi, yerr_i, xz, yz, yerr_z)

    vectorized_sanders = np.vectorize(sanders, otypes=[float])
    xs = np.linspace(56397.2 - 30, 56397.2 + 500, 10000)
    distance = Distance(z=0.098, unit='pc')

    # plt.scatter(xr, convert_flux(yr) - 5 * (np.log10(distance.value) - 1))
    # plt.plot(xs, convert_lum(vectorized_sanders(xs, 55789.3, -1.0, -2.5, -4.7, -2.7, -3.9, 1.0, 6.0, 84., 12., 1.02)))
    # plt.gca().invert_yaxis()
    # plt.show()

    with np.load('PSc500332.npz') as data:
        r_amp = data['r_amp']
        r_beta = data['r_beta']
        r_gamma = data['r_gamma']
        r_t0 = data['t0']
        r_tr = data['r_tr']
        r_tf = data['r_tf']
        g_amp = data['r_amp'] * data['g_scale_a']
        g_beta = data['r_beta'] * data['g_scale_b']
        g_gamma = data['r_gamma'] * data['g_scale_g']
        g_tr = data['r_tr'] * data['g_scale_tr']
        g_tf = data['r_tf'] * data['g_scale_tf']
    plt.errorbar(xr, yr, yerr_r)
    plt.show()
    plt.scatter(xr, convert_flux(yr) - 5 * (np.log10(distance.value) - 1))
    plt.plot(xs, convert_lum(vectorized_sanders(xs, 56397.2, -1.0, -2.3, -3.2, -3.0, -4.3, 1.0, 9.0, 105., 11., 2.21)))
    for i in range(100, 130):
        plt.plot(xs, convert_flux(flux_equation(xs, 10. ** r_amp[0][i], r_beta[0][i], 10. ** r_gamma[0][i], r_t0[0][i], r_tr[0][i], r_tf[0][i])) - 5 * (np.log10(distance.value) - 1), alpha=0.1)
        # plt.plot(xs, convert_flux(flux_equation(xs, 10. ** g_amp[0][i], g_beta[0][i], 10. ** g_gamma[0][i], r_t0[0][i], g_tr[0][i], g_tf[0][i])) - 5 * (np.log10(distance.value) - 1), alpha=0.1)
    plt.gca().invert_yaxis()
    plt.show()
