import os
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Distance
import pandas as pd


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
            if entree[2] / entree[3] > 3.:
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

    # print('rise value (first)', -np.exp(-0.921034/a) * (-np.exp(0.921034 / a) * t0 - np.exp((b1 * tp) / a) * t1))
    # print('rise value (second)', -(-b1 * tp - b1 * t0 - b1 * t1 + 0.921034) / b1)
    # print('decline value', -(-b2 * tp -b2 * t0 - b2 * t1 - 0.921034) / b2)
    # print(t0, t0+t1, t0+t1+tp, t0+t1+tp+t2, t0+t1+tp+t2+td)
    if t < t0:
        return 0.0
    elif t < t0 + t1:
        return m_1 * ((t - t0) / t1) ** a
    elif t < t0 + t1 + tp:  # Peak at t = t0 + t1 + tp
        # print('peak', -2.5 * np.log10(10 ** 7 * m_1 * np.exp(b1 * tp)))
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
    # Load PS1 names, PSc names, redshifts, and time of explosions
    with np.load('PS1-list.npz') as data2:
        PS1_list = data2['SN_list']

    psc_list = ['PSc000001', 'PSc000076', 'PSc000098', 'PSc000218', 'PSc010163', 'PSc050581', 'PSc060230', 'PSc060284',
                'PSc061196', 'PSc070291', 'PSc071072', 'PSc080768', 'PSc091034', 'PSc100170', 'PSc120113', 'PSc120175',
                'PSc120215', 'PSc120228', 'PSc120333', 'PSc120419', 'PSc130327', 'PSc130816', 'PSc130913', 'PSc131014',
                'PSc131015', 'PSc131089', 'PSc131094', 'PSc140103', 'PSc140357', 'PSc140418', 'PSc150249', 'PSc150267',
                'PSc150412', 'PSc150507', 'PSc150576', 'PSc150692', 'PSc160042', 'PSc190299', 'PSc190369', 'PSc300008',
                'PSc300112', 'PSc300221', 'PSc310359', 'PSc320347', 'PSc330027', 'PSc330040', 'PSc330053', 'PSc330064',
                'PSc330232', 'PSc340337', 'PSc340346', 'PSc350092', 'PSc350330', 'PSc370407', 'PSc370519', 'PSc380014',
                'PSc380056', 'PSc400158', 'PSc420393', 'PSc440011', 'PSc440066', 'PSc440202', 'PSc450008', 'PSc450284',
                'PSc480154', 'PSc490025', 'PSc500012', 'PSc500332', 'PSc500491', 'PSc510185', 'PSc510238', 'PSc530128',
                'PSc540215', 'PSc550130', 'PSc580084', 'PSc590045']
    redshift_list = [0.071, 0.260, 0.057, 0.093, 0.086, 0.333, 0.127, 0.064, 0.059, 0.132, 0.141, 0.123, 0.184, 0.074,
                     0.040, 0.105, 0.092, 0.120, 0.150, 0.093, 0.096, 0.082, 0.210, 0.068, 0.070, 0.055, 0.101, 0.145,
                     0.148, 0.191, 0.133, 0.209, 0.236, 0.073, 0.168, 0.077, 0.146, 0.077, 0.052, 0.061, 0.095, 0.043,
                     0.103, 0.068, 0.059, 0.180, 0.175, 0.145, 0.132, 0.040, 0.123, 0.093, 0.042, 0.079, 0.107, 0.104,
                     0.101, 0.096, 0.096, 0.070, 0.086, 0.145, 0.109, 0.140, 0.149, 0.079, 0.077, 0.098, 0.167, 0.185,
                     0.240, 0.113, 0.097, 0.095, 0.121, 0.130]
    t0_list = [55207, 55207, 55212, 55214, 55242, 55362, 55389, 55389, 55407, 55414, 55428, 55467, 55499, 55510, 55566,
               55566, 55567, 55568, 55572, 55576, 55597, 55597, 55602, 55605, 55605, 55615, 55615, 55633, 55636, 55648,
               55666, 55666, 55675, 55675, 55675, 55675, 55705, 55795, 55803, 55775, 55792, 55800, 55831, 55852, 55883,
               55883, 55883, 55883, 55893, 55911, 55911, 55941, 55948, 56000, 56013, 56029, 56029, 56091, 56166, 56207,
               56210, 56210, 56235, 56240, 56326, 56353, 56385, 56399, 56408, 56417, 56417, 56478, 56513, 56564, 56624,
               56659]

    run = True
    plot = False

    name = list()
    band = list()
    peak_height = list()
    rise_slope = list()
    plateau_slope = list()
    if run:
        for SN in range(len(psc_list)):
            number = psc_list[SN]
            if number == 'PSc131014' or number == 'PSc300008' or number == 'PSc120113' or number == 'PSc130327' or \
                    number == 'PSc130816' or number == 'PSc340346' or number == 'PSc350092' or number == 'PSc350330'\
                    or number == 'PSc540215':
                continue
            else:
                file_name = 'PS1_PS1MD_' + number + '.snana.dat'
                xr, yr, yerr_r = read_data('r', filename=file_name)
                xg, yg, yerr_g = read_data('g', filename=file_name)
                xi, yi, yerr_i = read_data('i', filename=file_name)
                xz, yz, yerr_z = read_data('z', filename=file_name)

                # Find distance to object in parsecs
                distance = Distance(z=redshift_list[SN], unit='pc')

                # Find distance modulus
                yr = convert_flux(yr) - 5 * (np.log10(distance.value) - 1)
                yg = convert_flux(yg) - 5 * (np.log10(distance.value) - 1)
                yi = convert_flux(yi) - 5 * (np.log10(distance.value) - 1)
                yz = convert_flux(yz) - 5 * (np.log10(distance.value) - 1)

                # Load Variables
                with np.load(number + '.npz') as data:
                    r_amp = data['r_amp']
                    r_beta = data['r_beta']
                    r_gamma = data['r_gamma']
                    r_t0 = data['t0']
                    r_tr = data['r_tr']
                    r_tf = data['r_tf']

                    g_amp = data['r_amp'] + data['g_scale_a']
                    g_beta = data['r_beta'] * 10. ** data['g_scale_b']
                    g_gamma = data['r_gamma'] + data['g_scale_g']
                    g_tr = data['r_tr'] * 10. ** data['g_scale_tr']
                    g_tf = data['r_tf'] * 10. ** data['g_scale_tf']

                    i_amp = data['r_amp'] + data['i_scale_a']
                    i_beta = data['r_beta'] * 10. ** data['i_scale_b']
                    i_gamma = data['r_gamma'] + data['i_scale_g']
                    i_tr = data['r_tr'] * 10. ** data['i_scale_tr']
                    i_tf = data['r_tf'] * 10. ** data['i_scale_tf']

                    z_amp = i_amp + data['z_scale_a']
                    z_beta = i_beta * 10. ** data['z_scale_b']
                    z_gamma = i_gamma + data['z_scale_g']
                    z_tr = i_tr * 10. ** data['z_scale_tr']
                    z_tf = i_tf * 10. ** data['z_scale_tf']

                vectorized_sanders = np.vectorize(sanders, otypes=[float])
                # Variables for setting plot limits
                xs = np.linspace(t0_list[SN] - 30, t0_list[SN] + 125, 10000)

                # Get y magnitudes
                magnitude_r = convert_flux(flux_equation(xs, 10. ** r_amp[0][0], r_beta[0][0], 10. ** r_gamma[0][0],
                                                    r_t0[0][0], r_tr[0][0], r_tf[0][0])) - 5 * (np.log10(distance.value) - 1)
                magnitude_g = convert_flux(flux_equation(xs, 10. ** g_amp[0][0], g_beta[0][0], 10. ** g_gamma[0][0],
                                                    r_t0[0][0], g_tr[0][0], g_tf[0][0])) - 5 * (np.log10(distance.value)-1)
                magnitude_i = convert_flux(flux_equation(xs, 10. ** i_amp[0][0], i_beta[0][0], 10. ** i_gamma[0][0],
                                                    r_t0[0][0], i_tr[0][0], i_tf[0][0])) - 5 * (np.log10(distance.value)-1)
                magnitude_z = convert_flux(flux_equation(xs, 10. ** z_amp[0][0], z_beta[0][0], 10. ** z_gamma[0][0],
                                                    r_t0[0][0], z_tr[0][0], z_tf[0][0])) - 5 * (np.log10(distance.value)-1)

                # Time of peak magnitude
                time_max_r = xs[np.argmin(magnitude_r)]
                time_max_g = xs[np.argmin(magnitude_g)]
                time_max_i = xs[np.argmin(magnitude_i)]
                time_max_z = xs[np.argmin(magnitude_z)]
                # Magnitude peak
                peak_r = np.min(magnitude_r)
                peak_g = np.min(magnitude_g)
                peak_i = np.min(magnitude_i)
                peak_z = np.min(magnitude_z)

                # Add SN name, band, and magnitude peak
                name.extend((PS1_list[SN], PS1_list[SN], PS1_list[SN], PS1_list[SN]))
                band.extend(('r', 'g', 'i', 'z'))
                peak_height.extend((peak_r, peak_g, peak_i, peak_z))

                # Slopes for all banes
                plateau_slope.append(xs[np.argmin(magnitude_r) + np.argmin(np.abs(magnitude_r[np.nonzero(xs > time_max_r)] - (peak_r + 1)))]
                                     - time_max_r)
                rise_slope.append(time_max_r -
                                  xs[np.argmin(np.abs(magnitude_r[np.nonzero(xs < time_max_r)] - (peak_r + 1)))])
                plateau_slope.append(xs[np.argmin(magnitude_g) + np.argmin(np.abs(magnitude_g[np.nonzero(xs > time_max_g)] - (peak_g + 1)))]
                                     - time_max_g)
                rise_slope.append(time_max_g -
                                  xs[np.argmin(np.abs(magnitude_g[np.nonzero(xs < time_max_g)] - (peak_g + 1)))])
                plateau_slope.append(xs[np.argmin(magnitude_i) + np.argmin(np.abs(magnitude_i[np.nonzero(xs > time_max_i)] - (peak_i + 1)))]
                                     - time_max_i)
                rise_slope.append(time_max_z -
                                  xs[np.argmin(np.abs(magnitude_i[np.nonzero(xs < time_max_i)] - (peak_i + 1)))])
                plateau_slope.append(xs[np.argmin(magnitude_z) + np.argmin(np.abs(magnitude_z[np.nonzero(xs > time_max_z)] - (peak_z + 1)))]
                                     - time_max_z)
                rise_slope.append(time_max_z -
                                  xs[np.argmin(np.abs(magnitude_z[np.nonzero(xs < time_max_z)] - (peak_z + 1)))])

                if plot:
                    # X_temp used to set plot limits
                    x_temp = np.linspace(t0_list[SN] - 5, t0_list[SN] + 5)
                    ys = np.concatenate(
                        (convert_flux(flux_equation(x_temp, 10. ** r_amp[0][0], r_beta[0][0], 10. ** r_gamma[0][0],
                                                    r_t0[0][0], r_tr[0][0], r_tf[0][0])) - 5 * (
                                     np.log10(distance.value) - 1),
                         convert_flux(flux_equation(x_temp, 10. ** g_amp[0][0], g_beta[0][0], 10. ** g_gamma[0][0],
                                                    r_t0[0][0], g_tr[0][0], g_tf[0][0])) - 5 * (
                                     np.log10(distance.value) - 1),
                         convert_flux(flux_equation(x_temp, 10. ** i_amp[0][0], i_beta[0][0], 10. ** i_gamma[0][0],
                                                    r_t0[0][0], i_tr[0][0], i_tf[0][0])) - 5 * (
                                     np.log10(distance.value) - 1),
                         convert_flux(flux_equation(x_temp, 10. ** z_amp[0][0], z_beta[0][0], 10. ** z_gamma[0][0],
                                                    r_t0[0][0], z_tr[0][0], z_tf[0][0])) - 5 * (
                                     np.log10(distance.value) - 1)))

                    # Plot data
                    plt.figure()
                    plt.scatter(xr, yr, color='r', label='R band')
                    plt.scatter(xg, yg, color='g', label='G Band')
                    plt.scatter(xi, yi, color='b', label='I band')
                    plt.scatter(xz, yz, color='y', label='Z band')

                    # Plot fits
                    for i in range(0, 25):
                        plt.plot(xs, magnitude_r, alpha=0.1, color='r')
                        plt.plot(xs, magnitude_g, alpha=0.1, color='g')
                        plt.plot(xs, magnitude_i, alpha=0.1, color='b')
                        plt.plot(xs, magnitude_z, alpha=0.1, color='y')
                    plt.gca().invert_yaxis()
                    plt.legend()
                    plt.xlim(xs[0], xs[-1])
                    plt.ylim(np.max(ys) + 2.5, np.min(ys) - 1.5)
                    plt.show()
                    plt.savefig(number, dpi=300)

    # Save model statistics
    output_df = pd.DataFrame({'name': name, 'band': band,
                              'peak_height': peak_height,
                              'rise_slope': rise_slope,
                              'plateau_slope': plateau_slope})
    output_df.to_csv('model.dat')

    # Sanders statistics
    sanders_name = list()
    sanders_band = list()
    sanders_peak_height = list()
    sanders_rise_slope = list()
    sanders_plateau_slope = list()

    s_data = np.genfromtxt('sanders_params.dat', dtype=None, usecols=(0, 1, 2, 5, 8, 11, 20, 23, 32), encoding='utf-8')
    for s_entree in s_data:
        if s_entree[1] in PS1_list:
            # Track Variables
            s_t0 = s_entree[2]
            s_a = np.exp(s_entree[3])
            s_b1 = np.exp(s_entree[4])
            s_b2 = np.exp(s_entree[5])
            s_t1 = s_entree[6]
            s_tp = s_entree[7]
            s_m_p = s_entree[8]
            s_m_1 = s_m_p / np.exp(s_b1 * s_tp)

            sanders_name.append(s_entree[1])
            sanders_band.append(s_entree[0])
            # Check if rise slope falls between t0 + t1 and t0 + t1 + tp
            if s_tp > 460517 / (500000 * s_b1):
                sanders_rise_slope.append((s_t0 + s_t1 + s_tp) + (-s_b1 * s_tp - s_b1 * s_t0 - s_b1 * s_t1 + 0.921034) / s_b1)
            # Rise slope falls between t0 and t0 + t1
            else:
                sanders_rise_slope.append((s_t0 + s_t1 + s_tp) + np.exp(-0.921034 / s_a) * (-np.exp(0.921034 / s_a) * s_t0 - np.exp((s_b1 * s_tp) / s_a) * s_t1))
            # Plateau time - Peak time
            sanders_plateau_slope.append(-(-s_b2 * s_tp - s_b2 * s_t0 - s_b2 * s_t1 - 0.921034) / s_b2 - (s_t0 + s_t1 + s_tp))
            # Amplitude
            sanders_peak_height.append(-2.5 * np.log10(10 ** 7 * s_m_1 * np.exp(s_b1 * s_tp)))

    # Save sanders statistics
    output_df = pd.DataFrame({'sanders_name': sanders_name, 'sanders_band': sanders_band, 'sanders_peak_height':sanders_peak_height,
                              'sanders_rise_slope': sanders_rise_slope, 'sanders_plateau_slope': sanders_plateau_slope})
    output_df.to_csv('sanders.dat')