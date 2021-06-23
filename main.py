import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename: str = 'PS1_PS1MD_PSc370330.snana.dat'):
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
        if entree[1] == 'r':
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


def flux_plot(flux_vars, time_limit, points, sigma, mu=0):
    """
    Plot the flux of a transient with gaussian noise

    :param flux_vars: See flux_equation() parameters
    :param time_limit: Total Time Range (days)
    :param points: Number of Data Points (days)
    :param sigma: Standard Deviation of Noise (days)
    :param mu: Mean of Noise (days). Default set to 0
    """
    amplitude, beta, gamma, t0, tau_rise, tau_fall = flux_vars
    time = np.linspace(0, time_limit, num=points)

    plt.plot(time, flux_equation(time, amplitude, beta, gamma, t0, tau_rise, tau_fall) +
             np.random.normal(mu, sigma, points))  # Add Gaussian Noise to Flux
    plt.xlabel('Phase (days)')
    plt.ylabel('Flux')
    plt.show()


def metric_log_likelihood(flux_vars, x, y, yerr):
    """
    Computes the log likelihood of a model

    :param flux_vars: See flux_equation() parameters
    :param x: Times (days)
    :param y: (flux)
    :param yerr: (flux)
    :return: Log Likelihood
    """
    amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars

    sigma2 = s_n ** 2 + yerr ** 2
    return -0.5 * (np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / sigma2 +
                          np.log(sigma2)))


def log_prior(flux_vars, fobs_max):
    # TODO: How to set range for s_n?
    # Sets Priors for all flux_vars variables
    amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars

    # Uniform Priors
    if not np.log(1) < np.log(amplitude) < np.log(100 * fobs_max) and \
            0 < beta < 0.01 and \
            0 < gamma and \
            xs[np.where(ys == np.max(ys))][0] - 100 < t0 < xs[np.where(ys == np.max(ys))][0] + 300 and \
            0.01 < tau_rise < 50 and \
            1 < tau_fall < 300 and \
            0 < s_n < 100:
        return -np.inf

    if gamma < 0:
        return -np.inf

    # Gaussian Prior for plateau duration (gamma)
    mu1 = 5
    sigma1 = 25
    mu2 = 60
    sigma2 = 900
    return 2 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (gamma - mu1) ** 2 / sigma1 ** 2 + \
        1 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (gamma - mu2) ** 2 / sigma2 ** 2


def log_probability(flux_vars, x, y, yerr):
    # Sums log prior and likelihood
    fobs_max = np.max(y)  # Maximum observed flux

    # Calls Priors
    lp = log_prior(flux_vars, fobs_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + metric_log_likelihood(flux_vars, x, y, yerr)


def mcmc(x, y, yerr):
    """
    Runs emcee given x, y, yerr and plots the best fit and flux_vars corner plot

    :param x:
    :param y:
    :param yerr:
    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 7, 150
    p0 = [200, 0.005, 100, xs[np.where(ys == np.max(ys))][0], 5, 10, 20]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, 10000)

    # Plot example walks
    labels = ["Amplitude", "Plateau Slope", "Plateau Duration", "Reference Epoch", "Rise Time", "Fall Time", "Scatter"]
    half = int(np.floor(len(labels) / 2))
    second_half = len(labels) - half
    
    fig1, axes = plt.subplots(half, sharex=True)
    samples = sampler.get_chain()
    for i in range(half):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Step Number")
    fig, axes2 = plt.subplots(second_half, sharex=True)
    for i in range(second_half):
        ax = axes2[i]
        ax.plot(samples[:, :, i + half], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i + half])

    axes[-1].set_xlabel("Step Number")
    plt.show()

    print(sampler.get_chain()[:, 0:, 2][:, 0].shape)
    plt.plot(sampler.get_chain()[:, 0:, 2][:, 0])
    plt.figure()

    # Plot Best Fit to Data
    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)

    plt.plot(x, flux_equation(x, *best[:-1]), label='Best Fit')
    plt.errorbar(x, y, yerr, ls='none')
    plt.legend()
    plt.show()

    # Remove Burn-In Time
    # tau = sampler.get_autocorr_time()
    # print('tau ', tau)  # Roughly 100

    # Corner Plot
    flat_samples = sampler.get_chain(discard=600, thin=15, flat=True)
    labels = ["Amplitude", "Plateau Slope", "Plateau Duration", "Reference Epoch", "Rise Time", "Fall Time", "Scatter"]
    corner.corner(flat_samples, labels=labels)
    plt.show()

    '''
    # Plot Sample Chain
    plt.plot(flat_samples[:, 0])
    plt.xlabel('Iteration Number')
    plt.ylabel('Amplitude')
    plt.title('A Sample Chain')
    plt.show()


    # Print 16, 50, 84 Percentiles for Variables
    print("Amplitude", np.percentile(samples[:, 0], [16, 50, 84]))
    print("Plateau Slope", np.percentile(samples[:, 1], [16, 50, 84]))
    print("Plateau Duration", np.percentile(samples[:, 2], [16, 50, 84]))
    '''


def fit_red(x, y, yerr):
    mcmc(x, y, yerr)


if __name__ == '__main__':
    xs, ys, yerrs = read_data()
    fit_red(xs, ys, yerrs)

