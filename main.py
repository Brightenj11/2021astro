import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


def flux_equation(time, amplitude, beta, gamma, t0, tau_rise, tau_fall):
    """
    Equation of Flux for a transient

    :param time:
    :param amplitude: Amplitude (flux)
    :param beta: Plateau Slope (days^-1)
    :param gamma: Plateau Duration (days)
    :param t0: Reference Epoch (days)
    :param tau_rise: Rise    Time (days)
    :param tau_fall: Fall Time (days)
    :return: Flux over time
    """
    return (amplitude * ((1 - beta * np.minimum(time - t0, gamma)) *
                         np.exp(-(np.maximum(time - t0, gamma) - gamma) / tau_fall)) /
            (1 + np.exp(-(time - t0) / tau_rise)))


def flux_plot(flux_vars, time_limit, points, sigma, mu=0):
    """
    Plot the flux of a transient

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


def metric_log_likelihood(flux_vars, x, y):
    """
    Computes the log likelihood of a model

    :param flux_vars: See flux_equation() parameters
    :param x: Times (days)
    :param y: True Values
    :return: Log Likelihood
    """
    amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars
    return -0.5 * (
        np.sum((y - flux_equation(x, amplitude, beta, gamma, t0, tau_rise, tau_fall)) ** 2 / s_n ** 2 +
               np.log(s_n ** 2)))


def log_prior(flux_vars, fobs_max=1400):
    # TODO: Add docstring
    # TODO: How to set fobs_max properly
    # TODO: How to set range for s_n?
    amplitude, beta, gamma, t0, tau_rise, tau_fall, s_n = flux_vars
    # Uniform Priors
    if not np.log(1) < amplitude < np.log(fobs_max) and 0 < beta < 0.01 and -50 < t0 < 300 and 0.01 < tau_rise < 50 \
            and 1 < tau_fall < 300 and 0 < s_n < 100:
        return -np.inf

    # Gaussian Prior for Plateau Duration (gamma)
    mu1 = 5
    sigma1 = 25
    mu2 = 60
    sigma2 = 900
    return 2 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (gamma - mu1) ** 2 / sigma1 ** 2 + \
        1 / 3 * np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (gamma - mu2) ** 2 / sigma2 ** 2


def log_probability(flux_vars, x, y, fobs_max=1400):
    # Calls Priors
    lp = log_prior(flux_vars, fobs_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + metric_log_likelihood(flux_vars, x, y)


def mcmc(time_limit, points, sigma, mu=0):
    """
    Runs a MCMC and plots example chains, the best fit vs data, and the posterior distribution

    :param time_limit: Length of times
    :param points: How often to sample up to the time_limit
    :param sigma: Standard Deviation of Noise (days)
    :param mu: Mean of Noise (days). Default set to 0
    """
    # Set number of dimensions, walkers, and initial position
    num_dim, num_walkers = 7, 200
    p0 = [10 ** 3, 0.005, 80, 20, 5, 5, sigma]
    pos = [p0 + 1e-4 * np.random.randn(num_dim) for i in range(num_walkers)]

    # Generate Data
    x = np.linspace(0, time_limit, num=points)
    y = flux_equation(x, *p0[:-1]) + np.random.normal(mu, sigma, points)

    # TODO: is y_err used??? remove from likelihood and probability
    # y_err = np.zeros(points) + sigma

    # Run MCMC
    sampler = emcee.EnsembleSampler(num_walkers, num_dim, log_probability, args=(x, y))
    sampler.run_mcmc(pos, 10000)

    # Plot example walks
    labels = ["Amplitude", "Plateau Slope", "Plateau Duration", "Reference Epoch", "Rise Time", "Fall Time", "Scatter"]
    half = int(np.floor(len(labels)/2))
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

    # Plot Best Fit to Data
    samples = sampler.get_chain(flat=True)
    best = samples[np.argmax(sampler.get_log_prob())]
    print('best', best)

    plt.plot(x, flux_equation(x, *best[:-1]), label='Best Fit')
    plt.plot(x, flux_equation(x, *p0[:-1]) +
             np.random.normal(mu, p0[-1], points), label='Data')
    plt.legend()
    plt.show()

    # Remove Burn-In Time
    tau = sampler.get_autocorr_time()
    print('tau ', tau)  # Roughly 100

    # Corner Plot
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    corner.corner(flat_samples, labels=labels, truths=[*p0])
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


if __name__ == '__main__':
    # Test Run
    # flux_plot([10 ** 3, 0.0001, 175, 20, 5, 5], 250, 250, 100, 10)
    mcmc(250, 250, 50)
