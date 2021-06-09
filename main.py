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
    :param tau_rise: Rise Time (days)
    :param tau_fall: Fall Time (days)
    :return: Flux
    """
    return (amplitude * ((1 - beta * np.minimum(time - t0, gamma)) *
                         np.exp(-(np.maximum(time - t0, gamma) - gamma) / tau_fall)) /
            (1 + np.exp(-(time - t0) / tau_rise)))


def flux_plot(flux_vars, time_limit, points, mu, sigma):
    """
    Plot the flux of a transient

    :param flux_vars: See flux_equation() parameters
    :param time_limit: Total Time Range (days)
    :param points: Number of Data Points (days)
    :param mu: Mean of Noise (days)
    :param sigma: Standard Deviation of Noise (days)
    """
    amplitude, beta, gamma, t0, tau_rise, tau_fall = flux_vars
    time = np.linspace(0, time_limit, num=points)
    plt.plot(time, flux_equation(time, amplitude, beta, gamma, t0, tau_rise, tau_fall) +
             np.random.normal(mu, sigma, points))  # Add Gaussian Noise to Flux
    plt.xlabel('Phase (days)')
    plt.ylabel('Flux')
    plt.show()


def metric_log_likelihood(flux_vars, time, data, sigmas):
    amplitude, beta, gamma, t0, tau_rise, tau_fall = flux_vars
    return -0.5*(np.sum((data - flux_equation(time, amplitude, beta, gamma, t0, tau_rise, tau_fall))**2/sigmas**2 +
                        np.log(sigmas**2)))


def mcmc(time_limit, points, mu, sigma):
    time = np.linspace(0, time_limit, num=points)
    my_data = flux_equation(time, 10 ** 3, 0.0001, 175, 20, 5, 5) + np.random.normal(mu, sigma, points)
    my_sigmas = np.zeros(points) + sigma

    num_dim, num_walkers = 6, 100
    p0 = [10 ** 3, 0.0001, 175, 20, 5, 5]
    pos = [p0 + np.random.randn(num_dim) for i in range(num_walkers)]  # WHAT IS THIS?????

    sampler = emcee.EnsembleSampler(num_walkers, num_dim, metric_log_likelihood, args=(time, my_data, my_sigmas))
    sampler.run_mcmc(pos, 1000)

    samples = sampler.chain[:, 500:, :].reshape((-1, num_dim))
    corner.corner(samples, labels=["1", "2", "3", "4", "5", "6"])
    plt.show()

    plt.plot(samples[:, 0])
    plt.xlabel('Iteration Number')
    plt.ylabel('Beta')
    plt.title('A Sample Chain')
    plt.show()

    '''
    print "CENTER",np.percentile(samples[:,0],[16,50,84])
    print "HEIGHT",np.percentile(samples[:,1],[16,50,84])
    print "WIDTH",np.percentile(samples[:,2],[16,50,84])
    '''


if __name__ == '__main__':
    # Test Run
    # flux_plot([10 ** 3, 0.0001, 175, 20, 5, 5], 250, 250, 100, 10)
    mcmc(250, 250, 100, 10)
