import numpyro
import numpyro.distributions as dist
import numpy as np
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.infer.reparam import TransformReparam
from jax import random
import jax.numpy as jnp

import arviz as az

import matplotlib.pyplot as plt


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
            if entree[3] < 100:
                x.append(entree[0])
                y.append(entree[2])
                yerr.append(entree[3])
    return jnp.array(x), jnp.array(y), jnp.array(yerr)


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
    return jnp.nan_to_num(amplitude * ((1 - beta * jnp.minimum(time - t0, gamma)) *
                                       jnp.exp(-(jnp.maximum(time - t0, gamma) - gamma) / tau_fall)) /
                          (1 + jnp.exp(-(time - t0) / tau_rise)))


def model(x, y=None, yerr=None, fobs_max=200, time_max=55200):
    log_amplitude = numpyro.sample('log_amplitude', dist.Uniform(jnp.log10(1), jnp.log10(100. * fobs_max)))
    beta = numpyro.sample('beta', dist.Uniform(0, 0.01))
    log_gamma = numpyro.sample('log_gamma', dist.Uniform(-1., 3.))
    t0 = numpyro.sample('t0', dist.Uniform(time_max - 50., time_max + 300.))
    tau_rise = numpyro.sample('tau_rise', dist.Uniform(0.01, 50))
    tau_fall = numpyro.sample('tau_fall', dist.Uniform(1, 300))
    # s_n = numpyro.sample('s_n', dist.Uniform(0, 3 * jnp.std(yerr)))
    s_n = numpyro.sample('s_n', dist.Uniform(0., 20.))

    flux_eq = numpyro.deterministic('flux_eq', flux_equation(x, 10. ** log_amplitude, beta, 10. ** log_gamma, t0,
                                                             tau_rise, tau_fall))
    # flux_eq = flux_equation(x, 10. ** log_amplitude, beta, 10. ** log_gamma, t0,
    #                         tau_rise, tau_fall)
    if y is not None and yerr is not None:
        numpyro.sample('obs', dist.Normal(flux_eq, yerr ** 2 + s_n ** 2), obs=y)


if __name__ == '__main__':
    # Get data
    file_name = 'PS1_PS1MD_PSc370330.snana.dat'
    xr, yr, yerr_r = read_data('r', filename=file_name)

    # Run NUTS
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model)

    mcmc = MCMC(kernel, num_warmup=15000, num_samples=5000, thinning=1, num_chains=3)
    mcmc.run(rng_key_, x=xr, y=yr, yerr=yerr_r, fobs_max=np.max(yr), time_max=xr[np.argmax(yr)])
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Plot sampling
    ds = az.from_numpyro(mcmc)
    labels = ["log_amplitude", "beta", "log_gamma", "t0", "tau_rise", "tau_fall", "s_n"]
    # az.plot_pair(ds, var_names=labels, divergences=True)
    # plt.show()

    # Plot trace
    # print(ds.sample_stats['diverging'])
    az.plot_trace(ds.posterior)
    plt.show()

    # Plot variable densities
    az.plot_density(ds.posterior, var_names=labels)
    plt.show()

    # Plot random samples
    rng_key, rng_key_ = random.split(rng_key)

    t = np.linspace(np.min(xr), np.max(xr), 2000)
    predictive = Predictive(model, posterior_samples=samples)
    predictions = predictive(rng_key_, x=t, y=None, yerr=None, fobs_max=np.max(yr), time_max=xr[np.argmax(yr)])

    plt.scatter(xr, yr)
    plt.plot(t, predictions['flux_eq'][0])
    plt.plot(t, predictions['flux_eq'][3])
    plt.plot(t, predictions['flux_eq'][10])
    plt.show()

    ''' 
    plt.scatter(xr, yr)
    plt.plot(xr, samples['flux_eq'][3], label='1')
    plt.plot(xr, samples['flux_eq'][100], label='2')
    plt.plot(xr, samples['flux_eq'][200], label='3')
    plt.plot(xr, samples['flux_eq'][300], label='4')
    plt.plot(xr, samples['flux_eq'][500], label='5')
    plt.plot(xr, samples['flux_eq'][1000], label='6')
    plt.legend()
    plt.show()
    '''

    # Plot first few samples
    n_samples = 100
    plt.scatter(xr, yr)
    plt.plot(xr, ds.posterior.flux_eq[0, :n_samples].T, color='k', alpha=0.1)
    plt.show()

    '''
    # Plot means
    plt.scatter(xr, yr)
    plt.plot(xr, flux_equation(xr, 10 ** jnp.mean(samples['log_amplitude']), jnp.mean(samples['beta']),
                               10. ** jnp.mean(samples['log_gamma']),
                               jnp.mean(samples['t0']), jnp.mean(samples['tau_rise']), jnp.mean(samples['tau_fall'])))
    plt.show()
    '''
