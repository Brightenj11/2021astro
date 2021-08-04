import numpyro
import numpyro.distributions as dist
import numpy as np
from numpyro.infer import NUTS, MCMC, Predictive
from jax import random
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import arviz as az

import matplotlib.pyplot as plt


def read_data(filter_name, filename: str = 'PS1_PS1MD_PSc370330.snana.dat'):
    """
    General function to read .snana.dat files and get specific filter data

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
            if entree[3] < 100.:
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
    return jnp.nan_to_num(amplitude * ((1. - beta * jnp.minimum(time - t0, gamma)) *
                                       jnp.exp(-(jnp.maximum(time - t0, gamma) - gamma) / tau_fall)) /
                          (1. + jnp.exp(-(time - t0) / tau_rise)))


def model(x, x2, x3, x4, y=None, y2=None, y3=None, y4=None, yerr=None, yerr2=None, yerr3=None, yerr4=None, fobs_max=200., time_max=56000.):
    # R band
    log_amplitude = numpyro.sample('log_amplitude', dist.Uniform(jnp.log10(1.), jnp.log10(100. * fobs_max)))
    beta = numpyro.sample('beta', dist.Uniform(0., 0.01))
    log_gamma = numpyro.sample('log_gamma', dist.Uniform(-1., 3.))
    t0 = numpyro.sample('t0', dist.Uniform(time_max - 50., time_max + 300.))
    tau_rise = numpyro.sample('tau_rise', dist.Uniform(0.01, 50.))
    tau_fall = numpyro.sample('tau_fall', dist.Uniform(1., 300.))
    s_n = numpyro.sample('s_n', dist.HalfNormal(jnp.std(yerr)))

    # Sample G band
    scale_a = numpyro.sample('scale_a', dist.TruncatedNormal(0., 1., .5))
    scale_b = numpyro.sample('scale_b', dist.TruncatedNormal(0., 1., .5))
    scale_g = numpyro.sample('scale_g', dist.TruncatedNormal(0., 1., .5))
    scale_tr = numpyro.sample('scale_tr', dist.TruncatedNormal(0., 1., .5))
    scale_tf = numpyro.sample('scale_tf', dist.TruncatedNormal(0., 1., .5))
    g_sn = numpyro.sample('g_sn', dist.HalfNormal(jnp.std(yerr2)))
    # Scale G band
    g_amp = log_amplitude + jnp.log10(scale_a)
    g_beta = beta * scale_b
    g_gamma = log_gamma + jnp.log10(scale_g)
    g_tr = tau_rise * scale_tr
    g_tf = tau_fall * scale_tf

    # I band
    i_scale_a = numpyro.sample('i_scale_a', dist.TruncatedNormal(0., 1., .5))
    i_scale_b = numpyro.sample('i_scale_b', dist.TruncatedNormal(0., 1., .5))
    i_scale_g = numpyro.sample('i_scale_g', dist.TruncatedNormal(0., 1., .5))
    i_scale_tr = numpyro.sample('i_scale_tr', dist.TruncatedNormal(0., 1., .5))
    i_scale_tf = numpyro.sample('i_scale_tf', dist.TruncatedNormal(0., 1., .5))
    i_sn = numpyro.sample('i_sn', dist.HalfNormal(jnp.std(yerr3)))

    i_amp = log_amplitude + jnp.log10(i_scale_a)
    i_beta = beta * i_scale_b
    i_gamma = log_gamma + jnp.log10(i_scale_g)
    i_tr = tau_rise * i_scale_tr
    i_tf = tau_fall * i_scale_tf

    # Z band
    z_scale_a = numpyro.sample('z_scale_a', dist.TruncatedNormal(0., 1., .5))
    z_scale_b = numpyro.sample('z_scale_b', dist.TruncatedNormal(0., 1., .5))
    z_scale_g = numpyro.sample('z_scale_g', dist.TruncatedNormal(0., 1., .5))
    z_scale_tr = numpyro.sample('z_scale_tr', dist.TruncatedNormal(0., 1., .5))
    z_scale_tf = numpyro.sample('z_scale_tf', dist.TruncatedNormal(0., 1., .5))
    z_sn = numpyro.sample('z_sn', dist.HalfNormal(jnp.std(yerr4)))

    z_amp = i_amp + jnp.log10(z_scale_a)
    z_beta = i_beta * z_scale_b
    z_gamma = i_gamma + jnp.log10(z_scale_g)
    z_tr = i_tr * z_scale_tr
    z_tf = i_tf * z_scale_tf

    # Calculate flux values
    flux_eq = numpyro.deterministic('flux_eq', flux_equation(x, 10. ** log_amplitude, beta, 10. ** log_gamma, t0,
                                                             tau_rise, tau_fall))
    flux_eq_g = numpyro.deterministic('flux_eq_g', flux_equation(x2, 10. ** g_amp, g_beta, 10. ** g_gamma, t0,
                                                                 g_tr, g_tf))
    flux_eq_i = numpyro.deterministic('flux_eq_i', flux_equation(x3, 10. ** i_amp, i_beta, 10. ** i_gamma, t0,
                                                                 i_tr, i_tf))
    flux_eq_z = numpyro.deterministic('flux_eq_z', flux_equation(x4, 10. ** z_amp, z_beta, 10. ** z_gamma, t0,
                                                                 z_tr, z_tf))
    '''
    flux_eq = flux_equation(x, 10. ** log_amplitude, beta, 10. ** log_gamma, t0,
                            tau_rise, tau_fall)
    flux_eq_g = flux_equation(x2, 10. ** g_amp, g_beta, 10. ** g_gamma, t0,
                              g_tr, g_tf)
    flux_eq_i = flux_equation(x3, 10. ** i_amp, i_beta, 10. ** i_gamma, t0,
                              i_tr, i_tf)
    flux_eq_z = flux_equation(x4, 10. ** z_amp, z_beta, 10. ** z_gamma, t0,
                              z_tr, z_tf)
    ''' 
    if y is not None and yerr is not None:
        numpyro.sample('obs', dist.Normal(flux_eq, yerr ** 2. + s_n ** 2.), obs=y)
        numpyro.sample('obs2', dist.Normal(flux_eq_g, yerr2 ** 2. + g_sn ** 2.), obs=y2)
        numpyro.sample('obs3', dist.Normal(flux_eq_i, yerr3 ** 2. + i_sn ** 2.), obs=y3)
        numpyro.sample('obs4', dist.Normal(flux_eq_z, yerr4 ** 2. + z_sn ** 2.), obs=y4)


if __name__ == '__main__':
    # Get data
    file_name = 'PS1_PS1MD_PSc370330.snana.dat'
    xr, yr, yerr_r = read_data('r', filename=file_name)
    xg, yg, yerr_g = read_data('g', filename=file_name)
    xi, yi, yerr_i = read_data('i', filename=file_name)
    xz, yz, yerr_z = read_data('z', filename=file_name)

    # Run NUTS
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model)

    mcmc = MCMC(kernel, num_warmup=15000, num_samples=10000, thinning=1, num_chains=3)
    mcmc.run(rng_key_, x=xr, x2=xg, x3=xi, x4=xz, y=yr, y2=yg, y3=yi, y4=yz, yerr=yerr_r, yerr2=yerr_g, yerr3=yerr_i,
             yerr4=yerr_z, fobs_max=np.max(yr), time_max=xr[np.argmax(yr)])
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Plot sampling
    ds = az.from_numpyro(mcmc)
    labels = ["log_amplitude", "beta", "log_gamma", "t0", "tau_rise", "tau_fall", "s_n"]
    # az.plot_pair(ds, var_names=labels, divergences=True)
    # plt.show()

    # Plot trace
    # print(ds.sample_stats['diverging'])
    az.rcParams['plot.max_subplots'] = 25
    az.plot_trace(ds.posterior)
    plt.show()

    # Plot variable densities
    az.plot_posterior(ds.posterior, var_names=labels)
    plt.show()

    # Plot random samples
    rng_key, rng_key_ = random.split(rng_key)

    t = np.linspace(np.min(xr), np.max(xr), 3000)
    predictive = Predictive(model, posterior_samples=samples)
    predictions = predictive(rng_key_, x=t, x2=t, x3=t, x4=t, yerr=yerr_r, yerr2=yerr_g, yerr3=yerr_i, yerr4=yerr_z,
                             fobs_max=np.max(yr), time_max=xr[np.argmax(yr)])
    ''' 
    plt.scatter(xr, yr)
    plt.plot(t, predictions['flux_eq'][0])
    plt.plot(t, predictions['flux_eq'][3])
    plt.plot(t, predictions['flux_eq'][10])
    plt.show()

     
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
    n_samples = 50
    plt.scatter(xr, yr, color='r')
    plt.plot(xr, ds.posterior.flux_eq[0, :n_samples].T, color='r', alpha=0.1)

    plt.scatter(xg, yg, color='g')
    plt.plot(xg, ds.posterior.flux_eq_g[0, :n_samples].T, color='g', alpha=0.1)

    plt.scatter(xi, yi, color='b')
    plt.plot(xi, ds.posterior.flux_eq_i[0, :n_samples].T, color='b', alpha=0.1)

    plt.scatter(xz, yz, color='y')
    plt.plot(xz, ds.posterior.flux_eq_z[0, :n_samples].T, color='y', alpha=0.1)
    plt.xlim([55900, 56100])
    plt.show()

    # Plot sample with extra times using Prediction
    plt.scatter(xr, yr, color='r')
    for i in range(n_samples):
        plt.plot(t, predictions['flux_eq'][i], color='r', alpha=0.1)

    plt.scatter(xg, yg, color='g')
    for i in range(n_samples):
        plt.plot(t, predictions['flux_eq_g'][i], color='g', alpha=0.1)

    plt.scatter(xi, yi, color='b')
    for i in range(n_samples):
        plt.plot(t, predictions['flux_eq_i'][i], color='b', alpha=0.1)

    plt.scatter(xz, yz, color='y')
    for i in range(n_samples):
        plt.plot(t, predictions['flux_eq_z'][i], color='y', alpha=0.1)
    plt.xlim([55900, 56100])
    plt.show()
    '''
    # Plot means
    plt.scatter(xr, yr)
    plt.plot(xr, flux_equation(xr, 10 ** jnp.mean(samples['log_amplitude']), jnp.mean(samples['beta']),
                               10. ** jnp.mean(samples['log_gamma']),
                               jnp.mean(samples['t0']), jnp.mean(samples['tau_rise']), jnp.mean(samples['tau_fall'])))
    plt.show()
    '''
