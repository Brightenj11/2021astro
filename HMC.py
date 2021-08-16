import os
import numpyro
import numpyro.distributions as dist
import numpy as np
from numpyro.infer import NUTS, MCMC, Predictive
from jax import random
import jax.numpy as jnp
from jax.config import config
import arviz as az
import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)


def read_data(filter_name, filename: str = 'PS1_PS1MD_PSc370330.snana.dat'):
    """
    General function to read .snana.dat files and get specific filter data

    :param filter_name: Gets data from this filter character
    :param filename:
    :return: times (x), flux values (y), and flux errors (yerr)
    """
    data = np.genfromtxt(os.getcwd() + '/ps1_sne_zenodo/ps1_sne_zenodo/' + filename, dtype=None, skip_header=17,
                         skip_footer=1, usecols=(1, 2, 4, 5), encoding=None)

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


def model(x1, x2, x3, x4, y1=None, y2=None, y3=None, y4=None, yerr1=None, yerr2=None, yerr3=None, yerr4=None,
          fobs_max=200., time_max=56000.):
    # R band
    r_amp = numpyro.sample('r_amp', dist.Uniform(jnp.log10(1.), jnp.log10(100. * fobs_max)))  # Log Amplitude
    r_beta = numpyro.sample('r_beta', dist.Uniform(0., 0.01))  # Plateau Slope
    r_gamma = numpyro.sample('r_gamma', dist.Uniform(-1., 3.))  # Log Plateau Duration
    t0 = numpyro.sample('t0', dist.Uniform(time_max - 50., time_max + 300.))  # Reference Epoch
    r_tr = numpyro.sample('r_tr', dist.Uniform(0.01, 50.))  # Rise Time
    r_tf = numpyro.sample('r_tf', dist.Uniform(1., 300.))  # Fall time
    r_sn = numpyro.sample('r_sn', dist.HalfNormal(jnp.std(yerr1)))  # Intrinsic Scatter

    # Sample G band
    g_scale_a = numpyro.sample('g_scale_a', dist.TruncatedDistribution(base_dist=dist.Normal(0., 0.25), low=None,
                                                                       high=jnp.log10(2)))
    g_scale_b = numpyro.sample('g_scale_b', dist.Normal(0., 0.25))
    g_scale_g = numpyro.sample('g_scale_g', dist.Normal(0., 0.25))
    g_scale_tr = numpyro.sample('g_scale_tr', dist.Normal(0., 0.25))
    g_scale_tf = numpyro.sample('g_scale_tf', dist.Normal(0., 0.25))
    g_sn = numpyro.sample('g_sn', dist.HalfNormal(jnp.std(yerr2)))

    # Scale G band
    g_amp = r_amp + g_scale_a
    g_beta = r_beta * 10. ** g_scale_b
    g_gamma = r_gamma + g_scale_g
    g_tr = r_tr * 10. ** g_scale_tr
    g_tf = r_tf * 10. ** g_scale_tf

    # I band
    i_scale_a = numpyro.sample('i_scale_a', dist.TruncatedDistribution(base_dist=dist.Normal(0., 0.25), low=None,
                                                                       high=jnp.log10(2)))
    i_scale_b = numpyro.sample('i_scale_b', dist.Normal(0., 0.25))
    i_scale_g = numpyro.sample('i_scale_g', dist.Normal(0., 0.25))
    i_scale_tr = numpyro.sample('i_scale_tr', dist.Normal(0., 0.25))
    i_scale_tf = numpyro.sample('i_scale_tf', dist.Normal(0., 0.25))
    i_sn = numpyro.sample('i_sn', dist.HalfNormal(jnp.std(yerr3)))

    i_amp = r_amp + i_scale_a
    i_beta = r_beta * 10. ** i_scale_b
    i_gamma = r_gamma + i_scale_g
    i_tr = r_tr * 10. ** i_scale_tr
    i_tf = r_tf * 10. ** i_scale_tf

    # Z band
    z_scale_a = numpyro.sample('z_scale_a', dist.TruncatedDistribution(base_dist=dist.Normal(0., 0.25), low=None,
                                                                       high=jnp.log10(2)))
    z_scale_b = numpyro.sample('z_scale_b', dist.Normal(0., 0.25))
    z_scale_g = numpyro.sample('z_scale_g', dist.Normal(0., 0.25))
    z_scale_tr = numpyro.sample('z_scale_tr', dist.Normal(0., 0.25))
    z_scale_tf = numpyro.sample('z_scale_tf', dist.Normal(0., 0.25))
    z_sn = numpyro.sample('z_sn', dist.HalfNormal(jnp.std(yerr4)))

    z_amp = i_amp + z_scale_a
    z_beta = i_beta * 10. ** z_scale_b
    z_gamma = i_gamma + z_scale_g
    z_tr = i_tr * 10. ** z_scale_tr
    z_tf = i_tf * 10. ** z_scale_tf

    # Calculate flux values
    flux_eq_r = numpyro.deterministic('flux_eq_r', flux_equation(x1, 10. ** r_amp, r_beta, 10. ** r_gamma, t0,
                                                                 r_tr, r_tf))
    flux_eq_g = numpyro.deterministic('flux_eq_g', flux_equation(x2, 10. ** g_amp, g_beta, 10. ** g_gamma, t0,
                                                                 g_tr, g_tf))
    flux_eq_i = numpyro.deterministic('flux_eq_i', flux_equation(x3, 10. ** i_amp, i_beta, 10. ** i_gamma, t0,
                                                                 i_tr, i_tf))
    flux_eq_z = numpyro.deterministic('flux_eq_z', flux_equation(x4, 10. ** z_amp, z_beta, 10. ** z_gamma, t0,
                                                                 z_tr, z_tf))

    if y1 is not None and yerr1 is not None:
        numpyro.sample('obs1', dist.Normal(flux_eq_r, jnp.sqrt(yerr1 ** 2. + r_sn ** 2.)), obs=y1)
        numpyro.sample('obs2', dist.Normal(flux_eq_g, jnp.sqrt(yerr2 ** 2. + g_sn ** 2.)), obs=y2)
        numpyro.sample('obs3', dist.Normal(flux_eq_i, jnp.sqrt(yerr3 ** 2. + i_sn ** 2.)), obs=y3)
        numpyro.sample('obs4', dist.Normal(flux_eq_z, jnp.sqrt(yerr4 ** 2. + z_sn ** 2.)), obs=y4)


if __name__ == '__main__':
    # Get data
    number = '330064'
    file_name = 'PS1_PS1MD_PSc' + number + '.snana.dat'
    to_plot = False
    xr, yr, yerr_r = read_data('r', filename=file_name)
    xg, yg, yerr_g = read_data('g', filename=file_name)
    xi, yi, yerr_i = read_data('i', filename=file_name)
    xz, yz, yerr_z = read_data('z', filename=file_name)

    # Run NUTS
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model)

    mcmc = MCMC(kernel, num_warmup=80000, num_samples=10000, thinning=1, num_chains=1)
    mcmc.run(rng_key_, x1=xr, x2=xg, x3=xi, x4=xz, y1=yr, y2=yg, y3=yi, y4=yz, yerr1=yerr_r, yerr2=yerr_g, yerr3=yerr_i,
             yerr4=yerr_z, fobs_max=np.max(yr), time_max=xr[np.argmax(yr)])
    mcmc.print_summary()
    samples = mcmc.get_samples()
    ds = az.from_numpyro(mcmc)
    if to_plot:
        # Plot data
        plt.scatter(xr, yr, color='r', label='R band')
        plt.scatter(xg, yg, color='g', label='G Band')
        plt.scatter(xi, yi, color='b', label='I band')
        plt.scatter(xz, yz, color='y', label='Z band')
        plt.legend()
        plt.show()

        # Plot sampling
        labels = ["r_amp", "r_beta", "r_gamma", "t0", "r_tr", "r_tf", "r_sn", "g_scale_a", "g_scale_b",
                  "g_scale_g", "g_scale_tr", "g_scale_tf", "g_sn", "i_scale_a", "i_scale_b", "i_scale_g", "i_scale_tr",
                  "i_scale_tf", "i_sn", "z_scale_a", "z_scale_b", "z_scale_g", "z_scale_tr", "z_scale_tf", "z_sn"]
        # az.plot_pair(ds, var_names=labels, divergences=True)
        # plt.show()

        # Plot trace
        # print(ds.sample_stats['diverging'])
        az.rcParams['plot.max_subplots'] = 100
        az.rcParams.update()
        az.plot_trace(ds.posterior)
        plt.show()

        # Plot variable densities
        az.plot_posterior(ds.posterior, var_names=labels)
        plt.show()

        # Predict random samples
        rng_key, rng_key_ = random.split(rng_key)

        t = np.linspace(np.min(xr), np.max(xr), 3000)
        predictive = Predictive(model, posterior_samples=samples)
        predictions = predictive(rng_key_, x1=t, x2=t, x3=t, x4=t, yerr1=yerr_r, yerr2=yerr_g, yerr3=yerr_i,
                                 yerr4=yerr_z, fobs_max=np.max(yr), time_max=xr[np.argmax(yr)])

        # Plot first few samples
        n_samples = 50
        plt.scatter(xr, yr, color='r')
        plt.plot(xr, ds.posterior.flux_eq_r[0, :n_samples].T, color='r', alpha=0.1)

        plt.scatter(xg, yg, color='g')
        plt.plot(xg, ds.posterior.flux_eq_g[0, :n_samples].T, color='g', alpha=0.1)

        plt.scatter(xi, yi, color='b')
        plt.plot(xi, ds.posterior.flux_eq_i[0, :n_samples].T, color='b', alpha=0.1)

        plt.scatter(xz, yz, color='y')
        plt.plot(xz, ds.posterior.flux_eq_z[0, :n_samples].T, color='y', alpha=0.1)
        plt.xlim([xr[np.argmax(yr)] - 100, xr[np.argmax(yr)] + 100])
        plt.show()

        # Plot sample with extra times using Prediction
        plt.scatter(xr, yr, color='r')
        for i in range(n_samples):
            plt.plot(t, predictions['flux_eq_r'][i], color='r', alpha=0.1)

        plt.scatter(xg, yg, color='g')
        for i in range(n_samples):
            plt.plot(t, predictions['flux_eq_g'][i], color='g', alpha=0.1)

        plt.scatter(xi, yi, color='b')
        for i in range(n_samples):
            plt.plot(t, predictions['flux_eq_i'][i], color='b', alpha=0.1)

        plt.scatter(xz, yz, color='y')
        for i in range(n_samples):
            plt.plot(t, predictions['flux_eq_z'][i], color='y', alpha=0.1)
        plt.xlim([xr[np.argmax(yr)] - 100, xr[np.argmax(yr)] + 100])
        plt.show()

    np.savez(number+'.npz', r_amp=ds.posterior.r_amp, r_beta=ds.posterior.r_beta, r_gamma=ds.posterior.r_gamma,
             t0=ds.posterior.t0, r_tr=ds.posterior.r_tr, r_tf=ds.posterior.r_tf, r_sn=ds.posterior.r_sn,
             g_scale_a=ds.posterior.g_scale_a, g_scale_b=ds.posterior.g_scale_b, g_scale_g=ds.posterior.g_scale_g,
             g_scale_tr=ds.posterior.g_scale_tr, g_scale_tf=ds.posterior.g_scale_tf, g_sn=ds.posterior.g_sn,
             i_scale_a=ds.posterior.i_scale_a, i_scale_b=ds.posterior.i_scale_b, i_scale_g=ds.posterior.i_scale_g,
             i_scale_tr=ds.posterior.i_scale_tr, i_scale_tf=ds.posterior.i_scale_tf, i_sn=ds.posterior.i_sn,
             z_scale_a=ds.posterior.z_scale_a, z_scale_b=ds.posterior.z_scale_b, z_scale_g=ds.posterior.z_scale_g,
             z_scale_tr=ds.posterior.z_scale_tr, z_scale_tf=ds.posterior.z_scale_tf, z_sn=ds.posterior.z_sn)

    # with np.load('000001.npz') as data:
    #    a = data['r_amp']
    #    b = data['r_beta']
    #    g = data['r_gamma']
