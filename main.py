import numpy as np
import matplotlib.pyplot as plt


def flux(amplitude, beta, gamma, t0, tau_rise, tau_fall):
    """
    Plot the flux of a transient
    :param amplitude: Amplitude (flux)
    :param beta: Plateau Slope (days^-1)
    :param gamma: Plateau Duration (days)
    :param t0: Plateau Duration (days)
    :param tau_rise: Rise Time (days)
    :param tau_fall: Fall Time (days)
    """
    time = np.arange(0, 200)
    plt.plot(time, amplitude * ((1 - beta * np.minimum(time - t0, gamma)) *
                                np.exp(-(np.maximum(time - t0, gamma) - gamma) / tau_fall)) /
             (1 + np.exp(-(time - t0) / tau_rise)))
    plt.show()


if __name__ == '__main__':
    flux(10 ** 3, 0.005, 50, 100, 25, 10)
