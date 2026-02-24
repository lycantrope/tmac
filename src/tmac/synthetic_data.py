import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy import signal, stats

import tmac.fourier as tfo


def softplus(x, beta: float = 50) -> jax.Array:
    return jnp.log(1 + jnp.exp(beta * x)) / beta


def generate_synthetic_data(
    num_ind: int,
    num_neurons: int,
    mean_r: float,
    mean_g: float,
    variance_noise_r: float,
    variance_noise_g: float,
    variance_a: float,
    variance_m: float,
    tau_a: float,
    tau_m: float,
    frac_nan: float = 0.0,
    beta: float = 20,
    multiplicative: bool = False,
    rng_seed: int = 42,
):
    """Function that generates synthetic two channel imaging data

    Args:
        num_ind: number of measurements in time
        num_neurons: number of neurons recorded
        mean_r: mean fluoresence of the red channel
        mean_g: mean fluoresence of the green channel
        variance_noise_r: variance of the gaussian noise in the red channel
        variance_noise_g: variance of the gaussian noise in the green channel
        variance_a: variance of the calcium activity
        variance_m: variance of the motion artifact
        tau_a: timescale of the calcium activity
        tau_m: timescale of the motion artifact
        frac_nan: fraction of time points to set to NaN
        beta: parameter of softplus to keep values from going negative
        multiplicative: generate data assuming a multiplicative interaction between a and m
                        if false, generate data with additive interaction
    Returns:
        red_bleached: synthetic red channel data (motion + noise)
        green_bleached: synthetic green channel data (activity + motion + noise)
        a: activity Gaussian process
        m: motion artifact Gaussian process
    """
    key = random.key(jnp.array(rng_seed))
    key, *subkeys = random.split(key, num=6)
    fourier_basis, frequency_vec = tfo.get_fourier_basis(num_ind)

    # get the diagonal of radial basis kernel in fourier space
    c_diag_a = (
        variance_a
        * tau_a
        * jnp.sqrt(2 * jnp.pi)
        * jnp.exp(-0.5 * frequency_vec**2 * tau_a**2)
    )
    c_diag_m = (
        variance_m
        * tau_m
        * jnp.sqrt(2 * jnp.pi)
        * jnp.exp(-0.5 * frequency_vec**2 * tau_m**2)
    )

    a = fourier_basis @ (
        jnp.sqrt(c_diag_a[:, None])
        * random.normal(key=subkeys[0], shape=(num_ind, num_neurons))
    )
    m = fourier_basis @ (
        jnp.sqrt(c_diag_m[:, None])
        * random.normal(key=subkeys[1], shape=(num_ind, num_neurons))
    )

    # keep a and m from being negative for the multiplicative model
    # a has mean 1, m has mean 0
    a = softplus(a + 1, beta=beta)
    m = softplus(m + 1, beta=beta) - 1

    noise_r = jnp.sqrt(variance_noise_r) * random.normal(
        key=subkeys[2], shape=(num_ind, num_neurons)
    )
    noise_g = jnp.sqrt(variance_noise_g) * random.normal(
        key=subkeys[3], shape=(num_ind, num_neurons)
    )

    red_true = mean_r * softplus(m + 1 + noise_r, beta=beta)

    if multiplicative:
        green_true = mean_g * softplus(a * (m + 1) + noise_g, beta=beta)
    else:
        green_true = mean_g * softplus(a + m + noise_g, beta=beta)

    # add photobleaching
    photo_tau = num_ind / 3
    red_bleached = red_true * jnp.exp(-jnp.arange(num_ind)[:, None] / photo_tau)
    green_bleached = green_true * jnp.exp(-jnp.arange(num_ind)[:, None] / photo_tau)

    # nan a few values
    ind_to_nan = random.uniform(subkeys[4], shape=(num_ind,)) <= frac_nan
    red_bleached.at[ind_to_nan, :].set(jnp.nan)
    green_bleached.at[ind_to_nan, :].set(jnp.nan)

    return red_bleached, green_bleached, a, m


def col_corr(a_true, a_hat):
    """Calculate pearson correlation coefficient between each column of a_true and a_hat"""
    corr = jnp.zeros(a_true.shape[1])

    for c in range(a_true.shape[1]):
        true_vec = a_true[:, c] - jnp.mean(a_true[:, c])
        hat_vec = a_hat[:, c] - jnp.mean(a_hat[:, c])
        corr.at[c].set(
            jnp.mean(true_vec * hat_vec) / jnp.std(true_vec) / jnp.std(hat_vec)
        )

    return corr


def ratio_model(red, green, tau):
    # calculate the prediction from the ratio model
    # assumes red
    red = red / np.mean(red, axis=0)
    green = green / np.mean(green, axis=0)

    num_std = 3
    num_filter_ind = np.round(tau * num_std) * 2 + 1
    filter_x = np.arange(num_filter_ind) - (num_filter_ind - 1) / 2
    filter_shape = stats.norm.pdf(filter_x / tau) / tau
    green_filtered = signal.convolve2d(green, filter_shape[:, None], "same")
    red_filtered = signal.convolve2d(red, filter_shape[:, None], "same")
    ratio = green_filtered / red_filtered - 1

    return ratio
