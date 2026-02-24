import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import torch
from jax.scipy import optimize

# from scipy import optimize
import tmac.fourier as tfo
import tmac.optimization as opt
import tmac.preprocessing as pp
import tmac.probability_distributions as tpd


def tmac_ac(
    red_np,
    green_np,
    optimizer="BFGS",
    verbose=False,
    truncate_freq=True,
):
    """Implementation of the Two-channel motion artifact correction method (TMAC)

    This is tmac_ac because it is the additive and circular boundary version
    This code takes in imaging fluoresence data from two simultaneously recorded channels and attempts to remove
    shared motion artifacts between the two channels

    Args:
        red_np: numpy array, [time, neurons], activity independent channel
        green_np: numpy array, [time, neurons], activity dependent channel
        optimizer: string, scipy optimizer
        verbose: boolean, if true, outputs when inference is complete on each neuron and estimates time to finish
        truncate_freq: boolean, if true truncates low amplitude frequencies in Fourier domain. This should give the same
            results but may give sensitivity to the initial conditions

    Returns: a dictionary containing: all the inferred parameters of the model
    """

    # optimization is performed using Scipy optimize, so all tensors should stay on the CPU

    red = pp.check_input_format(red_np)
    green = pp.check_input_format(green_np)

    assert jnp.all(jnp.isfinite(red)) and jnp.all(
        jnp.isfinite(green)
    ), "Input data cannot have any nan or inf"
    assert red.shape == green.shape, "red and green matricies must be the same shape"

    T, C = red.shape
    # convert data to units of fold mean and subtract mean
    mean_red = jnp.mean(red, axis=0)
    mean_green = jnp.mean(green, axis=0)
    red.at[::].divide(mean_red)
    red.at[::].subtract(1)
    green.at[::].divide(mean_green)
    green.at[::].subtract(1)

    # convert to tensors and fourier transform
    red_fft = tfo.real_fft(red)
    green_fft = tfo.real_fft(green)

    # estimate all model parameters from the data
    variance_r_noise_init = jnp.var(red, axis=0)
    variance_g_noise_init = jnp.var(green, axis=0)
    variance_a_init = jnp.var(green, axis=0)
    variance_m_init = jnp.var(red, axis=0)

    # initialize length scale
    length_scale_a_init = jnp.ones(C)
    length_scale_m_init = jnp.ones(C)

    # preallocate space for all the training variables
    a_trained = jnp.zeros((T, C))
    m_trained = jnp.zeros((T, C))

    variance_r_noise_trained = jnp.zeros(C)
    variance_g_noise_trained = jnp.zeros(C)
    variance_a_trained = jnp.zeros(C)
    length_scale_a_trained = jnp.zeros(C)
    variance_m_trained = jnp.zeros(C)
    length_scale_m_trained = jnp.zeros(C)

    # loop through each neuron and perform inference
    start = time.time()
    freq = tfo.get_fourier_freq(T)

    for n in range(C):
        # get the initial values for the hyperparameters of this neuron
        # All hyperparameters are positive, so we fit them in log space
        evidence_training_variables = jnp.log(
            jnp.array(
                [
                    variance_r_noise_init[n],
                    variance_g_noise_init[n],
                    variance_a_init[n],
                    length_scale_a_init[n],
                    variance_m_init[n],
                    length_scale_m_init[n],
                ]
            )
        )

        # define the evidence loss function. This function takes in and returns pytorch tensors
        def evidence_loss_fn(
            training_variables,
            red,
            red_fft,
            green,
            green_fft,
            freq,
        ):
            return -tpd.tmac_evidence(
                red,
                red_fft,
                training_variables[0],
                green,
                green_fft,
                training_variables[1],
                training_variables[2],
                training_variables[3],
                training_variables[4],
                training_variables[5],
                freq=freq,
                truncate_freq=truncate_freq,
            )

        trained_variances = optimize.minimize(
            evidence_loss_fn,
            evidence_training_variables,
            args=(red[:, n], red_fft[:, n], green[:, n], green_fft[:, n], freq),
            method=optimizer,
        )

        # calculate the posterior values
        # The posterior is gaussian so we don't need to optimize, we find a and m in one step
        a, m = tpd.tmac_posterior(
            red[:, n],
            red_fft[:, n],
            trained_variances.x[0],
            green[:, n],
            green_fft[:, n],
            trained_variances.x[1],
            trained_variances.x[2],
            trained_variances.x[3],
            trained_variances.x[4],
            trained_variances.x[5],
            truncate_freq=truncate_freq,
        )

        a_trained.at[:, n].set(a)
        m_trained.at[:, n].set(m)
        variance_r_noise_trained.at[n].set(jnp.exp(trained_variances.x[0]))
        variance_g_noise_trained.at[n].set(jnp.exp(trained_variances.x[1]))
        variance_a_trained.at[n].set(jnp.exp(trained_variances.x[2]))
        length_scale_a_trained.at[n].set(jnp.exp(trained_variances.x[3]))
        variance_m_trained.at[n].set(jnp.exp(trained_variances.x[4]))
        length_scale_m_trained.at[n].set(jnp.exp(trained_variances.x[5]))

        if verbose:
            decimals = 1e3
            # print out timing
            elapsed = time.time() - start
            remaining = elapsed / (n + 1) * (red.shape[1] - (n + 1))
            elapsed_truncated = np.round(elapsed * decimals) / decimals
            remaining_truncated = np.round(remaining * decimals) / decimals
            print(str(n + 1) + "/" + str(red.shape[1]) + " neurons complete")
            print(
                str(elapsed_truncated)
                + "s elapsed, estimated "
                + str(remaining_truncated)
                + "s remaining"
            )

    trained_variables = {
        "a": a_trained,
        "m": m_trained,
        "variance_r_noise": variance_r_noise_trained,
        "variance_g_noise": variance_g_noise_trained,
        "variance_a": variance_a_trained,
        "length_scale_a": length_scale_a_trained,
        "variance_m": variance_m_trained,
        "length_scale_m": length_scale_m_trained,
    }

    return trained_variables


def initialize_length_scale(y: jax.Array):
    """Function to fit a Gaussian to the autocorrelation of y

    Args:
        y: numpy vector

    Returns: Standard deviation of a Gaussian fit to the autocorrelation of y
    """
    y_len = y.shape[0]
    x = jnp.arange(-y_len / 2, y_len / 2) + 0.5
    y_z_score = (y - jnp.mean(y)) / jnp.std(y)
    y_corr = jnp.correlate(y_z_score, y_z_score, mode="same")

    # fit the std of a gaussian to the correlation function
    @partial(jax.jit)
    def fit_to_gaussian(p: jax.Array):
        return p[0] * jsp.stats.norm.pdf(x, 0, p[1]) - y_corr

    p_init = jnp.array((jnp.max(y_corr), 1.0))

    p_hat = optimize.minimize(fit_to_gaussian, p_init, method="BFGS")[0]

    # return the standard deviation
    return p_hat[1]
