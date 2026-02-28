from functools import partial
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from jax.scipy import optimize as joptimize
from jax.scipy import stats as jstats

import tmac.fourier as tfo
import tmac.preprocessing as pp
import tmac.probability_distributions as tpd


@partial(jax.jit, static_argnames=("truncate_freq"))
def tmac_ac(
    red_np: Union[np.ndarray, jax.Array],
    green_np: Union[np.ndarray, jax.Array],
    truncate_freq: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Implementation of the Two-channel motion artifact correction method (TMAC)

    This is tmac_ac because it is the additive and circular boundary version
    This code takes in imaging fluoresence data from two simultaneously recorded channels and attempts to remove
    shared motion artifacts between the two channels

    Args:
        red_np: numpy array, [time, neurons], activity independent channel
        green_np: numpy array, [time, neurons], activity dependent channel
        truncate_freq: boolean, if true truncates low amplitude frequencies in Fourier domain. This should give the same
            results but may give sensitivity to the initial conditions

    Returns: a_trained, m_trained, trained_params
    """

    # optimization is performed using Scipy optimize, so all tensors should stay on the CPU

    # assert np.all(np.isfinite(red_np)) & np.all(
    #     np.isfinite(green_np)
    # ), "Input data cannot have any nan or inf"
    # assert (
    #     red_np.shape == green_np.shape
    # ), "red and green matricies must be the same shape"

    red = pp.check_input_format(red_np)
    green = pp.check_input_format(green_np)
    dtype = red.dtype
    T, C = red.shape
    # convert data to units of fold mean and subtract mean
    mean_red = jnp.mean(red, axis=0)
    mean_green = jnp.mean(green, axis=0)
    red = (red / mean_red) - 1.0
    green = (green / mean_green) - 1.0

    # convert to tensors and fourier transform
    red_fft = tfo.real_fft(red)
    green_fft = tfo.real_fft(green)

    # Parameters per neurons
    init_parameters = jnp.stack(
        [
            jnp.var(red, axis=0),  # var_r_noise
            jnp.var(green, axis=0),  # var_g_noise
            jnp.var(green, axis=0),  # var_a
            jnp.ones(C, dtype=dtype),  # scale_a
            jnp.var(red, axis=0),  # var_m,
            jnp.ones(C, dtype=dtype),  # scale_m,
        ],
        axis=0,
    )

    # define the evidence loss function. This function takes in and returns pytorch tensors
    def evidence_loss_fn(
        training_variables,
        red,
        red_fft,
        green,
        green_fft,
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
            truncate_freq=truncate_freq,
        )

    @partial(jax.vmap, in_axes=1, out_axes=1)
    def _tmac_estimate_all(init_variances, r, r_fft, g, g_fft):
        trained_variances = joptimize.minimize(
            evidence_loss_fn,
            jnp.log(init_variances),
            args=(r, r_fft, g, g_fft),
            method="BFGS",
        )
        return jnp.exp(trained_variances.x)

    # loop through each neuron and perform inference

    # calculate the posterior values
    # The posterior is gaussian so we don't need to optimize, we find a and m in one step
    @partial(jax.vmap, in_axes=1, out_axes=-1)
    def _tmac_posterior_all(trained_variances, r, r_fft, g, g_fft):
        trained_variances = jnp.log(trained_variances)
        return tpd.tmac_posterior(
            r,
            r_fft,
            trained_variances[0],
            g,
            g_fft,
            trained_variances[1],
            trained_variances[2],
            trained_variances[3],
            trained_variances[4],
            trained_variances[5],
            truncate_freq=truncate_freq,
        )

    trained_params = _tmac_estimate_all(init_parameters, red, red_fft, green, green_fft)
    a_trained, m_trained = _tmac_posterior_all(
        trained_params, red, red_fft, green, green_fft
    )
    return (
        a_trained,
        m_trained,
        trained_params,
    )


@jax.jit
def initialize_length_scale(y: jax.Array) -> float:
    """Function to fit a Gaussian to the autocorrelation of y

    Args:
        y: jax vector

    Returns: Standard deviation of a Gaussian fit to the autocorrelation of y
    """
    y_len = y.shape[0]
    x = jnp.arange(-y_len / 2, y_len / 2) + 0.5
    y_z_score = (y - jnp.mean(y)) / jnp.std(y)
    y_corr = jnp.correlate(y_z_score, y_z_score, mode="same")

    p_init = jnp.array((jnp.max(y_corr), 1.0))

    def gaussian_residuals(p, x, y_corr):
        """
        p[0]: Amplitude (Scale)
        p[1]: Standard Deviation (Sigma)
        """
        # Using jax.scipy.stats ensures Autograd can calculate the derivative
        amplitude = p[0]
        sigma = jnp.abs(p[1]) + 1e-6
        prediction = amplitude * jstats.norm.pdf(x, loc=0, scale=sigma)
        return prediction - y_corr

    # 1. Initialize the solver
    lm = jaxopt.LevenbergMarquardt(residual_fun=gaussian_residuals)

    # 2. Run the optimization
    # JAX will compile the math, the gradients, and the solver logic into one kernel
    p_hat = lm.run(p_init, x=x, y_corr=y_corr)

    return p_hat.params[1]
