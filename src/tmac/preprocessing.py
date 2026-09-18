from functools import partial
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy import optimize as joptimize

from tmac import optimize


@partial(jax.jit, inline=True)
def check_input_format(data: Any) -> jax.Array:
    """Validates the input data type and ensures it is a 2D JAX array
    Args: data: Any
    Returns:
        A 2D JAX array where 1D inputs are promoted to column vectors.
    Raise:
        TypeError: If the input is not a numpy or JAX array.
        ValueError: If the input has more than 2 dimensions.
    """
    if not isinstance(data, (np.ndarray, jax.Array)):
        raise TypeError("The red and green matricies must be the numpy or jax arrays")

    if data.ndim not in (1, 2):
        raise ValueError(
            f"The red and green matricies should be 1 or 2 dimensional: {data.ndim}"
        )

    return jnp.atleast_2d(data)


def _interp_nans_column(y) -> jax.Array:
    n = y.shape[0]
    idx = jnp.arange(n)
    valid = jnp.isfinite(y)

    prev = lax.cummax(jnp.where(valid, idx, -1))  # last valid index <= i
    nxt = lax.cummin(jnp.where(valid, idx, n), reverse=True)  # next valid index >= i

    prev_c = jnp.clip(prev, 0, n - 1)
    nxt_c = jnp.clip(nxt, 0, n - 1)
    y0, y1 = y[prev_c], y[nxt_c]

    y0 = jnp.where(prev < 0, y1, y0)  # leading NaNs -> hold first valid value
    y1 = jnp.where(nxt >= n, y0, y1)  # trailing NaNs -> hold last valid value

    span = jnp.where(nxt_c == prev_c, 1, nxt_c - prev_c).astype(y.dtype)
    w = (idx - prev_c).astype(y.dtype) / span
    return jnp.where(valid, y, y0 + w * (y1 - y0))


@jax.jit
def interpolate_over_nans(input_mat):
    """Function to interpolate over NaN values along the first dimension of a matrix

    Args:
        input_mat: numpy or jax array, [time, neurons]

    Returns: Interpolated input_mat, interpolated time
    """
    input_mat = check_input_format(input_mat)
    return jax.vmap(_interp_nans_column, in_axes=1, out_axes=1)(input_mat)


@jax.jit
def photobleach_correction(time_by_neurons: Union[np.ndarray, jax.Array]) -> jax.Array:
    """Function to fit an exponential with a shared tau to all the columns of time_by_neurons

    This function fits the function A*exp(-t / tau) to the matrix time_by_neurons. Tau is a single time constant shared
    between every column in time_by_neurons. A is an amplitude vector that is fit separately for each column. The
    correction is time_by_neurons / exp(-t / tau), preserving the amplitude of the data.

    This function can handle nans in the input

    Args:
        time_by_neurons: numpy or jax array [time, neurons]

    Returns: time_by_neurons divided by the exponential
    """

    # convert inputs to tensors
    time_by_neurons = check_input_format(time_by_neurons)
    t = jnp.arange(time_by_neurons.shape[0], dtype=time_by_neurons.dtype)
    tau_0 = t[-1, None] / 2
    a_0 = jnp.nanmean(time_by_neurons, axis=0)
    p_0 = jnp.concatenate((tau_0, a_0), axis=0)

    # mask out any un f
    isfinite = jnp.isfinite(time_by_neurons)

    def loss_fn(p):
        exponential_approx = p[None, 1:] * jnp.exp(-t[:, None] / p[0])
        # set unmeasured values to 0, so they don't show up in the sum
        squared_error = (
            exponential_approx
            - jnp.where(
                isfinite,
                time_by_neurons,
                exponential_approx,
            )
        ) ** 2
        return squared_error.sum()  # type: ignore

    p_hat = optimize._lbfgs_minimize(loss_fn, p_0, n_steps=100)

    time_by_neurons_corrected = time_by_neurons / jnp.exp(-t[:, None] / p_hat[0])
    # put the unmeasured value nans back in
    time_by_neurons_corrected = jnp.where(isfinite, time_by_neurons_corrected, jnp.nan)

    return time_by_neurons_corrected
