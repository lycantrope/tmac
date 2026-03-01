from functools import partial
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy import optimize as joptimize


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


@jax.jit
def interpolate_over_nans(input_mat: Union[np.ndarray, jax.Array]) -> jax.Array:
    """Function to interpolate over NaN values along the first dimension of a matrix

    Args:
        input_mat: numpy or jax array, [time, neurons]

    Returns: Interpolated input_mat, interpolated time
    """

    input_mat = check_input_format(input_mat)

    def fill_nan_smooth(arr, kernel_size=3) -> jax.Array:
        # 1. Create a mask of NaNs
        nan_mask = ~jnp.isfinite(arr)

        # 2. Replace NaNs with mean for convolution
        clean_arr = jnp.nan_to_num(arr, nan=jnp.nanmean(arr))

        # 3. Create a kernel for local averaging (e.g., 3x3)
        kernel = jnp.ones(kernel_size) / kernel_size

        # 4. Convolve to get local averages
        # For 1D, use convolve. For 2D, adjust kernel and input dimensions.
        smoothed = jnp.convolve(clean_arr, kernel, mode="same")

        # 5. Fill only the original NaN locations with smoothed values
        return jnp.where(nan_mask, smoothed, arr)  # type: ignore

    # Loop over the input_mat if is nan return original data, else return fill_nan_smooth
    @partial(jax.vmap, in_axes=1, out_axes=1)
    def fill_nan_smooth_all(all_nan, arr):
        return lax.cond(all_nan[0], lambda x: x, fill_nan_smooth, arr)

    # check each neuron if all data is nan.
    all_nan = jnp.all(~jnp.isfinite(input_mat), axis=0, keepdims=True)

    return fill_nan_smooth_all(all_nan, input_mat)


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
    t = jnp.arange(time_by_neurons.shape[0])
    tau_0 = t[-1, None] / 2
    a_0 = jnp.nanmean(time_by_neurons, axis=0)
    p_0 = jnp.concatenate((tau_0, a_0), axis=0)

    # mask out any un f
    isfinite = jnp.isfinite(time_by_neurons)

    def loss_fn(p, t, time_by_neurons, isfinite):
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

    p_hat = joptimize.minimize(
        loss_fn,
        p_0,
        args=(t, time_by_neurons, isfinite),
        method="BFGS",
    )
    time_by_neurons_corrected = time_by_neurons / jnp.exp(-t[:, None] / p_hat.x[0])
    # put the unmeasured value nans back in
    time_by_neurons_corrected = jnp.where(isfinite, time_by_neurons_corrected, jnp.nan)

    return time_by_neurons_corrected
