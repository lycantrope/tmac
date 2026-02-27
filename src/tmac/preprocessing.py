import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy import optimize as joptimize, signal as jsignal
from jax import lax


def check_input_format(data) -> jax.Array:

    if not isinstance(data, (np.ndarray, jax.Array)):
        raise TypeError("The red and green matricies must be the numpy or jax arrays")

    if data.ndim not in (1, 2):
        raise ValueError(
            f"The red and green matricies should be 1 or 2 dimensional: {data.ndim}"
        )

    if data.ndim == 1:
        data = data[:, None]

    return jnp.array(data)


def interpolate_over_nans(input_mat):
    """Function to interpolate over NaN values along the first dimension of a matrix

    Args:
        input_mat: numpy array, [time, neurons]

    Returns: Interpolated input_mat, interpolated time
    """

    input_mat = check_input_format(input_mat)

    # if t is not specified, assume it has been sampled at regular intervals

    def fill_nan_smooth(arr, kernel_size=3) -> jax.Array:
        # 1. Create a mask of NaNs
        nan_mask = jnp.isnan(arr)

        # 2. Replace NaNs with mean for convolution
        clean_arr = jnp.nan_to_num(arr, nan=jnp.nanmean(arr))

        # 3. Create a kernel for local averaging (e.g., 3x3)
        kernel = jnp.ones(kernel_size) / kernel_size

        # 4. Convolve to get local averages
        # For 1D, use convolve. For 2D, adjust kernel and input dimensions.
        smoothed = jsignal.convolve(clean_arr, kernel, mode="same")

        # 5. Fill only the original NaN locations with smoothed values
        return jnp.where(nan_mask, smoothed, arr)  # type: ignore

    # loop through each column of the data and interpolate them separately

    output_mat = lax.map(
        lambda x: lax.select(jnp.all(~jnp.isfinite(x)), x, fill_nan_smooth(x)),
        input_mat.T,
    )

    return output_mat.T


@jax.jit
def photobleach_correction(time_by_neurons):
    """Function to fit an exponential with a shared tau to all the columns of time_by_neurons

    This function fits the function A*exp(-t / tau) to the matrix time_by_neurons. Tau is a single time constant shared
    between every column in time_by_neurons. A is an amplitude vector that is fit separately for each column. The
    correction is time_by_neurons / exp(-t / tau), preserving the amplitude of the data.

    This function can handle nans in the input

    Args:
        time_by_neurons: numpy array [time, neurons]

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
        squared_error = (exponential_approx - time_by_neurons) ** 2
        # set unmeasured values to 0, so they don't show up in the sum
        squared_error = jnp.where(isfinite, squared_error, 0.0)
        return squared_error.sum()  # type: ignore

    p_hat = joptimize.minimize(
        loss_fn,
        p_0,
        args=(t, time_by_neurons, isfinite),
        method="BFGS",
    )

    time_by_neurons_corrected = time_by_neurons / jnp.exp(-t[:, None] / p_hat.x[0])
    # put the unmeasured value nans back in
    time_by_neurons_corrected = jnp.where(~isfinite, jnp.nan, time_by_neurons_corrected)

    return time_by_neurons_corrected
