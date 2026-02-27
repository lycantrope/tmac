from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(0,))
def get_fourier_freq(t_max: int):
    """Returns the frequencies for a vector of length t_max"""
    n_sin = (t_max - 1) // 2  # number of sine terms (negative freqs)
    # n_cos = t_max - n_sin  # number of cosine terms (positive freqs)
    w_vec = jnp.arange(t_max)
    w_vec = jnp.roll(w_vec - n_sin, shift=-n_sin)
    frequencies = 2 * jnp.pi / t_max * w_vec
    return frequencies


@partial(jax.jit, static_argnums=(0,))
def get_fourier_basis(n_ind: int) -> Tuple[jax.Array, jax.Array]:
    x = jnp.arange(n_ind)
    n_sin = (n_ind - 1) // 2  # number of sine terms (negative freqs)
    n_cos = n_ind - n_sin  # number of cosine terms (positive freqs)
    # frequency vector
    frequency_vec = (
        jnp.roll(jnp.arange(n_ind) - n_sin, shift=-n_sin, axis=0) * 2 * jnp.pi / n_ind
    )

    fourier_basis = frequency_vec[:, None] * x[None, :]

    idx = x[:, None]
    fourier_basis = jnp.where(
        idx < n_cos, jnp.cos(fourier_basis), jnp.sin(fourier_basis)
    )
    fourier_basis = fourier_basis.at[:].divide(np.sqrt(n_ind / 2))
    fourier_basis = fourier_basis.at[0].divide(np.sqrt(2))

    fourier_basis = fourier_basis.at[n_cos - 1].divide(np.sqrt(2 - n_ind % 2))
    return fourier_basis, frequency_vec


@jax.jit
def real_fft(x: jax.Array):
    """Performs a real discrete 1D Fourier transform of the columns of x"""
    single_vec = x.ndim == 1
    if single_vec:
        x = x[:, None]

    n = x.shape[0]

    n_sin = (n - 1) // 2
    n_cos = n - n_sin

    x_fft = jnp.fft.fft(x, n, axis=0) / np.sqrt(n / 2)
    x_fft = x_fft.at[0].divide(np.sqrt(2))
    x_fft = x_fft.at[n_sin].divide(np.sqrt(2 - n % 2))

    idx = jnp.arange(n)[:, None]
    x_hat = jnp.where(idx < n_cos, x_fft.real, -x_fft.imag)

    if single_vec:
        x_hat = x_hat[:, 0]

    return x_hat


@jax.jit
def real_ifft(x_hat: jax.Array):
    """Performs an inverse of a real discrete 1D Fourier transform of the columns of x"""

    single_vec = x_hat.ndim == 1
    if single_vec:
        x_hat = x_hat[:, None]

    nxh = x_hat.shape[0]
    n_sin = (nxh - 1) // 2
    n_cos = nxh - n_sin

    x_hat = x_hat.at[0].multiply(np.sqrt(2))
    x_hat = x_hat.at[n_sin].multiply(np.sqrt(2 - nxh % 2))

    x_hat_shift_r = jnp.roll(jnp.flip(x_hat), 1, axis=0)
    idx = jnp.arange(nxh)[:, None]
    xfft = jnp.where(
        idx < n_cos,
        x_hat + 1.0j * x_hat_shift_r,
        x_hat_shift_r - 1.0j * x_hat,
    )
    xfft = xfft.at[0].set(x_hat[0])
    nyquist_val = jnp.where(
        n_cos == n_sin + 1,
        xfft[n_cos - 1],
        xfft[n_cos - 1].real + 0.0j,
    )
    xfft = xfft.at[n_sin + 1].set(nyquist_val)

    x = jnp.fft.ifft(xfft, axis=0).real
    x = x[:nxh, :] * np.sqrt(nxh / 2)

    if single_vec:
        x = x[:, 0]

    return x
