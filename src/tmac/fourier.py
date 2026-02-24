from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy as jcp
import numpy as np
import torch


def get_fourier_freq(t_max: int):
    """Returns the frequencies for a vector of length t_max"""
    n_cos = jnp.ceil((t_max + 1) / 2)  # number of cosine terms (positive freqs)
    n_sin = jnp.floor((t_max - 1) / 2)  # number of sine terms (negative freqs)
    w_cos = jnp.arange(0, n_cos)  # cosine freqs
    w_sin = jnp.arange(-n_sin, 0)  # sine freqs
    w_vec = jnp.concatenate((w_cos, w_sin), axis=0)
    frequencies = 2 * jnp.pi / t_max * w_vec

    return frequencies


def real_fft(x: jax.Array, n: Optional[int] = None):
    """Performs a real discrete 1D Fourier transform of the columns of x"""

    single_vec = False
    if len(x.shape) == 1:
        single_vec = True
        x = x[:, None]

    if n is None:
        n = x.shape[0]

    x_fft = jnp.fft.fft(x, n, axis=0) / jnp.sqrt(n / 2)
    x_fft = x_fft.at[0].divide(jnp.sqrt(2))
    if jnp.mod(n, 2) == 0:
        imx = int(jnp.ceil((n - 1) / 2))
        x_fft = x_fft.at[imx].divide(jnp.sqrt(2))

    x_hat = x_fft.real
    isin = int(jnp.ceil((n + 1) / 2))
    x_hat.at[isin:, :].set(-x_fft[isin:, :].imag)

    if single_vec:
        x_hat = x_hat[:, 0]

    return x_hat


def real_ifft(x_hat: jax.Array, n: Optional[int] = None):
    """Performs an inverse of a real discrete 1D Fourier transform of the columns of x"""

    single_vec = x_hat.ndim == 1
    if single_vec:
        x_hat = x_hat[:, None]

    if n is None:
        n = x_hat.shape[0]

    nxh = x_hat.shape[0]
    n_cos = int(np.ceil((nxh + 1) / 2))
    n_sin = int(np.floor((nxh - 1) / 2))

    x_hat = x_hat.at[0].multiply(jnp.sqrt(2))

    if jnp.mod(nxh, 2) == 0:
        x_hat.at[n_cos - 1, :].multiply(jnp.sqrt(2))

    sin_basis = jnp.flip(x_hat[1 : n_sin + 1, :], axis=0)
    sin_basis = sin_basis - 1j * sin_basis
    cos_basis = x_hat[1 : n_sin + 1, :] + 1j * jnp.flip(x_hat[n_cos:, :], axis=0)

    xfft = jnp.concatenate(
        [x_hat[:1, :], sin_basis, x_hat[n_sin + 1 : n_cos, :], cos_basis], axis=0
    )
    x = jnp.fft.ifft(xfft, axis=0).real * jnp.sqrt(nxh / 2)
    x = x[:n, :]
    if single_vec:
        x = x[:, 0]

    return x


def get_fourier_basis(n_ind):
    """Returns an orthonormal real Fourier basis for a vector of length n_ind"""

    n_cos = jnp.ceil((n_ind + 1) / 2)
    n_sin = jnp.floor((n_ind - 1) / 2)

    cos_freq = 2 * jnp.pi / n_ind * jnp.arange(n_cos)
    sin_freq = 2 * jnp.pi / n_ind * jnp.arange(-n_sin, 0)
    frequency_vec = np.concatenate((cos_freq, sin_freq), axis=0)  # frequency vector

    x = jnp.arange(n_ind)
    cos_basis = jnp.cos(cos_freq[:, None] * x[None, :]) / jnp.sqrt(n_ind / 2)
    sin_basis = jnp.sin(sin_freq[:, None] * x[None, :]) / jnp.sqrt(n_ind / 2)
    fourier_basis = jnp.concatenate((cos_basis, sin_basis), axis=0)

    fourier_basis = fourier_basis.at[0].divide(jnp.sqrt(2))

    if jnp.mod(n_ind, 2) == 0:
        fourier_basis = fourier_basis.at[int(n_cos - 1)].divide(jnp.sqrt(2))

    return fourier_basis, frequency_vec
