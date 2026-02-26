from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def get_fourier_freq(t_max: int):
    """Returns the frequencies for a vector of length t_max"""
    n_cos = jnp.ceil((t_max + 1) / 2)  # number of cosine terms (positive freqs)
    n_sin = jnp.floor((t_max - 1) / 2)  # number of sine terms (negative freqs)
    w_vec = jnp.arange(t_max)
    w_vec = jnp.roll(w_vec - n_sin, shift=-n_sin)
    frequencies = 2 * jnp.pi / t_max * w_vec
    return frequencies


@partial(jax.jit, static_argnums=(0,))
def get_fourier_basis(n_ind: int) -> Tuple[jax.Array, jax.Array]:
    x = jnp.arange(n_ind)
    n_cos = jnp.ceil((n_ind + 1) / 2).astype(int)
    n_sin = jnp.floor((n_ind - 1) / 2).astype(int)
    # frequency vector
    frequency_vec = jnp.roll(x - n_sin, shift=-n_sin, axis=0) * 2 * jnp.pi / n_ind

    fourier_basis = frequency_vec[:, None] * x[None, :]

    idx = x[:, None]
    fourier_basis = jnp.where(
        idx < n_cos, jnp.cos(fourier_basis), jnp.sin(fourier_basis)
    )
    fourier_basis = fourier_basis.at[:].divide(jnp.sqrt(n_ind / 2))
    fourier_basis = fourier_basis.at[0].divide(jnp.sqrt(2))

    fourier_basis = fourier_basis.at[n_cos - 1].divide(jnp.sqrt(2 - n_ind % 2))
    return fourier_basis, frequency_vec


@jax.jit
def real_fft(x: jax.Array):
    """Performs a real discrete 1D Fourier transform of the columns of x"""
    single_vec = x.ndim == 1
    if single_vec:
        x = x[:, None]

    n = x.shape[0]

    x_fft = jnp.fft.fft(x, n, axis=0) / jnp.sqrt(n / 2)
    x_fft = x_fft.at[0].divide(jnp.sqrt(2))
    imx = jnp.ceil((n - 1) // 2).astype(int)
    x_fft = x_fft.at[imx].divide(jnp.sqrt(2 - n % 2))

    isin = jnp.ceil((n + 1) // 2).astype(int)

    idx = jnp.arange(n)[:, None]
    x_hat = jnp.where(idx < isin, x_fft.real, -x_fft.imag)

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

    x_hat = x_hat.at[0].multiply(jnp.sqrt(2))
    imx = jnp.ceil((nxh - 1) // 2).astype(int)
    x_hat = x_hat.at[imx].multiply(jnp.sqrt(2 - nxh % 2))

    n_cos = jnp.ceil((nxh + 1) / 2).astype(int)
    n_sin = jnp.floor((nxh - 1) / 2).astype(int)
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
    x = x[:nxh, :] * jnp.sqrt(nxh / 2)

    if single_vec:
        x = x[:, 0]

    return x
