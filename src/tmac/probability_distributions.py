from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch

import tmac.fourier as tfo


def tmac_evidence(
    r,
    fourier_r,
    log_variance_r_noise,
    g,
    fourier_g,
    log_variance_g_noise,
    log_variance_a,
    log_tau_a,
    log_variance_m,
    log_tau_m,
    freq,
    threshold=1e8,
    truncate_freq=True,
):
    """Two-channel motion artifact correction (TMAC) evidence and posterior distribution

    Args:
        r: red channel
        fourier_r: fourier transform of the red channel
        log_variance_r_noise: log of the variance of the red channel Gaussian noise
        g: green channel
        fourier_g: fourier transform of the green channel
        log_variance_g_noise: log of the variance of the green channel Gaussian noise
        log_variance_a: log of the variance of the activity
        log_tau_a: log of the timescale of the activity
        log_variance_m: log of the variance of the motion artifact
        log_tau_m: log of the timescale of the motion artifact
        threshold: maximum condition number of the radial basis function kernel for the Gaussian process
        calculate_posterior: boolean, whether to calculate the posterior
        truncate_freq: boolean, if true truncates low amplitude frequencies in Fourier domain. This should give the same
            results but may give sensitivity to the initial conditions
    if calculate_posterior:
        Returns: a_hat, m_hat
    else:
        Returns: log probability of the evidence
    """

    # exponentiate the log variances and invert the noise variances
    variance_r_noise = jnp.exp(log_variance_r_noise)
    variance_g_noise = jnp.exp(log_variance_g_noise)
    variance_a = jnp.exp(log_variance_a)
    length_scale_a = jnp.exp(log_tau_a)
    variance_m = jnp.exp(log_variance_m)
    length_scale_m = jnp.exp(log_tau_m)

    variance_r_noise_inv = 1 / variance_r_noise
    variance_g_noise_inv = 1 / variance_g_noise
    dtype = r.dtype
    # smallest length scale (longest in fourier space)
    min_length = jnp.array((length_scale_a, length_scale_m)).min()
    t_max = r.shape[0]

    if truncate_freq:
        max_freq = 2 * jnp.log(threshold) / min_length**2
        mask = (freq**2 < max_freq).astype(freq.dtype)
    else:
        mask = jnp.ones_like(freq)

    cutoff = jnp.array(1 / threshold, dtype=dtype)

    # compute the diagonals of the covariances in fourier space
    covariance_a_fft = jnp.maximum(
        jnp.exp(-0.5 * freq**2 * length_scale_a**2) * mask, cutoff
    )
    covariance_a_fft = (
        variance_a * (length_scale_a * np.sqrt(2 * np.pi)) * covariance_a_fft
    )
    covariance_m_fft = jnp.maximum(
        jnp.exp(-0.5 * freq**2 * length_scale_m**2) * mask, cutoff
    )
    covariance_m_fft = (
        variance_m * (length_scale_m * np.sqrt(2 * np.pi)) * covariance_m_fft
    )

    f11 = 1 / covariance_a_fft + variance_g_noise_inv
    f22 = 1 / covariance_m_fft + variance_r_noise_inv + variance_g_noise_inv
    f12 = jnp.tile(variance_g_noise_inv, f11.shape)

    f_det = f11 * f22 - f12**2

    k = f11 - f12**2 / f22

    f11_inv = 1 / k
    f22_inv = 1 / f22 + f12**2 / f22**2 / k
    f12_inv = -f12 / f22 / k

    log_det_term = -(
        jnp.log(f_det).sum()
        + jnp.log(covariance_a_fft * covariance_m_fft).sum()
        + t_max * jnp.log(variance_g_noise * variance_r_noise)
    )

    # compute the quadratic term

    auto_corr_term = (variance_r_noise_inv * r**2 + variance_g_noise_inv * g**2).sum()

    normalized_r_fft = variance_r_noise_inv * fourier_r
    normalized_g_fft = variance_g_noise_inv * fourier_g

    f_quad_mult_1 = normalized_g_fft
    f_quad_mult_2 = normalized_r_fft + normalized_g_fft

    f_quad = (
        (f11_inv * f_quad_mult_1**2).sum()
        + (f22_inv * f_quad_mult_2**2).sum()
        + 2 * (f12_inv * f_quad_mult_1 * f_quad_mult_2).sum()
    )

    quad_term = -(auto_corr_term - f_quad)

    return jnp.mean(log_det_term + quad_term)


def tmac_posterior(
    r,
    fourier_r,
    log_variance_r_noise,
    g,
    fourier_g,
    log_variance_g_noise,
    log_variance_a,
    log_tau_a,
    log_variance_m,
    log_tau_m,
    threshold=1e8,
    truncate_freq=True,
):
    """Two-channel motion artifact correction (TMAC) evidence and posterior distribution

    Args:
        r: red channel
        fourier_r: fourier transform of the red channel
        log_variance_r_noise: log of the variance of the red channel Gaussian noise
        g: green channel
        fourier_g: fourier transform of the green channel
        log_variance_g_noise: log of the variance of the green channel Gaussian noise
        log_variance_a: log of the variance of the activity
        log_tau_a: log of the timescale of the activity
        log_variance_m: log of the variance of the motion artifact
        log_tau_m: log of the timescale of the motion artifact
        threshold: maximum condition number of the radial basis function kernel for the Gaussian process
        calculate_posterior: boolean, whether to calculate the posterior
        truncate_freq: boolean, if true truncates low amplitude frequencies in Fourier domain. This should give the same
            results but may give sensitivity to the initial conditions
    if calculate_posterior:
        Returns: a_hat, m_hat
    else:
        Returns: log probability of the evidence
    """

    # exponentiate the log variances and invert the noise variances
    variance_r_noise = jnp.exp(log_variance_r_noise)
    variance_g_noise = jnp.exp(log_variance_g_noise)
    variance_a = jnp.exp(log_variance_a)
    length_scale_a = jnp.exp(log_tau_a)
    variance_m = jnp.exp(log_variance_m)
    length_scale_m = jnp.exp(log_tau_m)

    variance_r_noise_inv = 1 / variance_r_noise
    variance_g_noise_inv = 1 / variance_g_noise

    device = r.device
    dtype = r.dtype

    # calculate the gaussian process components in fourier space
    t_max = r.shape[0]

    all_freq = tfo.get_fourier_freq(t_max)
    # smallest length scale (longest in fourier space)
    min_length = jnp.array((length_scale_a, length_scale_m)).min()

    if truncate_freq:
        max_freq = 2 * jnp.log(threshold) / min_length**2
        frequencies_to_keep = all_freq**2 < max_freq
    else:
        frequencies_to_keep = jnp.full(all_freq.shape, True)

    freq = all_freq[frequencies_to_keep]
    cutoff = jnp.array(1 / threshold, device=device, dtype=dtype)

    # compute the diagonals of the covariances in fourier space
    covariance_a_fft = jnp.maximum(jnp.exp(-0.5 * freq**2 * length_scale_a**2), cutoff)
    covariance_a_fft = (
        variance_a * (length_scale_a * jnp.sqrt(2 * jnp.pi)) * covariance_a_fft
    )
    covariance_m_fft = jnp.maximum(jnp.exp(-0.5 * freq**2 * length_scale_m**2), cutoff)
    covariance_m_fft = (
        variance_m * (length_scale_m * np.sqrt(2 * np.pi)) * covariance_m_fft
    )

    f11 = 1 / covariance_a_fft + variance_g_noise_inv
    f22 = 1 / covariance_m_fft + variance_r_noise_inv + variance_g_noise_inv
    f12 = jnp.tile(variance_g_noise_inv, f11.shape)

    k = f11 - f12**2 / f22

    f11_inv = 1 / k
    f22_inv = 1 / f22 + f12**2 / f22**2 / k
    f12_inv = -f12 / f22 / k

    # compute the quadratic term
    fourier_r_trimmed = fourier_r[frequencies_to_keep]
    fourier_g_trimmed = fourier_g[frequencies_to_keep]

    normalized_r_fft = variance_r_noise_inv * fourier_r_trimmed
    normalized_g_fft = variance_g_noise_inv * fourier_g_trimmed

    f_quad_mult_1 = normalized_g_fft
    f_quad_mult_2 = normalized_r_fft + normalized_g_fft

    a_fft = f11_inv * f_quad_mult_1 + f12_inv * f_quad_mult_2
    m_fft = f22_inv * f_quad_mult_2 + f12_inv * f_quad_mult_1

    m_padded = jnp.zeros_like(r, device=device, dtype=dtype)
    a_padded = jnp.zeros_like(r, device=device, dtype=dtype)

    m_padded.at[frequencies_to_keep].set(m_fft)
    a_padded.at[frequencies_to_keep].set(a_fft)

    m_hat = tfo.real_ifft(m_padded)
    a_hat = tfo.real_ifft(a_padded)

    return a_hat + 1, m_hat
