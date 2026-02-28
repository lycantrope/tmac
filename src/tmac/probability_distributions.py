from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import tmac.fourier as tfo


@partial(jax.jit, inline=True)
def covariance_fft(
    variance: float, scale: float, freq: jax.Array, cutoff: float
) -> jax.Array:
    return (
        np.sqrt(2 * np.pi)
        * variance
        * scale
        * jnp.maximum(jnp.exp(-0.5 * (scale * freq) ** 2), cutoff)
    )


@partial(jax.jit, static_argnames=("threshold", "truncate_freq"))
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

    # smallest length scale (longest in fourier space)
    min_length = jnp.array((length_scale_a, length_scale_m)).min()
    t_max = r.shape[0]

    freq = tfo.get_fourier_freq(t_max)

    if truncate_freq:
        max_freq = 2 * jnp.log(threshold) / min_length**2
        mask = jnp.array(freq**2 < max_freq)
    else:
        mask = jnp.ones_like(freq) > 0

    cutoff = float(1 / threshold)

    # compute the diagonals of the covariances in fourier space
    covariance_a_fft = covariance_fft(variance_a, length_scale_a, freq, cutoff)
    covariance_m_fft = covariance_fft(variance_m, length_scale_m, freq, cutoff)

    f11 = 1 / covariance_a_fft + variance_g_noise_inv
    f22 = 1 / covariance_m_fft + variance_r_noise_inv + variance_g_noise_inv
    f12 = jnp.tile(variance_g_noise_inv, f11.shape)

    f_det = f11 * f22 - f12**2

    k = f11 - f12**2 / f22

    f11_inv = 1 / k
    f22_inv = 1 / f22 + f12**2 / f22**2 / k
    f12_inv = -f12 / f22 / k

    # we filter out the truncate sequence.
    log_det_term = -(
        jnp.where(mask, jnp.log(f_det), 0.0).sum()
        + jnp.where(mask, jnp.log(covariance_a_fft * covariance_m_fft), 0.0).sum()
        + t_max * jnp.log(variance_g_noise * variance_r_noise)
    )

    # compute the quadratic term
    fourier_r_trimmed = jnp.where(mask, fourier_r, 0.0)
    fourier_g_trimmed = jnp.where(mask, fourier_g, 0.0)

    auto_corr_term = (variance_r_noise_inv * r**2 + variance_g_noise_inv * g**2).sum()
    normalized_r_fft = variance_r_noise_inv * fourier_r_trimmed
    normalized_g_fft = variance_g_noise_inv * fourier_g_trimmed

    f_quad_mult_1 = normalized_g_fft
    f_quad_mult_2 = normalized_r_fft + normalized_g_fft  # type: ignore

    f_quad = (
        (f11_inv * f_quad_mult_1**2).sum()  # type: ignore
        + (f22_inv * f_quad_mult_2**2).sum()  # type: ignore
        + 2 * (f12_inv * f_quad_mult_1 * f_quad_mult_2).sum()
    )

    quad_term = -(auto_corr_term - f_quad)

    return jnp.mean(log_det_term + quad_term)


@partial(jax.jit, static_argnames=("threshold", "truncate_freq"))
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
    Returns: a_hat, m_hat
    """

    # exponentiate the log variances and invert the noise variances
    variance_r_noise = jnp.exp(log_variance_r_noise)
    variance_g_noise = jnp.exp(log_variance_g_noise)
    variance_a = jnp.exp(log_variance_a)
    length_scale_a = jnp.exp(log_tau_a)
    variance_m = jnp.exp(log_variance_m)
    length_scale_m = jnp.exp(log_tau_m)

    variance_r_noise_inv = 1.0 / variance_r_noise
    variance_g_noise_inv = 1.0 / variance_g_noise

    # calculate the gaussian process components in fourier space
    t_max = r.shape[0]

    freq = tfo.get_fourier_freq(t_max)
    # smallest length scale (longest in fourier space)
    min_length = jnp.array((length_scale_a, length_scale_m)).min()

    if truncate_freq:
        max_freq = 2 * np.log(threshold) / min_length**2
        frequencies_to_keep = freq**2 < max_freq
    else:
        frequencies_to_keep = jnp.full(freq.shape, True)

    cutoff = float(1.0 / threshold)

    # compute the diagonals of the covariances in fourier space
    covariance_a_fft = covariance_fft(variance_a, length_scale_a, freq, cutoff)
    covariance_m_fft = covariance_fft(variance_m, length_scale_m, freq, cutoff)

    f11 = 1 / covariance_a_fft + variance_g_noise_inv
    f22 = 1 / covariance_m_fft + variance_r_noise_inv + variance_g_noise_inv
    f12 = jnp.tile(variance_g_noise_inv, f11.shape)

    k = f11 - f12**2 / f22

    f11_inv = 1 / k
    f22_inv = 1 / f22 + f12**2 / f22**2 / k
    f12_inv = -f12 / f22 / k

    # compute the quadratic term
    normalized_r_fft = variance_r_noise_inv * fourier_r
    normalized_g_fft = variance_g_noise_inv * fourier_g

    f_quad_mult_1 = normalized_g_fft
    f_quad_mult_2 = normalized_r_fft + normalized_g_fft

    a_fft = f11_inv * f_quad_mult_1 + f12_inv * f_quad_mult_2
    m_fft = f22_inv * f_quad_mult_2 + f12_inv * f_quad_mult_1

    a_padded = jnp.where(frequencies_to_keep, a_fft, 0.0 + 0.0j)
    m_padded = jnp.where(frequencies_to_keep, m_fft, 0.0 + 0.0j)

    return jnp.stack([tfo.real_ifft(a_padded) + 1, tfo.real_ifft(m_padded)])
