import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import optimize
from scipy.stats import norm

import tmac.fourier as tfo
import tmac.optimization as opt
import tmac.preprocessing as pp
import tmac.probability_distributions as tpd


@dataclass(slots=True, frozen=True)
class TMACResult:
    a: np.ndarray
    m: np.ndarray
    variance_r_noise: np.ndarray
    variance_g_noise: np.ndarray
    variance_a: np.ndarray
    length_scale_a: np.ndarray
    variance_m: np.ndarray
    length_scale_m: np.ndarray

    def __getitem__(self, key):
        if key not in self.__slots__:
            raise KeyError(key)
        return getattr(self, key)

    def __repr__(self) -> str:
        fmt = []
        for key in self.__slots__:
            val = getattr(self, key)
            if hasattr(val, "shape"):
                fmt.append(f"{key}=Array{val.shape}")
            else:
                fmt.append(f"{key}={val}")

        return f"TMACResult({', '.join(fmt)})"


def tmac_ac(
    red_np,
    green_np,
    optimizer="BFGS",
    verbose=False,
    truncate_freq=True,
    device="cpu",
) -> TMACResult:
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
    dtype = torch.float64

    red_np = pp.check_input_format(red_np)
    green_np = pp.check_input_format(green_np)

    assert np.all(np.isfinite(red_np)) and np.all(
        np.isfinite(green_np)
    ), "Input data cannot have any nan or inf"

    assert (
        red_np.shape == green_np.shape
    ), "red and green matricies must be the same shape"

    T, C = red_np.shape
    # convert data to units of fold mean and subtract mean
    mean_red = np.mean(red_np, axis=0)
    mean_green = np.mean(green_np, axis=0)
    red_np = red_np / mean_red - 1
    green_np = green_np / mean_green - 1

    # convert to tensors and fourier transform
    red = torch.tensor(red_np, device=device, dtype=dtype)
    green = torch.tensor(green_np, device=device, dtype=dtype)
    red_fft = tfo.real_fft(red)
    green_fft = tfo.real_fft(green)

    # estimate all model parameters from the data
    variance_r_noise_init = np.var(red_np, axis=0)
    variance_g_noise_init = np.var(green_np, axis=0)
    variance_a_init = np.var(green_np, axis=0)
    variance_m_init = np.var(red_np, axis=0)

    # initialize length scale
    length_scale_a_init = np.ones(C)
    length_scale_m_init = np.ones(C)

    # preallocate space for all the training variables
    a_trained = np.zeros((T, C), dtype=red_np.dtype)
    m_trained = np.zeros((T, C), dtype=red_np.dtype)
    variance_r_noise_trained = np.zeros(C)
    variance_g_noise_trained = np.zeros(C)
    variance_a_trained = np.zeros(C)
    length_scale_a_trained = np.zeros(C)
    variance_m_trained = np.zeros(C)
    length_scale_m_trained = np.zeros(C)

    # loop through each neuron and perform inference
    start = time.time()
    for n in range(C):
        # get the initial values for the hyperparameters of this neuron
        # All hyperparameters are positive, so we fit them in log space
        evidence_training_variables = np.log(
            [
                variance_r_noise_init[n],
                variance_g_noise_init[n],
                variance_a_init[n],
                length_scale_a_init[n],
                variance_m_init[n],
                length_scale_m_init[n],
            ]
        )

        # define the evidence loss function. This function takes in and returns pytorch tensors
        def evidence_loss_fn(
            training_variables,
            r=red[:, n],
            fourier_r=red_fft[:, n],
            g=green[:, n],
            fourier_g=green_fft[:, n],
        ):
            return -tpd.tmac_evidence(
                r=r,
                fourier_r=fourier_r,
                log_variance_r_noise=training_variables[0],
                g=g,
                fourier_g=fourier_g,
                log_variance_g_noise=training_variables[1],
                log_variance_a=training_variables[2],
                log_tau_a=training_variables[3],
                log_variance_m=training_variables[4],
                log_tau_m=training_variables[5],
                truncate_freq=truncate_freq,
            )

        trained_variances = opt.scipy_minimize_with_grad(
            evidence_loss_fn,
            evidence_training_variables,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
        )

        # calculate the posterior values
        # The posterior is gaussian so we don't need to optimize, we find a and m in one step
        trained_variance_torch = torch.tensor(
            trained_variances.x, dtype=dtype, device=device
        )
        a, m = tpd.tmac_posterior(
            red[:, n],
            red_fft[:, n],
            trained_variance_torch[0],
            green[:, n],
            green_fft[:, n],
            trained_variance_torch[1],
            trained_variance_torch[2],
            trained_variance_torch[3],
            trained_variance_torch[4],
            trained_variance_torch[5],
            truncate_freq=truncate_freq,
        )
        trained_variance_exp = torch.exp(trained_variance_torch).numpy()
        a_trained[:, n] = a.numpy()
        m_trained[:, n] = m.numpy()
        variance_r_noise_trained[n] = trained_variance_exp[0]
        variance_g_noise_trained[n] = trained_variance_exp[1]
        variance_a_trained[n] = trained_variance_exp[2]
        length_scale_a_trained[n] = trained_variance_exp[3]
        variance_m_trained[n] = trained_variance_exp[4]
        length_scale_m_trained[n] = trained_variance_exp[5]

        if verbose:
            # print out timing
            elapsed = time.time() - start
            remaining = elapsed / (n + 1) * (red_np.shape[1] - (n + 1))
            print(f"{n+1:d}/{C:d} neurons complete")
            print(f"{elapsed:.3f}s elapsed, estimated {remaining:.3f}s remaining")

    return TMACResult(
        a=a_trained,
        m=m_trained,
        variance_r_noise=variance_r_noise_trained,
        variance_g_noise=variance_g_noise_trained,
        variance_a=variance_a_trained,
        length_scale_a=length_scale_a_trained,
        variance_m=variance_m_trained,
        length_scale_m=length_scale_m_trained,
    )


def initialize_length_scale(y):
    """Function to fit a Gaussian to the autocorrelation of y

    Args:
        y: numpy vector

    Returns: Standard deviation of a Gaussian fit to the autocorrelation of y
    """

    x = np.arange(-len(y) / 2, len(y) / 2) + 0.5
    y_z_score = (y - np.mean(y)) / np.std(y)
    y_corr = np.correlate(y_z_score, y_z_score, mode="same")

    # fit the std of a gaussian to the correlation function
    def loss(p):
        return p[0] * norm.pdf(x, 0, p[1]) - y_corr

    p_init = np.array((np.max(y_corr), 1.0))
    p_hat = optimize.leastsq(loss, p_init)[0]

    # return the standard deviation
    return p_hat[1]
