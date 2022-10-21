import numpy as np

from paderwasn.synchronization.sync import compensate_sro


def ornstein_uhlenbeck(
        seq_len, start_val, mean_inf, sigma, theta, rng=np.random
):
    """Discrete-time Euler-Maruyama approximation of an Ornstein-Uhlenbeck
    process

    For example, the Ornstein-Uhlenbeck process could be used to model a
    time-varying sampling rate offeset (SRO) as described in "On
    Synchronization of Wireless Acoustic Sensor Networks in the presence of
    Time-Varying Sampling Rate Offsets and Speaker Changes"

    Args:
        seq_len (int):
            Length of the sequence to be generated using the Ornstein-Uhlenbeck
            process
        start_val (float):
            Value at which the Ornstein-Uhlenbeck process starts
        mean_inf (float):
            Mean value reached after all transient effects have died out
        sigma (float):
            Variance of the Gaussian distribution involved in the
            Euler-Maruyama approximation
        theta (float):
            Factor specifying the convergence speed to the static mean
        rng:
            A random number generator object
    Returns:
        x (numpy.ndarray):
            Vector corresponding to a realization of the random process
    """
    x = np.zeros(seq_len)
    x[0] = start_val
    for i in range(1, seq_len):
        x[i] = (1 - theta) * x[i - 1]
        x[i] += theta * mean_inf + sigma * rng.normal()
    return x


def sim_sro(sig, sro, fft_size=8192):
    """Simulate the sampling rate offset (SRO) of the the given signal using
    the STFT resampling method introduced in "Efficient Sampling Rate Offset
    Compensation - An Overlap-Save Based Approach"

    The function can handle constant and time-varying SROs (SRO trajectories)

    Args:
        sig (array-like):
            Vector corresponding to the signal to be resampled
        sro (float or array-like):
            SRO in parts-per-million (ppm) to be simulated. If sro is a
            scalar value it will be treated as constant SRO. Otherwise, sro
            is interpreted as SRO trajectory corresponding to a time-varying
            SRO.
        fft_size (int):
            FFT size used for calculating the STFT. The STFT uses a frame size
            of fft_size/2 and a frame shift of fft_size/4.
    Returns:
        sig_resampled (numpy.ndarray):
            Vector corresponding to the resampled signal
    """
    return compensate_sro(sig, -sro, fft_size)
