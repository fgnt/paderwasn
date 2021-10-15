import numpy as np


def coarse_sync(sig, ref_sig, len_sync):
    """Coarsely synchronize the given signals based on a crosscorrelation-based
    offset estimate

    Args:
        sig (array-like):
            Vector corresponding to a discrete-time signal
        ref_sig (array-like):
            Vector corresponding to a discrete-time reference signal
        len_sync (int):
            Amount of samples used for offset estimation
    Returns:
        sig (array-like):
            Coarsely synchronized signal
        ref_sig (array-like):
            Coarsely synchronized referencesignal
        offset (int):
            Offset between the gnal and the reference signal
    """
    x_corr = np.correlate(sig[:len_sync], ref_sig[:len_sync], mode='full')
    offset = int(np.argmax(np.abs(x_corr)) - (len_sync - 1))
    if offset > 0:
        return sig[offset:], ref_sig[:-offset], offset
    elif offset < 0:
        return sig[:offset], ref_sig[-offset:], offset
    return sig, ref_sig, offset


def golden_section_max_search(function, search_interval, tolerance=1e-4):
    """Search for the value that maximizes the given function f(x)

    Args:
        function (callable):
            Function f(x) to be maximized
        search_interval (tuple):
            Tuple (lower, upper) specifying the lower limit and the upper limit
            of the search interval
        tolerance (float):
            The search stops if the distance of the lower and the upper bound
            of the currently considered search interval are smaller or equal
            to tolerance, i.e. (upper_limit - lower_limit) <= tolerance.
    Returns:
        x_max (float):
            Argument which maximizes the given function f(x)
    """
    left_limit, right_limit = search_interval
    invphi = (np.sqrt(5) - 1) / 2
    invphi2 = (3 - np.sqrt(5)) / 2
    dist_limits = right_limit - left_limit
    intersec_first = left_limit + invphi2 * dist_limits
    intersec_sec = left_limit + invphi * dist_limits
    val_first = function(intersec_first)
    val_sec = function(intersec_sec)
    n_iter = np.ceil(np.log(tolerance / dist_limits) / np.log(invphi))
    for _ in range(int(n_iter)):
        if val_first > val_sec:
            right_limit = intersec_sec
            intersec_sec = intersec_first
            val_sec = val_first
            dist_limits *= invphi
            intersec_first = left_limit + invphi2 * dist_limits
            val_first = function(intersec_first)
        else:
            left_limit = intersec_first
            intersec_first = intersec_sec
            val_first = val_sec
            dist_limits *= invphi
            intersec_sec = left_limit + invphi * dist_limits
            val_sec = function(intersec_sec)
    if val_first < val_sec:
        lambda_max = (left_limit + intersec_sec) / 2
    else:
        lambda_max = (right_limit + intersec_first) / 2
    return lambda_max


def ornstein_uhlenbeck(seq_len, start_val, mean_inf, sigma_ou, theta):
    """Discrete-time Euler-Maruyama approximation of an Ornstein-Uhlenbeck
    process

    Args:
        seq_len (int):
            Length of the sequence to be generated using the Ornstein-Uhlenbeck
            process
        start_val (float):
            Value at which the Ornstein-Uhlenbeck process starts
        mean_inf (float):
            Mean value reached after all transient effects have died out
        sigma_ou (float):
            Variance of the Gaussian distribution involved in the
            Euler-Maruyama approximation
        theta (float):
            Factor specifying the convergence speed to the static mean
    Returns:
        x (numpy.ndarray):
            Vector corresponing to a realization of the random process
    """
    x = np.zeros(seq_len)
    x[0] = start_val
    for i in range(1, seq_len):
        x[i] = (1 - theta)  * x[i - 1]
        x[i] += theta * mean_inf + sigma_ou * np.random.normal()
    return x
