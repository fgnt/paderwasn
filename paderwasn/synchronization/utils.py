import numpy as np


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
    n_iter = np.ceil(
        np.log(tolerance / dist_limits) / np.log(invphi))
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
