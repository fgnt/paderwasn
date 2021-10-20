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


def ransac(observations,
           est_params,
           fit_select,
           min_cardinality,
           start_cardinality,
           fit_level,
           stop_cardinality,
           max_rounds):
    """Random sample consensus (RANSAC) for outlier rejection for parameter
    optimization

    Args:
        observations (array-like):
            Vector corresponding to the observations from which the wanted
            parameters should be estimated
        est_params (callable):
            Function estimating the wanted parameters (as parameter vector)
            from the observations by minimizing a cost function
            fun(observations) -> params
        fit_select (callable):
            Function to determine which observations fit to the given
            parameter set by comparing the costs corresponding to an
            observation for the given to a given threshold (fit_level)
            fun(observations, params, fit_level) -> consensus
        min_cardinality (int):
            Minimum cardinality needed such that a consensus set is valid
        start_cardinality (int):
            Cardinality of the consensus after initialization
        fit_level (float):
            Threshold determining if an observation fits to the given
            parameter set
        stop_cardinality (float):
            If the cardinality of the consensus set reaches stop_cardinality
            the RANSAC will stop
        max_rounds (int):
            Maximum amount of RANSAC rounds
    Returns:
        A vector corresponding to the estimate of the wanted parameter set
    """
    # The RANSAC method will stop if stop_cardinality percent of the
    # observation fit to the current parameter estimate
    n_obs = len(observations)
    stop_cardinality = int(stop_cardinality * n_obs)

    # The cardinality of a concsnensus set must be larger than cardinality
    # threshold so that it is considered as valid consensus set. After
    # initialization this threshold is set to min_cardinality. After updating
    # the consensus set the threshold corresponds to the cardinality of the
    # current consensus set.
    cardinality_threshold = min_cardinality

    # Randomly initialize the consensus set
    consensus = np.zeros(n_obs, dtype=bool)
    init_consensus = np.random.choice(
        np.arange(n_obs), start_cardinality, False
    )
    consensus[init_consensus] = 1

    n_rounds = 0
    largest_consensus = np.zeros(n_obs)
    while True:
        # Estimate the wanted parameters from the observations currently
        # belonging to the consensus set. Create a new consensus set containing
        # all observations which are considered to fit to the parameter
        # estimate.
        params = est_params(observations[consensus])
        new_consensus = fit_select(observations, params, fit_level)

        if np.sum(new_consensus) > cardinality_threshold:
            # If the cardinality of the new consensus set is larger than the
            # cardinality of the current consensus set the new consensus set
            # will be used to estimate the parameters in the next round of the
            # RANSAC method. Furthermore, the cardinality of the new consensus
            # set will be used as threshold for the cardinality in the next
            # round of the RANSAC method.
            consensus = new_consensus.copy()
            cardinality_threshold = np.sum(new_consensus)
        else:
            # If the cardinality of the new consensus set is not larger than
            # the cardinality of the current consensus set the RANSAC will
            # either stop or the next round of the RANSAC will be started.
            if np.sum(consensus) > np.sum(largest_consensus):
                # If the cardinality of the current consensus set is larger
                # than the cardinality of the currently largest consensus set
                # the current consensus set will be kept.
                largest_consensus = consensus.copy()
            if (cardinality_threshold >= stop_cardinality
                    or n_rounds == max_rounds):
                # Stop the RANSAC if the maximum number of RANSAC rounds is
                # reached or the cardinality of the consensus set fullfills the
                # stopping criterion.
                break
            else:
                # If none of the stopping criteria is met the consensus set is
                # randomly initialized again and the next RANSAC round is
                # started.
                n_rounds += 1
                cardinality_threshold = min_cardinality
                consensus = np.zeros(n_obs, dtype=bool)
                init_consensus = np.random.choice(
                    np.arange(n_obs), start_cardinality, False
                )
                consensus[init_consensus] = 1
    return est_params(observations[largest_consensus])
