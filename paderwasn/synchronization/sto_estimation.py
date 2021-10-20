import numpy as np

from paderwasn.synchronization.utils import ransac


def ransac_sto_est(observations,
                   est_params,
                   fit_select,
                   min_cardinality=30,
                   start_cardinality=1,
                   fit_level=2.5,
                   stop_cardinality=.7,
                   max_iter=1000):
    """Wrapper function to adjust the general RANSAC implementation to the STO
    estimation problem
    """
    return ransac(observations, est_params, fit_select,  min_cardinality,
                  start_cardinality, fit_level,  stop_cardinality, max_iter)


def est_sto(sig_shifts,
            dists,
            dists_ref,
            sampling_rate=16000,
            sound_velocity=343):
    """Robust estimation of the sampling time offset (STO)

    Estimate the STO from a given set of signal shift and source-microphone
    distance estimates as described in "Online Estimation of Sampling Rate
    Offset in Wireless Acoustic Sensor Networks with Packet Loss". For outlier
    rejection the STO estimation method is embedded in a RANSAC method.
    Estimates of the source-microphone distances are used as external
    information source to be able to distinguish between the influence of the
    STO and time differences of flight (TDOFs).

    Args:
        sig_shifts (array-like):
            Vector corresponding to estimates of the inter-signal shifts
        dists (array-like):
            Vector corresponding to estimates of the distance between an
            acoustic source and the microphone whose signal should be
            synchronized
        dists_ref (array-like):
            Vector corresponding to estimates of the distance between an
            acoustic source and the microphone whose signals is used as
            reference for synchronization
        sampling_rate (float):
            Nominal sampling frequency
        sound_velocity (float):
            Speed of sound
    Returns:
        Estimate of the STO
    """
    def _solve_ls_problem(sto_obs):
        """Estimate the STO from the STO observations by solving an least
        squares (LS) problem

        Args:
            sto_obs (array-like):
                Observed STO values
        Returns:
            Estimate for the STO
        """
        return np.mean(sto_obs)

    def _fit_select(sto_obs, sto_est, fit_level):
        """Select all STO observations which fit to the given STO estimate

        Args:
            sto_obs (array-like):
                Vector corresponding to the STO observations
            sto_est (float):
                Estimate of the STO
            fit_level (float):
                Threshold used to decide whether an observation fits to the
                given STO estimate. If the absolute deviation of the
                observation is not larger than fit_level it will be considered
                that the observation fits to the STO estimate.
        Returns:
            Vector corresponding to a binary mask representing if an
            observation fits to the STO estimate. If the observation fits to
            the STO estimate, the corresponding element of the vector is set
            to 1.
        """
        return abs(sto_obs - sto_est) <= fit_level

    sig_shifts = np.asarray(sig_shifts)
    dists = np.asarray(dists)
    dists_ref = np.asarray(dists_ref)

    tdofs = (dists - dists_ref) / sound_velocity * sampling_rate
    sto = ransac_sto_est(sig_shifts - tdofs, _solve_ls_problem, _fit_select)
    return sto
