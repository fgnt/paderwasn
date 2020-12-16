from copy import copy
import numpy as np
from paderwasn.geometry_calibration.utils import map2ref


def approximate_mds(dists):
    """Approximate multidimensional scaling (MDS)

    Estimate the inter-node distance matrix from source node distances as
    described in "Iterative Geometry Calibration from Distance Estimates for
    Wireless Acoustic Sensor Networks" (https://arxiv.org/abs/2012.06142).
    Subsequently the inter-node distance matrix is used to estimate the source
    positions via MDS.

    Args:
        dists (ndarray (shape=(n_srcs, n_nodes))):
            Array containing the source node distances
    """
    _, n_nodes = dists.shape
    approx_dist_mat = np.zeros((n_nodes, n_nodes))
    for id1 in range(n_nodes):
        for id2 in range(id1 + 1, n_nodes):
            approx_dist = (np.max(np.abs(dists[:, id1] - dists[:, id2]))
                           + np.min(dists[:, id1] + dists[:, id2])) / 2
            approx_dist_mat[id1, id2] = approx_dist
            approx_dist_mat[id2, id1] = approx_dist
    b = np.eye(n_nodes) - 1 / n_nodes * np.ones((n_nodes, n_nodes))
    b = - (b @ approx_dist_mat ** 2 @ b) / 2
    b = (b + b.T) / 2
    v, e = np.linalg.eigh(b)
    idx = v.argsort()[::-1]
    v = v[idx]
    e = e[:, idx]
    node_positions = e[:, :2] @ np.diag(np.sqrt(v[:2]))
    return node_positions.T


def get_sn_dists(src_pos, node_pos):
    """Estimate all source node distances."""
    _, n_srcs = src_pos.shape
    _, n_nodes = node_pos.shape
    dist_mat = np.linalg.norm(src_pos[:, :, None] - node_pos[:, None], axis=0)
    return dist_mat


def ls_loc(known_pos, dists, weight_dists=None):
    """Least squares localization using distance estimates

    Estimate the either the source or the node positions using the source node
    distances and a known set of positions as described in "Iterative Geometry
    Calibration from Distance Estimates for Wireless Acoustic Sensor Networks"
    (https://arxiv.org/abs/2012.06142).

    Args:
        known_pos (ndarray (shape=(2, n_pos))):
            Array containing the set of known positions. This can be the
            sources' or nodes position.
        dists (ndarray (shape=(n_unkowns, n_pos))):
            Array containing the source node distances.
        weight_dists:
            Array containing the distances used for weighting. If None
            (default) the given source node distances will be used as weights.
    """
    n_unkowns, _ = dists.shape
    positions = np.zeros((2, n_unkowns))
    for j in range(n_unkowns):
        nu = np.argmin(dists[j])
        ref_pos = known_pos[:, nu]
        dists_reduced = np.delete(dists[j], nu)
        known_pos_reduced = np.delete(known_pos, nu, -1) - ref_pos[:, None]
        r = 2 * known_pos_reduced.T
        b = dists[j, nu] ** 2 - dists_reduced ** 2
        b += known_pos_reduced[0] ** 2 + known_pos_reduced[1] ** 2
        if weight_dists is not None:
            weights_reduced = np.delete(weight_dists[j], nu)
        else:
            weights_reduced = dists_reduced
        w = np.diag(1 / weights_reduced ** 2)
        positions[:, j] = np.linalg.solve(r.T @ w @ r, r.T @ w @ b) + ref_pos
    return positions


def fit_select(dists, node_pos, src_pos, outlier_percent):
    """Observation selection used for outlier rejection

    Select the best observations based on how well the observations fit to the
    model defined by the given node positions and source positions.

    Args:
        dists (ndarray (shape=(n_srcs, n_nodes))):
            Array containing the source node distances.
        node_pos (ndarray (shape=(2, n_nodes))):
            Array containing the node positions.
        src_pos (ndarray (shape=(2, n_srcs))):
            Array containing the source positions.
        outlier_percent (float):
            Percentage of observations to be considered as outliers.
    """
    n_srcs, _ = dists.shape
    fit_levels = \
        np.mean(get_sn_dists(src_pos, node_pos) / (dists + 1e-9), axis=-1)
    fit_levels = abs(1 - fit_levels)
    select = \
        np.argsort(fit_levels)[:int(np.ceil((1 - outlier_percent) * n_srcs))]
    return select, fit_levels


def est_geometry(dists, n_iter, n_generations, generation_spread,
                 outlier_percent, lr):
    """Geometry calibration using the GARDE-algorithm

    Estimate the geometry of a wireless acoustic sensor network using the
    GARDE-algorithm which is proposed in "Iterative Geometry Calibration from
    Distance Estimates for Wireless Acoustic Sensor Networks"
    (https://arxiv.org/abs/2012.06142).

    Args:
        dists (ndarray (shape=(n_srcs, n_nodes))):
            Array containing the source node distances.
        n_iter (int):
            Number of iteration rounds.
        n_generations (int):
            Number of annealing rounds.
        generation_spread (float):
            Initial variance of used for simulated annealing.
        outlier_percent (float):
            Percentage of observations to be considered as outliers.
        lr (float):
            Learning factor used for the iterative updates of the positions.
    """
    def iterate(dists, node_pos, src_pos, n_iter, outlier_percent, lr):
        for i in range(1, n_iter):
            src_select, _ = \
                fit_select(dists, node_pos, src_pos, outlier_percent)
            src_pos_fit = src_pos[:, src_select]
            dists_fit = dists[src_select]
            weight_dists = get_sn_dists(src_pos_fit, node_pos)
            node_pos_fit = ls_loc(src_pos_fit, dists_fit.T, weight_dists.T)
            node_pos_update = \
                map2ref(node_pos_fit, node_pos, allow_reflection=True)
            node_pos = lr * node_pos_update + (1 - lr) * node_pos
            src_pos_update = ls_loc(node_pos, dists)
            src_pos = lr * src_pos_update + (1 - lr) * src_pos
        src_select, fit_levels = \
            fit_select(dists, node_pos, src_pos, outlier_percent)
        return node_pos, src_pos, src_select, fit_levels

    # Get first estimate for the nodes' position using approximate MDS
    node_pos = approximate_mds(dists)

    # Get first estimate for the sources' position using LS localization
    src_pos = ls_loc(node_pos, dists)

    # Do the first iteration rounds
    node_pos, src_pos, src_select, fit_levels = \
        iterate(dists, node_pos, src_pos, n_iter, outlier_percent, lr)

    # Use current estimates as best fitting estimates
    n_srcs, _ = dists.shape
    n_best = int(np.ceil((1 - outlier_percent) * n_srcs))
    avg_fit_level_best = np.mean(np.sort(fit_levels)[:n_best])
    node_pos_best = copy(node_pos)
    src_pos_best = copy(src_pos)

    # Simulated annealing
    for g in range(n_generations):
        node_pos, src_pos, src_select, fit_levels = \
            iterate(dists, node_pos, src_pos, n_iter, outlier_percent, lr)

        # Select best fitting positions
        avg_fit_level = np.mean(np.sort(fit_levels)[:n_best])
        if avg_fit_level < avg_fit_level_best:
            node_pos_best = copy(node_pos)
            src_pos_best = copy(src_pos)
            avg_fit_level_best = copy(avg_fit_level)

        node_pos = \
            generation_spread / (g + 1) * np.random.normal(size=node_pos.shape)
        node_pos += node_pos_best
        src_pos = \
            generation_spread / (g + 1) * np.random.normal(size=src_pos.shape)
        src_pos += src_pos_best
    return node_pos_best, src_pos_best
