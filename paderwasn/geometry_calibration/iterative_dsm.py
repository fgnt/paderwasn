import numpy as np
import paderwasn.geometry_calibration.utils as dsm


def get_rel_src_pos(doas, dists):
    """Determine the relative position of the acoustic sources

    Determine the relative position of the acoustic sources using the given
    DoAs described in the local coordinate system of the sensor nodes and
    source node distances.

    Args:
        doas (ndarray (shape=(n_srcs, n_nodes)):
            Array containing DoAs.
        doas (ndarray (shape=(n_srcs, n_nodes)):
            Array containing source node distances.
    """
    dir_vects = np.asarray([np.cos(doas), np.sin(doas)])
    src_pos = dists[None, :] * dir_vects
    return src_pos


def dsm_calib(rel_src_pos, ref_pos, weights=None):
    """Geometry calibration using data set matching

    Estimate the geometry of a wireless acoustic sensor network based on the
    rotation and translation required to project the relative source positions
    rel_src_pos ont the reference source positions ref_pos.

    Args:
        rel_src_pos (ndarray (shape=(2, n_srcs, n_nodes)):
            Array containing the relative source positions described in the
            local coordinate systems of the sensor nodes.
        ref_pos (ndarray (shape=(2, n_srcs)):
            Array containing the source positions onto which the relative
            source positions have to be projected.
        weights:
            1D array containing the weights. If None (default) weighting
            is omitted.
    """
    _, _, n_nodes = rel_src_pos.shape
    node_pos = np.zeros((2, n_nodes))
    node_orients = np.zeros(n_nodes)
    for node_id in range(n_nodes):
        if weights is not None:
            node_pos[:, node_id], rot_mat = dsm.est_transl_rot(
                rel_src_pos[:, :, node_id], ref_pos, weights[:, node_id]
            )
        else:
            node_pos[:, node_id], rot_mat = \
                dsm.est_transl_rot(rel_src_pos[:, :, node_id], ref_pos)
        node_orients[node_id] = dsm.rot_mat2rot_angle(rot_mat)
    return node_pos, node_orients


def src_localization(rel_src_pos, node_pos, node_orients, weights=None):
    """Geometry calibration using data set matching

    Estimate the geometry of a wireless acoustic sensor network based on the
    rotation and translation required to project the relative source positions
    rel_src_pos ont the reference source positions ref_pos.

    Args:
        rel_src_pos (ndarray (shape=(2, n_srcs, n_nodes)):
            Array containing the relative source positions described in the
            local coordinate systems of the sensor nodes.
        node_pos (ndarray (shape=(2, n_nodes)):
            Array containing the node positions to be used for
            source localization.
        node_orients (ndarray (shape=(2, n_nodes)):
            Array containing the node orientations to be used for
            source localization.
        weights:
            1D array containing the weights. If None (default) weighting
            is omitted.
    """
    _, n_srcs, n_nodes = rel_src_pos.shape
    rot_mats = dsm.rot_angle2rot_mat(node_orients)
    src_pos_candidates = np.einsum('abc, bdc -> adc', rot_mats, rel_src_pos)
    src_pos_candidates += np.expand_dims(node_pos, 1)
    if weights is not None:
        src_pos = \
            np.sum(weights * src_pos_candidates, -1) / np.sum(weights, -1)
    else:
        src_pos = np.mean(src_pos_candidates, -1)
    return src_pos, src_pos_candidates


def fit_select(rel_src_pos, node_pos, node_orients, outlier_percent, error):
    """Observation selection used for outlier rejection

    Select the best observations based on how well the observations fit to the
    model defined by the given node positions and orientatuons.

    Args:
        rel_src_pos (ndarray (shape=(2, n_srcs, n_nodes)):
            Array containing the relative source positions described in the
            local coordinate systems of the sensor nodes.
        node_pos (ndarray (shape=(2, n_nodes)):
            Array containing the node positions to be used for
            source localization.
        node_orients (ndarray (shape=(2, n_nodes)):
            Array containing the node orientations to be used for
            source localization.
        weights:
            1D array containing the weights. If None (default) weighting
            is omitted.
        outlier_percent (float):
            Percentage of observations to be considered as outliers.
        error (ndarray (shape=(2, n_srcs, n_nodes)):
            Distances between the observations after being projected and the
            common source position.
    """
    _, n_srcs, n_nodes = rel_src_pos.shape
    n_select = int((1 - outlier_percent) * n_srcs)
    devs = np.zeros((n_srcs, n_nodes, n_nodes - 1))
    rot_mats = dsm.rot_angle2rot_mat(node_orients)
    src_pos_candidates = np.einsum('abc, bdc -> adc', rot_mats, rel_src_pos)
    src_pos_candidates += np.expand_dims(node_pos, 1)
    for node_id in range(n_nodes):
        dev = (src_pos_candidates[:, :, node_id, None]
               - np.delete(src_pos_candidates[:, :], node_id, axis=-1))
        devs[:, node_id] = np.linalg.norm(dev, axis=0)
    dist2neighbors = np.mean(devs, -1)
    return np.argsort(error + dist2neighbors, axis=0)[:n_select]


def est_geometry(doas, dists, outlier_percent=.5, conv_th=1e-3, max_iter=100,
                 wls=True, wls_src_loc=True):
    """Geometry calibration using iterative data set matching

    Estimate the geometry of a wireless acoustic sensor network by iterative
    data set matching as described in "Geometry Calibration in Wireless
    Acoustic Sensor Networks Utilizing DoA and Distance Information" (Submitted
    to the EURASIP Journal on Audio, Speech, and Music Processing)

    Args:
        doas (ndarray (shape=(n_srcs, n_nodes)):
            Array containing DoAs.
        doas (ndarray (shape=(n_srcs, n_nodes)):
            Array containing source node distances.
        outlier_percent (float):
            Percentage of observations to be considered as outliers.
        conv_th (float):
            Threshold to determine if the geometry is converged by means of the
            change of the cost
        max_iter (int):
            Maximum number of iterations of the iterative data set matching and
            source localization procedure.
        wls:
            If True variance dependent weights are used for data set matching.
        wls_src_loc:
            If True variance dependent weights are used for
            source localization.
    """
    n_srcs, n_nodes = doas.shape
    rel_src_pos = get_rel_src_pos(doas, dists)
    src_pos = rel_src_pos[:, :, 0]
    node_pos, node_orients = dsm_calib(rel_src_pos, src_pos)
    src_pos, src_pos_candidates = \
        src_localization(rel_src_pos, node_pos, node_orients)
    devs = np.linalg.norm(
        np.expand_dims(src_pos, -1) - src_pos_candidates, axis=0
    )
    weights = 1 / (devs + 1e-3)
    n_it = 0
    old_costs = 1e-9
    while True:
        if wls:
            node_pos, node_orients = dsm_calib(rel_src_pos, src_pos, weights)
        else:
            node_pos, node_orients = dsm_calib(rel_src_pos, src_pos)
        if wls_src_loc:
            src_pos, src_pos_candidates = src_localization(
                rel_src_pos, node_pos, node_orients, weights
            )
        else:
            src_pos, src_pos_candidates = src_localization(
                rel_src_pos, node_pos, node_orients
            )
        devs = np.linalg.norm(
            np.expand_dims(src_pos, -1) - src_pos_candidates, axis=0
        )
        if wls:
            weights = 1 / (devs + 1e-3)
            new_costs = np.sum(weights * devs ** 2)
        else:
            new_costs = np.sum(devs ** 2)
        delta_costs = np.abs(old_costs - new_costs)
        old_costs = new_costs
        n_it += 1
        if delta_costs < conv_th or n_it == max_iter:
            break
    n_it = 0
    old_costs = 1e-9
    while True:
        if outlier_percent != 0:
            selection = fit_select(
                rel_src_pos, node_pos, node_orients, outlier_percent, devs
            )
            mask_fit = np.zeros((n_srcs, n_nodes))
            n_selected, _ = selection.shape
            for i in range(n_selected):
                for node_id in range(n_nodes):
                    mask_fit[selection[i, node_id], node_id] = 1
        else:
            mask_fit = np.ones((n_srcs, n_nodes))
        if wls:
            node_pos, node_orients = \
                dsm_calib(rel_src_pos, src_pos, mask_fit * weights)
        else:
            node_pos, node_orients = dsm_calib(rel_src_pos, src_pos)
        if wls_src_loc:
            src_pos, src_pos_candidates = src_localization(
                rel_src_pos, node_pos, node_orients, weights
            )
        else:
            src_pos, src_pos_candidates = src_localization(
                rel_src_pos, node_pos, node_orients
            )
        devs = np.linalg.norm(
            np.expand_dims(src_pos, -1) - src_pos_candidates, axis=0
        )
        if wls:
            weights = 1 / (devs + 1e-3)
            new_costs = np.sum(weights * mask_fit * devs ** 2)
        else:
            weights = np.ones((n_srcs, n_nodes))
            new_costs = np.sum(mask_fit * devs ** 2)
        delta_costs = np.abs(old_costs - new_costs)
        old_costs = new_costs
        n_it += 1
        if delta_costs < conv_th or n_it == max_iter:
            src_pos, _ = src_localization(
                rel_src_pos, node_pos, node_orients, weights
            )
            return node_pos, node_orients, src_pos, mask_fit
