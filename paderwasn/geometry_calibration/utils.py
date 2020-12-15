import numpy as np


def est_rot_mat(pos, ref_pos, weights=None, allow_reflection=False):
    """Rotation matrix estimation for data set matching

    Estimate the rotation matrix needed to project the positions pos onto
    the reference positions ref_pos. We refer to
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf for a detailed description.

    Args:
        pos (ndarray (shape=(dim, n_pos)):
            Array containing the positions to be projected.
        ref_pos (ndarray (shape=(dim, n_pos)):
            Array containing the reference position.
        weights (ndarray):
            1D array containing the weights. If None (default) weighting
            is omitted.
        allow_reflection (bool):
            If True reflections are allowed beside rotation and translation to
            project the positions pos onto the reference positions ref_pos.
            Otherwise, reflections will be compensated.
    """
    assert pos.shape == ref_pos.shape

    if weights is not None:
        centroid = \
            np.sum(np.expand_dims(weights, 0) * pos, keepdims=True, axis=-1)
        centroid /= np.sum(weights) + 1e-9
        centroid_ref = np.sum(
            np.expand_dims(weights, 0) * ref_pos, keepdims=True, axis=-1
        )
        centroid_ref /= np.sum(weights) + 1e-9
        h = (pos - centroid) @ np.diag(weights) @ (ref_pos - centroid_ref).T
    else:
        centroid = np.mean(pos, keepdims=True, axis=-1)
        centroid_ref = np.mean(ref_pos, keepdims=True, axis=-1)
        h = (pos - centroid) @ (ref_pos - centroid_ref).T
    u, _, vh = np.linalg.svd(h)
    rot_mat = (u @ vh).T
    # Compensate reflections if they are not allowed.
    if np.linalg.det(rot_mat) < 0 and not allow_reflection:
        vh[:, -1] *= -1
        rot_mat = (u @ vh).T
    return rot_mat


def est_translation(pos, ref_pos, rot_mat, weights=None):
    """Translation vector estimation for data set matching

    Estimate the translation vector needed to project the positions pos onto
    the reference positions ref_pos. We refer to
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf for a detailed description.

    Args:
        pos (ndarray (shape=(dim, n_pos)):
            Array containing the positions to be projected.
        ref_pos (ndarray (shape=(dim, n_pos)):
            Array containing the reference position.
        rot_mat ((ndarray (shape=(dim, dim)):
            Rotation matrix needed to project pos onto ref_pos.
        weights (ndarray):
            1D array containing the weights. If None (default) weighting
            is omitted.
    """
    assert pos.shape == ref_pos.shape
    if weights is not None:
        centroid = np.sum(
            np.expand_dims(weights, 0) * pos, keepdims=True, axis=-1
        )
        centroid /= np.sum(weights) + 1e-9
        centroid_ref = np.sum(
            np.expand_dims(weights, 0) * ref_pos, axis=-1
        )
        centroid_ref /= np.sum(weights) + 1e-9
    else:
        centroid = np.mean(pos, keepdims=True, axis=-1)
        centroid_ref = np.mean(ref_pos, axis=-1)
    trans_vec = centroid_ref - np.squeeze(rot_mat @ centroid, axis=-1)
    return trans_vec


def est_transl_rot(pos, ref_pos, weights=None, allow_reflection=False):
    """Data set matching

    Estimate the translation vector and rotation matrix needed to project the
    positions pos onto the reference positions ref_pos.

    Args:
        pos (ndarray (shape=(dim, n_pos)):
            Array containing the positions to be projected.
        ref_pos (ndarray (shape=(dim, n_pos)):
            Array containing the reference position.
        weights (ndarray):
            1D array containing the weights. If None (default) weighting
            is omitted.
        allow_reflection (bool):
            If True reflections are allowed beside rotation and translation to
            project the positions pos onto the reference positions ref_pos.
            Otherwise, reflections will be compensated.
    """
    assert pos.shape == ref_pos.shape
    rot_mat = est_rot_mat(pos, ref_pos, weights, allow_reflection)
    trans_vec = est_translation(pos, ref_pos, rot_mat, weights)
    return trans_vec, rot_mat


def map2ref(pos, ref_pos, allow_reflection=False):
    """Mapping of positions into the given reference coordinate system

    Project the positions pos into the cordinate system specified by the
    reference positions ref_pos using a data set matching. This corresponds
    to a projection of the positions pos onto the reference positions ref_pos.

    Args:
        pos (ndarray (shape=(dim, n_pos)):
            Array containing the positions to be projected.
        ref_pos (ndarray (shape=(dim, n_pos)):
            Array containing the reference position.
        allow_reflection (bool):
            If True reflections are allowed beside rotation and translation to
            project the positions pos onto the reference positions ref_pos.
            Otherwise, reflections will be compensated.
    """
    assert pos.shape == ref_pos.shape
    translation, rot_mat = \
        est_transl_rot(pos, ref_pos, allow_reflection=allow_reflection)
    return rot_mat @ pos + translation[:, None]


def rot_mat2rot_angle(rot_mat):
    """ Return the angle corresponding to the given 2D rotation matrix."""
    return np.arctan2(rot_mat[1, 0], rot_mat[0, 0])


def rot_angle2rot_mat(angle):
    """ Return the angle corresponding to the given 2D rotation matrix."""
    rot_mat = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rot_mat
