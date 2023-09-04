import numpy as np


def get_dominant_eigenvector(mat, init=None, n_iter=10):
    """Calculates the dominant eigenvectors of a set of  matrices
     via power iterations

    Args:
        mat (numpy.ndarray):
            Matrices (Array of shape (..., M, M)) for which the dominant
            eigenvectors should be calculated
        init (numpy.ndarray):
            Initial values (Array of shape (...,  M)) of the  vector used for
            the power iterations. If None a vector of ones is used
            as initialization.
        n_iter (int):
            Number of power iterations. If init is not None a single
            iteration is performed.

    Returns:
        eigen_vect (numpy.ndarray):
            The dominant eigenvectors of the given set of matrices (array
            with shape (..., M))
    """
    c = mat.shape[-1]
    if init is not None:
        # If an initialization is given a single power iteration is performed
        eigen_vect = init.copy()
        mat_vect_prod = np.einsum('... c d, ... d -> ... c', mat, eigen_vect)
        denomintator = np.maximum(
            np.linalg.norm(mat_vect_prod, axis=-1, keepdims=True),
            np.finfo(np.float64).eps
        )
        eigen_vect = mat_vect_prod / denomintator
    else:
        eigen_vect = np.ones(mat.shape[:-1]) / np.sqrt(c)
        for _ in range(n_iter):
            mat_vect_prod = \
                np.einsum('... c d, ... d -> ... c', mat, eigen_vect)
            denomintator = np.maximum(
                np.linalg.norm(mat_vect_prod, axis=-1, keepdims=True),
                np.finfo(np.float64).eps
            )
            eigen_vect = mat_vect_prod / denomintator
    return eigen_vect


def get_eigenvalue(mat, eigen_vect):
    """Calculates the eigenvalues belonging to the given eigenvectors of a
    set of  matrices

    Args:
        mat (numpy.ndarray):
            Matrices (Array of shape (..., M, M)) for which  the dominant
            eigenvectors should be calculated
        eigen_vect(numpy.ndarray):
            Eigenvectors (Array of shape (..., , M)) for which the
            corresponding eigenvalue should be calculated.

    Returns:
        eigen_value (numpy.ndarray):
            The eigenvalues belonging to the given eigenvectors of a set
            of matrices.
    """
    mat_vect_prod = np.einsum('... c d, ... d -> ... c', mat, eigen_vect)
    eigen_value = \
        np.einsum('... c , ... c -> ...', eigen_vect.conj(), mat_vect_prod)
    eigen_value /= np.maximum(
        np.einsum('... c , ... c -> ...', eigen_vect.conj(), eigen_vect),
        np.finfo(np.float64).eps
    )
    return eigen_value
