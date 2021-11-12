from pathlib import Path

import numpy as np


def load_binary(path, dtype=np.float32):
    """Loads a binary file and returns it as a numpy.ndarray.

    Args:
        path: String or ``pathlib.Path`` object.
        dtype: Data type used to determine the size and byte-order
            of the items in the file. The returned array will have the same
            data type.

    Returns:
        Content of the binary file
    """
    assert isinstance(path, (str, Path)), path
    path = Path(path).expanduser()

    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype)
    return data


def reverb_signal(signal, rirs):
    return np.asarray([np.convolve(signal, rir) for rir in rirs])
