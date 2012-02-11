import numpy as np
from numpy.lib import stride_tricks

__all__ = ['view_as_windows']

def view_as_windows(X, win_size=7):
    """Re-stride an array to simulate a sliding window.

    Parameters
    ----------
    X : 2D-ndarray
        Input image.
    win_size : int
        Size of the sliding window.

    Returns
    -------
    window : (N, M, win_size, win_size) ndarray
        A view on the original data, representing sliding windows.  Note:
        modifying this view will also modify the original data.

    """
    if not X.ndim == 2:
        raise ValueError('Input images must be 2-dimensional.')

    X = np.ascontiguousarray(X)
    r, c = X.shape

    strides = X.strides
    row_jump, el_jump = strides

    new_strides = (row_jump, el_jump, row_jump, el_jump)
    new_rows = r - win_size + 1
    new_cols = c - win_size + 1
    new_shape = (new_rows, new_cols, win_size, win_size)

    windows = stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return windows
