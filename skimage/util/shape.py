import numpy as np
from numpy.lib import stride_tricks

__all__ = ['view_as_windows']

def view_as_windows(X, win_size=7):
    """Return array of subarrays that simulate a sliding window.

    Each element of the returned array is a subarray representing a separate
    window of `X`. For example, if `X` is

    [ 0  1  2  3  4]
    [ 5  6  7  8  9]
    [10 11 12 13 14]
    [15 16 17 18 19]
    [20 21 22 23 24]

    then `view_as_windows(X, win_size=3)` would return a 4d array:

    [ 0,  1,  2]  [ 1,  2,  3]  [ 2,  3,  4]
    [ 5,  6,  7]  [ 6,  7,  8]  [ 7,  8,  9]
    [10, 11, 12]  [11, 12, 13]  [12, 13, 14]

    [ 5,  6,  7]  [ 6,  7,  8]  [ 7,  8,  9]
    [10, 11, 12]  [11, 12, 13]  [12, 13, 14]
    [15, 16, 17]  [16, 17, 18]  [17, 18, 19]

    [10, 11, 12]  [11, 12, 13]  [12, 13, 14]
    [15, 16, 17]  [16, 17, 18]  [17, 18, 19]
    [20, 21, 22]  [21, 22, 23]  [22, 23, 24]

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
