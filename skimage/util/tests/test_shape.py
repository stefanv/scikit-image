import numpy as np
from numpy.testing import assert_equal
from skimage.util import view_as_windows

def test_view_as_windows():
    X = np.arange(100).reshape((10, 10))
    W = view_as_windows(X, win_size=7)
    assert_equal(W.shape[:2], (4, 4))

    W = view_as_windows(X, win_size=3)
    assert_equal(W[0, 0], [[0, 1, 2],
                           [10, 11, 12],
                           [20, 21, 22]])


if __name__ == '__main__':
    np.testing.run_module_suite()
