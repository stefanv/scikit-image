import numpy as np

from skimage.filter import bilateral, bilateral_fast
from skimage.filter._bilateral import bilateral_slow

def test_slow_to_normal():
    '''
    Verify that bilateral_slow and bilateral return the same result
    '''
    mat = np.ones((50,50))
    mat[:25] = 100.0
    mat += np.random.randn(*mat.shape)
    mat_filt_norm = bilateral(mat, 3.5, 3.0, 13)
    mat_filt_slow = bilateral_slow(mat, 3.5, 3.0, 13)
    assert np.absolute(mat_filt_slow - mat_filt_slow).max() < 1.0E-4

def test_normal_to_fast():
    '''
    Verify that the fast implementation is good enough.
    '''
    mat = np.ones((50,50))
    mat[:25] = 100.0
    mat += np.random.randn(*mat.shape)
    mat_filt_norm = bilateral(mat, 3.5, 2.0, 13)
    mat_filt_fast = bilateral_fast(mat, 3.5, 2.0, 13)
    assert np.absolute(mat_filt_norm - mat_filt_fast).max() < 0.07

if __name__ == "__main__":
    np.testing.run_module_suite()
