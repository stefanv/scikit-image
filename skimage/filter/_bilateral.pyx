# -*- python -*-

"""
A cython implementation of the plain (and slow) algorithm for bilateral
filtering.

"""

from numpy.numarray.nd_image import generic_filter

cimport numpy as np
import numpy as np

cimport cython
from cython.parallel import prange, parallel

cdef extern from "math.h" nogil:
    double exp(double x)
    double fabs (double x)

cdef extern from "stdlib.h" nogil:
    int abs(int num)

cdef int GAUSS_SAMP = 32
cdef int GAUSS_IDX_MAX = GAUSS_SAMP - 1

cdef class _Bilat_fcn:
    '''
    This class provides the bilateral filter function to be called by
    ndimage.generic_filter.

    Example
    -------
    import numpy as np
    from scipy import ndimage
    # Initialize the bilateral filter
    >>> bilat = _Bilat_fcn(5.4, 3.0)
    # Prepare images to be filtered
    >>> vec = zeros(50)
    >>> vec[25:] = 100
    >>> img1 = np.resize(vec, (50,50))
    >>> img2 = np.resize(vec, (100, 51))
    >>> img1 += 5.4 * np.random.randn(*img1.shape)
    >>> img2 += 5.4 * np.random.randn(*img2.shape)
    # full accuracy bilateral filtering of img1
    >>> img1_filtered = ndimage.generic_filter(img1, bilat.cfilter, mode='mirror')
    # Reduced accuracy fast filtering of img2
    >>> img2_filtered = ndimage.generic_filter(img2, bilat.fc_filterd)
    '''
    cdef int xy_size, index
    cdef float inten_sig, x_quant
    cdef np.ndarray xy_ker, xy_kerf, gauss_lut, gauss_lutf

    def __init__(self, spat_sig, inten_sig, filter_size=None):
        '''
        Initialise data for the bilateral filter(s) call-back methods

        parameters
        ----------
        spat_sig : scalar
            The standard deviation of the spatial Gaussian convolution kernel.
        inten_sig : scalar
            The standard deviation of the intensity (z axis) Gaussian kernel.
        filter_size : int
            The spatial convolution kernel size. If not specified (None value)
            then filter_size =~ 4*spat_sig
        '''
        if filter_size is not None and filter_size >= 2:
            self.xy_size = int(filter_size)
        else:
            self.xy_size = int(round(spat_sig*4))
            # Make filter size odd
            self.xy_size += 1-self.xy_size%2
        x = np.arange(self.xy_size, dtype=float)
        x = (x-x.mean())**2
        #xy_ker: Spatial convolution kernel
        self.xy_ker = np.exp(-np.add.outer(x,x)/(2*spat_sig**2)).ravel()
        self.xy_ker /= self.xy_ker.sum()
        self.index = self.xy_size**2 // 2

        ## An initialization for LUT instead of a Gaussian function call
        ## (for the fc_filter method)

        x = np.linspace(0,3.0, GAUSS_SAMP)
        self.gauss_lut = np.exp(-x**2/2)
        self.x_quant = 3.0 * inten_sig / GAUSS_IDX_MAX
        self.inten_sig = 2.0 * inten_sig**2

    def info(self):
        '''
        Print internal parameters. For debugging purposes only.
        '''
        print "xy_size:    ", self.xy_size
        print "Z sigma:    ", self.inten_sig
        print "index:      ", self.index
        print "x_quant:    ", self.x_quant
        print "xy_ker:   \n", self.xy_ker
        print "gauss_lut:\n", self.gauss_lut

    ##
    ## Filtering functions
    ##

    def __call__ (self, data):
        '''
        An unoptimized (pure python) implementation of the bilateral kernel

        Parameters
        ---------
         data : A 1D array of a floating point type
             The pixel neighbourhood as a 1D array. Intended to be called from
             ndimage.generic_filter

        Returns : float
          A bilateral weighted average over the pixel neighbourhood.
        '''
        weight = np.exp(-(data-data[self.index])**2/self.inten_sig) * self.xy_ker
        return np.dot(data, weight) / weight.sum()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def cfilter(self, np.ndarray data):
        '''
        An optimized (Cython) implementation of the bilateral kernel

        Parameters
        ---------
         data : A 1D array of a np.float64
             The pixel neighbourhood as a 1D array. Intended to be called from
             ndimage.generic_filter

        Returns : float
          A bilateral weighted average over the pixel neighborhood.
        '''
        cdef np.ndarray kernel = self.xy_ker
        cdef double sigma   = self.inten_sig
        cdef double weight_i, weight, result, centre, dat_i
        cdef double *pdata=<double *>data.data, *pker=<double *>kernel.data
        cdef int i, dim = data.shape[0]
        centre = pdata[self.index]

        weight = 0.0
        result = 0.0
        for i from  0 <= i < dim:
            dat_i = pdata[i]
            weight_i = exp(-(dat_i-centre)**2 / sigma) * pker[i]
            weight += weight_i;
            result += dat_i * weight_i
        return result / weight

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fc_filterd(self, np.ndarray data):
        '''
        A fully optimized (Cython) implementation of the bilateral kernel, which
        trades speed on the expense of numerical accuracy.

        Parameters
        ---------
         data : A 1D array of a np.float64
             The pixel neighbourhood as a 1D array. Intended to be called from
             ndimage.generic_filter

        Returns : float
          A bilateral weighted average over the pixel neighbourhood.

        '''
        cdef np.ndarray kernel = self.xy_ker
        cdef np.ndarray gauss_lut_arr = self.gauss_lut
        cdef double sigma   = self.inten_sig
        cdef double weight_i, weight, result, centre, dat_i
        cdef double *pdata=<double *>data.data, *pker=<double *>kernel.data
        cdef unsigned int i, dim = data.shape[0]
        cdef unsigned int exp_i    # Entry index for the LUT
        cdef double x_quant = self.x_quant
        cdef double *gauss_lut = <double *>gauss_lut_arr.data

        centre = pdata[self.index]
        weight = 0.0
        result = 0.0
# The commented lines parallelize the main loop, but result in a slow
# execution with all cpus occupied (at least on mt PC)
#        with nogil, parallel():
#            for i in prange(dim, schedule='static'):
        for i in range(dim):
                dat_i = pdata[i]
                exp_i = <unsigned int> fabs((dat_i-centre) / x_quant)
                if exp_i > GAUSS_IDX_MAX:
                    continue
                weight_i = gauss_lut[exp_i] * pker[i]
                weight += weight_i;
                result += dat_i * weight_i
        return result / weight

#
# Filtering functions to be called from outside
#

def bilateral(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    '''
    Bilateral filter a 2D array.

    Parameters
    ---------
        mat : 2D array
          Array to be filtered
        xy_sig : scalar
          The sigma of the spatial Gaussian filter.
        z_sig :  scalar
          The sigma of the gray levels Gaussian filter.
        filter_size : scalar
          The size of the spatial filter kernel. If set to None (default)
          or values < 2 --- auto select.
        mode: str
          See scipy.ndimage.generic_filter documentation

    Returns
    -------
    A bilateral filtered mat: A 2D array of the same dimensions as mat and
    a float64 dtype

    Notes
    -----
      1. The spatial filter kernel size is ~4*xy_sig, unless specified otherwise
      2. The algorithm is slow but has a minimal memory footprint

    Example
    ------
    >>> m = np.arange(25.).reshape(5,5)
    >>> mfilt = bilateral(m, 1.0, 3.0, 3)
    >>> m
    array([[  0.,   1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.],
           [ 15.,  16.,  17.,  18.,  19.],
           [ 20.,  21.,  22.,  23.,  24.]])
    >>> mfilt
    array([[  0.6416,   1.4365,   2.4365,   3.4365,   4.2305],
           [  5.0933,   6.    ,   7.    ,   8.    ,   8.9067],
           [ 10.0933,  11.    ,  12.    ,  13.    ,  13.9067],
           [ 15.0933,  16.    ,  17.    ,  18.    ,  18.9067],
           [ 19.7695,  20.5635,  21.5635,  22.5635,  23.3584]])
    '''
    cdef _Bilat_fcn filter_fcn
    filter_fcn = _Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn.cfilter, size =(size,size), mode=mode)

def bilateral_slow(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    '''
    A pure python implementation of the bilateral filter, for documentation
    see bilateral function documentation.
    '''
    filter_fcn = _Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn, size =(size,size), mode=mode)

def bilateral_fast(mat, xy_sig, z_sig, filter_size=None, mode='reflect'):
    '''
    A fast implementation of bilateral filter (speed on the expense of accuracy),
    for documetation see bilateral function documentation.

    Example
    -------
    >>> m = np.arange(25.).reshape(5,5)
    >>> mfilt = bilateral(m, 1.0, 3.0, 3)
    >>> m
    array([[  0.,   1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.],
           [ 15.,  16.,  17.,  18.,  19.],
           [ 20.,  21.,  22.,  23.,  24.]])
    >>> mfiltf = B.bilateral_fast(m, 1.0, 3.0, 3)
    >>> mfiltf
    array([[  0.6623,   1.4643,   2.4643,   3.4643,   4.2521],
           [  5.087 ,   6.    ,   7.    ,   8.    ,   8.913 ],
           [ 10.087 ,  11.    ,  12.    ,  13.    ,  13.913 ],
           [ 15.087 ,  16.    ,  17.    ,  18.    ,  18.913 ],
           [ 19.7479,  20.5357,  21.5357,  22.5357,  23.3377]])
    '''
    cdef _Bilat_fcn filter_fcn
    filter_fcn = _Bilat_fcn(xy_sig, z_sig, filter_size)
    size = filter_fcn.xy_size
    return generic_filter(mat, filter_fcn.fc_filterd, size =(size,size), mode=mode)


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
