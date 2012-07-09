# -*- python -*-

__all__ = ['stereogram']

import numpy as np
cimport numpy as cnp

#@cython.boundscheck(False)
#@cython.wraparound(False)
def stereogram(cnp.ndarray[dtype=cnp.float64_t, ndim=2, mode="c"] depth_map,
               int separation=64,
               int depth=10):
    cdef:
        cnp.ndarray[dtype=cnp.float64_t, ndim=2, mode="c"] rand
        cnp.ndarray[dtype=cnp.float64_t, ndim=2, mode="c"] out
        int i, j
        double d
        size_t w, h, jj_left, jj_right

    h = depth_map.shape[0]
    w = depth_map.shape[1]

    rand = np.random.random((h, w))
    out = np.zeros((h, w), dtype=float)

    for i in range(h - 1, 0, -1):
        for j in range(w - 1, 0, -1):
            d = depth_map[i, j]
            jj_left = <int>(j + separation - depth * d)
            jj_right = <int>(j - separation + depth * d)
            if (jj_left < 0 or jj_left >= w):
                # Hack for hidden surface removal
                if (jj_right > 0 and jj_right < w):
                    out[i, j] = rand[i, jj_right]
                else:
                    out[i, j] = rand[i, j]
            else:
                out[i, j] = out[i, jj_left]
    return out
