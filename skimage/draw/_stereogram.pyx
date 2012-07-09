# -*- python -*-
# cython: cdivide=True

__all__ = ['stereogram']

import numpy as np
cimport numpy as cnp

#@cython.boundscheck(False)
#@cython.wraparound(False)
def stereogram(cnp.ndarray[dtype=cnp.float64_t, ndim=2, mode="c"] depth_map,
               float eye_separation_px=72):
    """Generate a stereogram from a depth map.

    Parameters
    ----------
    depth_map : ndarray, dtype float
        Depth map, ranging from 0 (furthest from viewer) to
        1 (closest to viewer).
    eye_separation_px : int
        Number of pixels that separate eyes.  For a typical
        72-dpi display, that is roughly 2.1 * 72 = 151.  However,
        since you're staring cross-eyed, it's closer to 1 * 72.
    
    """
    cdef:
        cnp.ndarray[dtype=cnp.float64_t, ndim=2, mode="c"] out
        int i, j, left_bound, right_bound
        double d, s, ds
        size_t w, h, jj_left, jj_right

    h = depth_map.shape[0]
    w = depth_map.shape[1]

    out = np.random.random((h, w))

    for i in range(h):
        for j in range(w):
            #
            # x    x <- projection on stereogram  -   -
            #  \   \                              | d |
            #    \ \                              |   |
            #      o  <- object                   -   |
            #       \                                 | 10d
            #       \ \                               |
            #       \  \                              |
            #       e   e <- eyes, separated by       -
            #                      eye_separation_px

            d = depth_map[i, j]
            ds = d / (10 - d)
            s = eye_separation_px * (1 - ds) / 2

            jj_left = <int>(j - s - (w//2 - eye_separation_px // 2 - j) * ds)
            jj_right = <int>(j + s - (w//2 + eye_separation_px // 2 - j) * ds)

            left_bound = (jj_left >= 0 and jj_left < w)
            right_bound = (jj_right >= 0 and jj_right < w)

            if left_bound and right_bound:
                out[i, jj_right] = out[i, jj_left]

    return out
