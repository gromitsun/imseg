from __future__ import print_function
from cython.parallel cimport prange
import numpy as np


cpdef void imclip(double[:, :] arr, double[:, :] out, double low_percentage=0.01, double high_percentage=0.99, int verbose=0):
    cdef ssize_t n = arr.size
    cdef ssize_t low_index = int(round(low_percentage * n) - 1)
    cdef ssize_t high_index = int(round(high_percentage * n) - 1)
    cdef double low_value, high_value
    low_value, high_value = \
        np.partition(arr, [low_index, high_index], axis=None)[[low_index, high_index]]
    if verbose:
        print('Clipping %d low pixels with low = %s; and %d pixels with high = %s...'
              % (low_index, low_value, n - high_index, high_value), end=' ')
    clip(arr, out, low_value, high_value)
    if verbose:
        print('Done!')


cdef inline void clip(double[:, :] arr, double[:, :] out, double low_value, double high_value):
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]
    if &arr[0, 0] != &out[0, 0]:
        out[...] = arr[...]
    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                if out[x, y] < low_value:
                    out[x, y] = low_value
                elif out[x, y] > high_value:
                    out[x, y] = high_value