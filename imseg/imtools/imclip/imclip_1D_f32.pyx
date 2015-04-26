from __future__ import print_function
from cython.parallel cimport prange
import numpy as np


cpdef void imclip(float[:] arr, float[:] out, float low_percentage=0.01, float high_percentage=0.99, int verbose=0):
    cdef ssize_t n = arr.size
    cdef ssize_t low_index = int(round(low_percentage * n) - 1)
    cdef ssize_t high_index = int(round(high_percentage * n) - 1)
    cdef float low_value, high_value
    low_value, high_value = \
        np.partition(arr, [low_index, high_index], axis=None)[[low_index, high_index]]
    if verbose:
        print('Clipping %d low pixels with low = %s; and %d pixels with high = %s...'
              % (low_index, low_value, n - high_index, high_value), end=' ')
    clip(arr, out, low_value, high_value)
    if verbose:
        print('Done!')


cpdef inline void clip(float[:] arr, float[:] out, float low_value, float high_value):
    cdef ssize_t x, nx = arr.shape[0]
    if &arr[0] != &out[0]:
        out[...] = arr[...]
    with nogil:
        for x in prange(nx):
            if out[x] < low_value:
                out[x] = low_value
            elif out[x] > high_value:
                out[x] = high_value
                        
                        
cpdef void clip_norm(float[:] arr, float[:] out, float low_value, float high_value):
    cdef ssize_t x, nx = arr.shape[0]
    if &arr[0] != &out[0]:
        out[...] = arr[...]
    with nogil:
        for x in prange(nx):
            if out[x] < low_value:
                out[x] = 0
            elif out[x] > high_value:
                out[x] = 1
            else:
                out[x] = (out[x] - low_value) / (high_value - low_value)