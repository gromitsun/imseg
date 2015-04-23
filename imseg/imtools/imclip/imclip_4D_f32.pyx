from __future__ import print_function
from cython.parallel cimport prange
import numpy as np


cpdef void clip_norm(float[:, :, :, :] arr, float[:, :, :, :] out, float low_value, float high_value):
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]
    cdef ssize_t z, nz = arr.shape[2]
    cdef ssize_t t, nt = arr.shape[3]
    if &arr[0, 0, 0, 0] != &out[0, 0, 0, 0]:
        out[...] = arr[...]
    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    for t in xrange(nt):
                        if out[x, y, z, t] < low_value:
                            out[x, y, z, t] = 0
                        elif out[x, y, z, t] > high_value:
                            out[x, y, z, t] = 1
                        else:
                            out[x, y, z, t] = (out[x, y, z, t] - low_value) / (high_value - low_value)