from __future__ import absolute_import, division, print_function
from cython.parallel cimport prange


cpdef void rmbg(float[:, :, :] arr, float[:, :, :] bg, float[:, :, :] out):
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]
    cdef ssize_t z, nz = arr.shape[2]
    cdef ssize_t xb, nxb = bg.shape[0]
    if (nx % nxb) != 0:
        raise ValueError("The first dimension of arr must be dividable by that of bg.")
    cdef ssize_t m = int(nx / nxb)
    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    out[x, y, z] = arr[x, y, z] - bg[x % nxb, y, z]


