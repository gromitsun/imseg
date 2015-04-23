from __future__ import absolute_import, division, print_function
from cython.parallel cimport prange


cpdef void rmbg(double[:, :, :, :] arr, double[:, :, :, :] bg, double[:, :, :, :] out):
    cdef ssize_t a0, na0 = arr.shape[0]
    cdef ssize_t a1, na1 = arr.shape[1]
    cdef ssize_t a2, na2 = arr.shape[2]
    cdef ssize_t a3, na3 = arr.shape[3]
    cdef ssize_t a0b, na0b = bg.shape[0]
    if (na0 % na0b) != 0:
        raise ValueError("The first dimension of arr must be dividable by that of bg.")
    cdef ssize_t m = int(na0 / na0b)
    with nogil:
        for a0 in prange(na0):
            for a1 in xrange(na1):
                for a2 in xrange(na2):
                    for a3 in xrange(na3):
                        out[a0, a1, a2, a3] = arr[a0, a1, a2, a3] - bg[a0 % na0b, a1, a2, a3]


