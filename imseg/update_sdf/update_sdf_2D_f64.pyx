from cython.parallel cimport prange


cpdef void update_sdf(double[:, :, :] sdf, double[:, :] err, double beta=0.5):
    cdef ssize_t i, ni=sdf.shape[0]
    cdef ssize_t x, nx=sdf.shape[1]
    cdef ssize_t y, ny=sdf.shape[2]
    with nogil:
        for i in xrange(ni):
            for x in prange(nx):
                for y in xrange(ny):
                    sdf[i, x, y] += beta * err[x, y]
