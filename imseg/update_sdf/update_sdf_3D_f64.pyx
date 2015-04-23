from cython.parallel cimport prange


cpdef void update_sdf(double[:, :, :, :] sdf, double[:, :, :] err, double beta=0.5):
    cdef ssize_t i, ni=sdf.shape[0]
    cdef ssize_t x, nx=sdf.shape[1]
    cdef ssize_t y, ny=sdf.shape[2]
    cdef ssize_t z, nz=sdf.shape[3]
    with nogil:
        for i in xrange(ni):
            for x in prange(nx):
                for y in xrange(ny):
                    for z in xrange(nz):
                        sdf[i, x, y, z] += beta * err[x, y, z]
