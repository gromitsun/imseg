from cython.parallel cimport prange
import numpy as np


# ctypedef long ssize_t

def init(double[:, :, :] arr, double[:] thresholds, double[:, :, :] ave, double[:, :, :] err, double[:, :, :, :] sdf):
    cdef int nthresholds = len(thresholds), i
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]
    cdef ssize_t z, nz = arr.shape[2]

    cdef double[:] region_values = np.empty(nthresholds + 1, dtype='float64')
    cdef ssize_t[:] region_count = np.empty(nthresholds + 1, dtype='int64')

    cdef char[:, :, :] labels = np.empty((nx, ny, nz), dtype='uint8')

    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    sdf[0, x, y, z] = arr[x, y, z] - thresholds[0]
                    if arr[x, y, z] <= thresholds[0]:
                        region_values[0] += arr[x, y, z]
                        region_count[0] += 1
                        labels[x, y, z] = 0
                    elif arr[x, y, z] > thresholds[nthresholds - 1]:
                        region_values[nthresholds] = arr[x, y, z]
                        region_count[nthresholds] += 1
                        labels[x, y, z] = nthresholds
                    else:
                        for i in xrange(1, nthresholds):
                            sdf[i, x, y, z] = arr[x, y, z] - thresholds[i]
                            if (arr[x, y, z] > thresholds[i - 1]) and (arr[x, y, z] <= thresholds[i]):
                                region_values[i] += arr[x, y, z]
                                region_count[i] += 1
                                labels[x, y, z] = i
                                break

        for i in prange(nthresholds):
            region_values[i] /= region_count[i]

        for x in prange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    ave[x, y, z] = region_values[labels[x, y, z]]
                    err[x, y, z] = arr[x, y, z] - ave[x, y, z]
                    for i in xrange(nthresholds):
                        sdf[i, x, y, z] /= region_values[i + 1] - region_values[i]


def update(double[:, :, :] arr, double[:, :, :, :] sdf, double[:, :, :] ave, double[:, :, :] err):
    cdef int nthresholds = sdf.shape[0], i
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]
    cdef ssize_t z, nz = arr.shape[2]

    cdef double[:] region_values = np.empty(nthresholds + 1, dtype='float64')
    cdef ssize_t[:] region_count = np.empty(nthresholds + 1, dtype='int64')

    cdef char[:, :, :] labels = np.empty((nx, ny, nz), dtype='uint8')

    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    if sdf[0, x, y, z] <= 0:
                        region_values[0] += arr[x, y, z]
                        region_count[0] += 1
                        labels[x, y, z] = 0
                    elif sdf[nthresholds - 1, x, y, z] > 0:
                        region_values[nthresholds] = arr[x, y, z]
                        region_count[nthresholds] += 1
                        labels[x, y, z] = nthresholds
                    else:
                        for i in xrange(1, nthresholds):
                            if (sdf[i - 1, x, y, z] > 0) and (sdf[i, x, y, z] <= 0):
                                region_values[i] += arr[x, y, z]
                                region_count[i] += 1
                                labels[x, y, z] = i
                                break

        for i in prange(nthresholds):
            region_values[i] /= region_count[i]
            
        for x in prange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    ave[x, y, z] = region_values[labels[x, y, z]]
                    err[x, y, z] = arr[x, y, z] - ave[x, y, z]