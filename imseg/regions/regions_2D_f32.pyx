from cython.parallel cimport prange
import numpy as np


# ctypedef long ssize_t

def init(float[:, :] arr, float[:] thresholds, float[:, :] ave, float[:, :] err, float[:, :, :] sdf):
    cdef int nthresholds = len(thresholds), i
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]

    cdef float[:] region_values = np.empty(nthresholds + 1, dtype='float32')
    cdef ssize_t[:] region_count = np.empty(nthresholds + 1, dtype='int64')

    cdef char[:, :] labels = np.empty((nx, ny), dtype='uint8')

    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                sdf[0, x, y] = arr[x, y] - thresholds[0]
                if arr[x, y] <= thresholds[0]:
                    region_values[0] += arr[x, y]
                    region_count[0] += 1
                    labels[x, y] = 0
                elif arr[x, y] > thresholds[nthresholds - 1]:
                    region_values[nthresholds] = arr[x, y]
                    region_count[nthresholds] += 1
                    labels[x, y] = nthresholds
                else:
                    for i in xrange(1, nthresholds):
                        sdf[i, x, y] = arr[x, y] - thresholds[i]
                        if (arr[x, y] > thresholds[i - 1]) and (arr[x, y] <= thresholds[i]):
                            region_values[i] += arr[x, y]
                            region_count[i] += 1
                            labels[x, y] = i
                            break

        for i in prange(nthresholds):
            region_values[i] /= region_count[i]

        for x in prange(nx):
            for y in xrange(ny):
                ave[x, y] = region_values[labels[x, y]]
                err[x, y] = arr[x, y] - ave[x, y]
                for i in xrange(nthresholds):
                    sdf[i, x, y] /= (region_values[i + 1] - region_values[i])


def update(float[:, :] arr, float[:, :, :] sdf, float[:, :] ave, float[:, :] err):
    cdef int nthresholds = sdf.shape[0], i
    cdef ssize_t x, nx = arr.shape[0]
    cdef ssize_t y, ny = arr.shape[1]

    cdef float[:] region_values = np.empty(nthresholds + 1, dtype='float32')
    cdef ssize_t[:] region_count = np.empty(nthresholds + 1, dtype='int64')

    cdef char[:, :] labels = np.empty((nx, ny), dtype='uint8')

    with nogil:
        for x in prange(nx):
            for y in xrange(ny):
                if sdf[0, x, y] <= 0:
                    region_values[0] += arr[x, y]
                    region_count[0] += 1
                    labels[x, y] = 0
                elif sdf[nthresholds - 1, x, y] > 0:
                    region_values[nthresholds] = arr[x, y]
                    region_count[nthresholds] += 1
                    labels[x, y] = nthresholds
                else:
                    for i in xrange(1, nthresholds):
                        if (sdf[i - 1, x, y] > 0) and (sdf[i, x, y] <= 0):
                            region_values[i] += arr[x, y]
                            region_count[i] += 1
                            labels[x, y] = i
                            break

        for i in prange(nthresholds):
            region_values[i] /= region_count[i]

        for x in prange(nx):
            for y in xrange(ny):
                ave[x, y] = region_values[labels[x, y]]
                err[x, y] = arr[x, y] - ave[x, y]

