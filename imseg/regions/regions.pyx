# cython: boundscheck=False
# cython: wraparound=False


cdef extern from "regions.h":
    void cinit_average(double * arr, ssize_t size, double * thresholds, int nthresholds, double * ave) nogil

def init_average(arr, thresholds, ave):
    cdef ssize_t size = arr.size
    cdef int nthresholds = len(thresholds)
    if arr == ave:
        arr = arr.copy()

    # cinit_average(memoryview(arr), size, thresholds, nthresholds, memoryview(ave))