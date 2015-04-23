import numpy as np
# cimport numpy as np
# from cython.parallel cimport prange
# from mpi4py cimport mpi_c, MPI
from mpi4py import MPI


def mpi_max(double[:, :, :] arr):
    comm = MPI.COMM_WORLD
    cdef int nprocs = comm.Get_size()
    cdef int rank = comm.Get_rank()
    cdef ssize_t nz = arr.shape[0]

    cdef ssize_t chunk_size = int(nz / (nprocs - 1))

    cdef double[:] mymax = np.array([np.max(arr[rank * chunk_size: (rank + 1) * chunk_size])]), max = np.array([0.0], dtype='float64')
    comm.Reduce([mymax, MPI.DOUBLE], [max, MPI.DOUBLE], op=MPI.MAX, root=0)

    if rank == 0:
        return max[0]
