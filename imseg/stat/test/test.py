from imseg.stat.max.max_3D_f64 import mpi_max
import numpy as np


a = np.arange(1000).reshape(10, 10, 10) * 1.
b = mpi_max(a)

print b