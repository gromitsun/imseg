from __future__ import absolute_import
from imseg.imtools.imclip import imclip_2D_f32, imclip_2D_f64, imclip_3D_f32, imclip_3D_f64


def imclip(arr, out=None, low_percentage=0.01, high_percentage=0.99, verbose=False):
    import numpy as np
    dtype = arr.dtype
    ndim = arr.ndim
    verbose = int(verbose)

    if out is None:
        out = np.empty_like(arr, dtype=dtype)
        ret = True
    else:
        assert out.dtype == dtype
        ret = False

    if (ndim == 2) and (dtype == 'float32'):
        imclip_2D_f32.imclip(arr, out, low_percentage, high_percentage, verbose)
    elif (ndim == 2) and (dtype == 'float64'):
        imclip_2D_f64.imclip(arr, out, low_percentage, high_percentage, verbose)
    elif (ndim == 3) and (dtype == 'float32'):
        imclip_3D_f32.imclip(arr, out, low_percentage, high_percentage, verbose)
    elif (ndim == 3) and (dtype == 'float64'):
        imclip_3D_f64.imclip(arr, out, low_percentage, high_percentage, verbose)

    if ret:
        return out