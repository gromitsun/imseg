from __future__ import absolute_import
from imseg.update_sdf import update_sdf_2D_f64, update_sdf_2D_f32, update_sdf_3D_f64, update_sdf_3D_f32


def update_sdf(sdf, err, beta=0.5):

    dtype = err.dtype
    ndim = err.ndim
    if ndim not in [2, 3]:
        raise ValueError("The input err array must be 2D or 3D")
    # check dimension agreement
    assert sdf.ndim == ndim + 1
    assert sdf.dtype == dtype

    #--- check datatype ---
    if dtype == 'float32':
        f32 = True
        f64 = False
    elif dtype == 'float64':
        f32 = False
        f64 = True
    else:
        raise TypeError('Only 32-bit and 64-bit arrays are accepted')

    if (ndim == 2) and f32:
        update_sdf_2D_f32.update_sdf(sdf, err, beta)
    elif (ndim == 2) and f64:
        update_sdf_2D_f64.update_sdf(sdf, err, beta)
    elif (ndim == 3) and f32:
        update_sdf_3D_f32.update_sdf(sdf, err, beta)
    elif (ndim == 3) and f64:
        update_sdf_3D_f64.update_sdf(sdf, err, beta)

del absolute_import
