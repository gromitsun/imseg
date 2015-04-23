from __future__ import absolute_import
import numpy as np
from imseg.regions import regions_2D_f64, regions_3D_f64, regions_2D_f32, regions_3D_f32


def init_regions(arr, thresholds, ave, err, sdf):
    dtype = arr.dtype
    if thresholds.dtype != dtype:
        thresholds = np.asarray(thresholds, dtype=dtype)
    assert ave.dtype == dtype
    assert sdf.dtype == dtype
    ndim = arr.ndim

    if (ndim == 2) and (dtype == 'float32'):
        regions_2D_f32.init(arr, thresholds, ave, err, sdf)
    elif (ndim == 2) and (dtype == 'float64'):
        regions_2D_f64.init(arr, thresholds, ave, err, sdf)
    elif (ndim == 3) and (dtype == 'float32'):
        regions_3D_f32.init(arr, thresholds, ave, err, sdf)
    elif (ndim == 3) and (dtype == 'float64'):
        regions_3D_f64.init(arr, thresholds, ave, err, sdf)


def update_regions(arr, sdf, ave, err):
    dtype = arr.dtype
    assert ave.dtype == dtype
    assert sdf.dtype == dtype
    assert err.dtype == dtype
    ndim = arr.ndim

    if (ndim == 2) and (dtype == 'float32'):
        regions_2D_f32.update(arr, sdf, ave, err)
    elif (ndim == 2) and (dtype == 'float64'):
        regions_2D_f64.update(arr, sdf, ave, err)
    elif (ndim == 3) and (dtype == 'float32'):
        regions_3D_f32.update(arr, sdf, ave, err)
    elif (ndim == 3) and (dtype == 'float64'):
        regions_3D_f64.update(arr, sdf, ave, err)


del absolute_import
