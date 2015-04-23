def test1():
    from imseg.regions.regions_2D_f64 import init_average
    import numpy as np

    a = np.arange(100).reshape(10, 10) / 100.
    b = np.empty_like(a)
    init_average(a, np.array([0.25, 0.75]), b)
    print a
    print b


def test2():
    from imseg.regions.regions_2D_f32 import init_average
    import numpy as np

    a = np.arange(100).reshape(10, 10) / 100.
    a = np.asarray(a, dtype='float32')
    b = np.empty_like(a, dtype='float32')
    init_average(a, np.array([0.25, 0.75], dtype='float32'), b)
    print a
    print b


def test3():
    from imseg.regions.regions_3D_f64 import init_average
    import numpy as np

    a = np.arange(27).reshape(3, 3, 3) / 27.
    b = np.empty_like(a)
    init_average(a, np.array([0.25, 0.75]), b)
    print a
    print b


def test4():
    from imseg.regions.regions_2D_f64 import update
    import numpy as np

    a = np.arange(100).reshape(10, 10) / 100.
    b = np.empty_like(a)

    sdf = np.empty((2, 10, 10))
    sdf[0] = a - 0.25
    sdf[1] = a - 0.75

    update(a, sdf, b)
    print a
    print b


def test5():
    from imseg.regions.regions_3D_f64 import update
    import numpy as np

    a = np.arange(27).reshape(3, 3, 3) / 27.
    b = np.empty_like(a)

    sdf = np.empty((2, 3, 3, 3))
    sdf[0] = a - 0.25
    sdf[1] = a - 0.75

    update(a, sdf, b)
    print a
    print b


def test6():
    from imseg.regions import *
    import numpy as np

    a = np.arange(27).reshape(3, 3, 3) / 27.
    ave = np.zeros_like(a)
    err = np.zeros_like(a)

    sdf = np.zeros((2, 3, 3, 3))
    sdf[0] = a - 0.25
    sdf[1] = a - 0.75

    update_regions(a, sdf, ave, err)
    print a
    print sdf
    print ave
    print err

test6()