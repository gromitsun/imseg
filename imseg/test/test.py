from imseg.segment import ImSeg, kwarg


def test():
    import h5py
    from skimage.filter import threshold_otsu
    from imseg.image_tools import imclip, normalize

    f = h5py.File('/Users/yue/data/obj.hdf5', 'r')
    im = f['object'][:, 350:900, 512]
    f.close()

    # im = anisodiff(im, niter=10)

    im = imclip(im, verbose=True)
    im = normalize(im, mode='median')
    thresholds = threshold_otsu(im)
    print(thresholds)

    # im_seg = ImSeg(im, out_path='/Users/yue/data/segment/sdf_anisodiff_2d_6-2/', debug=True, verbose=True)
    # im_seg.initialize(thresholds=thresholds,
    # diff_sdf_kwargs=kwarg(niter=5, kappa=50, gamma=0.1),
    # diff_error_kwargs=kwarg(niter=25, kappa=50, gamma=0.1),
    #                   diff_ave_kwargs=kwarg(Dt=0.5),
    #                   reinit_sdf_kwargs=kwarg(niter=5, delta_t=0.1),
    #                   init_reinit_sdf_kwargs=kwarg(niter=50, delta_t=0.1))

    # im_seg = ImSeg(im, out_path='/Users/yue/data/segment/sdf_diff_2d_4/', debug=True, verbose=True)
    # im_seg.initialize(thresholds=thresholds,
    #                   diff_sdf_kwargs=kwarg(Dt=0.5),
    #                   diff_error_kwargs=kwarg(Dt=2.5),
    #                   diff_ave_kwargs=kwarg(Dt=0.5),
    #                   reinit_sdf_kwargs=kwarg(niter=5, delta_t=0.1),
    #                   init_reinit_sdf_kwargs=kwarg(niter=50, delta_t=0.1))

    # # using im3D
    im_seg = ImSeg(im)
    im_seg.init(thresholds=thresholds,
                diff_sdf_kwargs=kwarg(it=5, dt=0.1),
                diff_error_kwargs=kwarg(it=20, dt=0.25),
                diff_ave_kwargs=kwarg(it=5, dt=0.1),
                reinit_sdf_kwargs=kwarg(niter=5, dt=0.1),
                init_reinit_sdf_kwargs=kwarg(niter=50, dt=0.1),
                out_path='/Users/yue/data/segment/sdf_diff_2d_7/', debug=True, verbose=True)

    im_seg.initialize()

    # im_seg.init_vars(thresholds)
    # im_seg.init_regions()
    # im_seg.init_sdf()
    # im_seg.calc_average()
    # im_seg.calc_error()
    # for i in xrange(10):
    # im_seg.reinit_sdf(niter=1)
    # im_seg.show()


    im_seg.iterate(200)
    im_seg.show()


def test_3d():
    import h5py
    from skimage.filter import threshold_otsu
    from imseg.image_tools import imclip, normalize
    # from imseg.imtools.imclip import imclip as imclip2


    f = h5py.File('/Users/yue/data/obj.hdf5', 'r')
    im = f['object'][:, 350:900, 350:900]
    f.close()

    # im = anisodiff(im, niter=10)

    im = imclip(im, verbose=False)
    im = normalize(im, mode='median')
    thresholds = threshold_otsu(im)
    print('Threshold = {}'.format(thresholds))

    im_seg = ImSeg(im)
    im_seg.init(thresholds=thresholds,
                diff_sdf_kwargs=kwarg(it=5, dt=0.1),
                diff_error_kwargs=kwarg(it=20, dt=0.25),
                diff_ave_kwargs=kwarg(it=5, dt=0.1),
                reinit_sdf_kwargs=kwarg(niter=5, dt=0.1),
                init_reinit_sdf_kwargs=kwarg(niter=50, dt=0.1),
                im_slice=(slice(None), slice(None), 512 - 350),
                out_path='/Users/yue/data/segment/sdf_diff_3d_1/', debug=True, verbose=True)
    im_seg.initialize()

    for i in xrange(10):
        im_seg.iterate(10)
        im_seg.save()


def test3():
    import numpy as np
    from mayavi import mlab
    sdf = np.load('/Users/yue/data/segment/sdf_diff_3d_1/imseg_iter_100.npz')['sdf']
    mlab.contour3d(sdf[0], contours=[0])
    # mlab.savefig('/Users/yue/data/segment/sdf_diff_3d_1/imseg_iter_100.png')
    mlab.show()


# def test2():
#     import matplotlib.pyplot as plt
#
#     a = np.sin(np.arange(100) / 15.)
#     im_seg = ImSeg(a)
#     im_seg.initialize(0)
#     # im_seg.init_vars(0)
#     for i in xrange(10):
#         im_seg.reinit_sdf(100, delta_t=0.1)
#         plt.figure()
#         plt.plot(im_seg.im)
#         plt.figure()
#         plt.plot(im_seg.sdf[0])
#
#         plt.show()
#
#
# def test3():
#     """2d simulated data; using diffusion.diffuse"""
#     from skimage.morphology import disk
#
#     im = np.zeros((512, 512))
#     im[100:201, 100:201] = disk(50)
#     noise = np.random.randn(512, 512)
#
#     im = im + noise
#
#     thresholds = 0.5
#
#     im_seg = ImSeg(im, debug=True)
#     im_seg.initialize(thresholds=thresholds,
#                       diff_sdf_kwargs=kwarg(niter=10, delta_t=0.01),
#                       diff_error_kwargs=kwarg(niter=10, delta_t=0.01),
#                       diff_ave_kwargs=kwarg(niter=0),
#                       reinit_sdf_kwargs=kwarg(niter=10, delta_t=0.002),
#                       init_reinit_sdf_kwargs=kwarg(niter=20, delta_t=0.1))
#
#     im_seg.iterate(35)
#     im_seg.show()
#
#
# def test4():
#     """2d simulated data; using green_diffusion.diffuse_2"""
#     from skimage.morphology import disk
#
#     im = np.zeros((512, 512))
#     im[100:201, 100:201] = disk(50)
#     noise = np.random.randn(512, 512)
#
#     im = im + noise
#
#     thresholds = 0.5
#
#     im_seg = ImSeg(im, debug=True)
#     im_seg.initialize(thresholds=thresholds,
#                       diff_sdf_kwargs=kwarg(Dt=1),
#                       diff_error_kwargs=kwarg(Dt=2),
#                       diff_ave_kwargs=kwarg(Dt=0),
#                       reinit_sdf_kwargs=kwarg(niter=10, delta_t=0.002),
#                       init_reinit_sdf_kwargs=kwarg(niter=20, delta_t=0.1))
#
#     im_seg.iterate(35)
#     im_seg.show()
#
#
# def test5():
#     """2d simulated data; using anisodiff.anisodiff"""
#     from skimage.morphology import disk
#
#     im = np.zeros((512, 512))
#     im[100:201, 100:201] = disk(50)
#     noise = np.random.randn(512, 512)
#
#     im = im + noise
#
#     thresholds = 0.5
#
#     im_seg = ImSeg(im, debug=True, out_path='/Users/yue/temp/segtest/')
#     im_seg.initialize(thresholds=thresholds,
#                       diff_sdf_kwargs=kwarg(niter=50),
#                       diff_error_kwargs=kwarg(niter=50),
#                       diff_ave_kwargs=kwarg(Dt=0),
#                       reinit_sdf_kwargs=kwarg(niter=10, delta_t=0.002),
#                       init_reinit_sdf_kwargs=kwarg(niter=20, delta_t=0.1))
#     im_seg.iterate(35)
#     im_seg.show()
#
#
# def test6():
#     """reinit_sdf 1d"""
#     a = np.zeros(512)
#     a[128:256] = 1
#     a[377:436] = 1
#     a[300] = 1
#     im_seg = ImSeg(a, verbose=True)
#     im_seg.initialize(thresholds=0.5,
#                       diff_sdf_kwargs=kwarg(Dt=0.1),
#                       diff_error_kwargs=kwarg(Dt=0.1),
#                       diff_ave_kwargs=kwarg(Dt=0),
#                       reinit_sdf_kwargs=kwarg(niter=5000, delta_t=0.1),
#                       init_reinit_sdf_kwargs=kwarg(niter=5000, delta_t=0.1))
#     plt.figure()
#     plt.plot(im_seg.sdf[0])
#     for i in xrange(10):
#         # im_seg.iterate(niter=1)
#         im_seg.reinit_sdf(niter=5000, delta_t=0.2)
#         plt.figure()
#         plt.plot(im_seg.sdf[0])
#         plt.show()
#
#
# def test7():
#     from time import sleep
#
#     for i in xrange(100000):
#         print('\rasdf %d' % i, end='')
#         stdout.flush()
#         sleep(0)
#     print('')
#
#
# def test8():
#     from time import time
#     import h5py
#     from skimage.filter import threshold_otsu
#     from imseg.image_tools import imclip, normalize
#
#     f = h5py.File('/Users/yue/data/obj.hdf5', 'r')
#     im = f['object'][:, 350:900, 512]
#     f.close()
#
#     # im = anisodiff(im, niter=10)
#
#     im = imclip(im, verbose=True)
#     im = normalize(im, mode='median')
#     thresholds = threshold_otsu(im)
#     print(thresholds)
#
#     im_seg = ImSeg(im, out_path='/Users/yue/data/segment/sdf_anisodiff_2d_6-1/', debug=True, verbose=True)
#     im_seg.initialize(thresholds=thresholds,
#                       diff_sdf_kwargs=kwarg(niter=5, gamma=50, kappa=0.1),
#                       diff_error_kwargs=kwarg(niter=25, gamma=50, kappa=0.1),
#                       diff_ave_kwargs=kwarg(Dt=0.5),
#                       reinit_sdf_kwargs=kwarg(niter=5, delta_t=0.1),
#                       init_reinit_sdf_kwargs=kwarg(niter=50, delta_t=0.1))
#
#     # start = time()
#     # im_seg.reinit_sdf(1000)
#     # print(time() - start)
#
#     start = time()
#     im_seg.reinit_sdf1(1000)
#     print(time() - start)
#
#
# def test9():
#     import h5py
#     from skimage.filter import threshold_otsu
#     from imseg.image_tools import imclip, normalize
#
#     f = h5py.File('/Users/yue/data/obj.hdf5', 'r')
#     im = f['object'][:, 350:900, 350:900]
#     f.close()
#
#     # im = anisodiff(im, niter=10)
#
#     im = imclip(im, verbose=True)
#     im = normalize(im, mode='median')
#     thresholds = threshold_otsu(im)
#     print(thresholds)
#     # out_path='/Users/yue/data/segment/sdf_anisodiff_3d/'
#
#     im_seg = ImSeg(im, debug=True, verbose=True)
#     im_seg.initialize(thresholds=thresholds,
#                       diff_sdf_kwargs=kwarg(niter=5),
#                       diff_error_kwargs=kwarg(niter=25),
#                       diff_ave_kwargs=kwarg(Dt=0.5),
#                       reinit_sdf_kwargs=kwarg(niter=5, delta_t=0.1),
#                       init_reinit_sdf_kwargs=kwarg(niter=5, delta_t=0.1),
#                       im_slice=(slice(None), slice(None), 512 - 350),
#                       mode='hard')
#     # im_seg.init_vars(thresholds)
#     # im_seg.init_regions()
#     # im_seg.init_sdf()
#     im_seg.show(im_slice=(slice(None), slice(None), 512 - 350))
#     im_seg.iterate(niter=1)
#     im_seg.show(im_slice=(slice(None), slice(None), 512 - 350))


if __name__ == "__main__":
    test3()