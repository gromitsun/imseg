from __future__ import division, absolute_import, print_function
import numpy as np
try:
    from skimage.filters import threshold_otsu
except ImportError:
    from skimage.filter import threshold_otsu


def imclip(im, low_percentage=0.01, high_percentage=0.99, verbose=False):
    """
    Clip image.
    :param im:
    :param low_percentage:
    :param high_percentage:
    :param verbose:
    :return:
    """
    n = im.size
    low_index = round(low_percentage * n) - 1
    high_index = round(high_percentage * n) - 1
    low_value, high_value = \
        np.partition(im, [low_index, high_index], axis=None)[[low_index, high_index]]
    if verbose:
        print('Clipped %d low pixels with low = %s; and %d pixels with high = %s.'
              % (low_index, low_value, n - high_index, high_value))
    return im.clip(low_value, high_value)


def normalize(im, vmin=0, vmax=1, mode='low-high'):
    """
    Normalize image to [vmin, vmax].
    :param im:
    :param vmin:
    :param vmax:
    :param mode:
    :return:
    """
    if mode == 'low-high':
        im_min = im.min()
        im_max = im.max()
        return ((im - im_min) / (im_max - im_min) + vmin) * (vmax - vmin)
    elif mode == 'median':
        th = threshold_otsu(im)
        median1 = np.median(im[im <= th])
        median2 = np.median(im[im > th])
        return ((im - median1) / (median2 - median1) + vmin) * (vmax - vmin)
    else:
        raise ValueError('Mode %s not understood!' % mode)


def test():
    # Read data
    import h5py

    f = h5py.File('/Users/yue/data/obj.hdf5', 'r')
    im = f['object'][:, :, 512]
    f.close()

    im_th = imclip(im)
    im_norm = normalize(im, mode='median')
    im_th_norm = normalize(im_th, mode='median')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(im, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.figure()
    plt.imshow(im_th, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.figure()
    plt.imshow(im_norm, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.figure()
    plt.imshow(im_th_norm, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test()
