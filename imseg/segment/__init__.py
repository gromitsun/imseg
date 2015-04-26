from __future__ import division, print_function, absolute_import
import numpy as np
# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import os
import im3D.sdf.inplace
import im3D.smoothing.inplace
from imseg.regions import init_regions, update_regions
from imseg.update_sdf import update_sdf


def arg(*args):
    return args


def kwarg(**kwargs):
    return kwargs


def arg_parse(*args, **kwargs):
    return args, kwargs


def require_array(a, dtype=None):
    if a is not None:
        return np.array(a, dtype, copy=False, ndmin=1)


class ImSeg(object):
    def __init__(self, im, init=False, **kwargs):
        self.im = np.asarray(im)
        self.dtype = self.im.dtype
        self.kwargs = kwargs
        if init:
            self.init()

    def init(self, **kwargs):
        """
        Initialize the ImSeg object (setting up parameter and settings).
        :param kwargs:
        :return:
        """
        if kwargs:
            self.kwargs = kwargs

        self.init_settings()
        self.init_parameters()
        self.write_log()

    def init_parameters(self):
        diff_func = self.kwargs.get('diff_func', 'isotropic')
        if self.im.ndim in [2, 3]:
            if diff_func == 'isotropic':
                self.diff_func = im3D.smoothing.inplace.ds
            elif diff_func == 'anisotropic':
                self.diff_func = im3D.smoothing.anisodiff
            else:
                raise KeyError('diff_func type {} not understood!'.format(diff_func))
        else:
            print('ndim = %d' % self.im.ndim)
            self.diff_func = im3D.smoothing.ds

        self.thresholds = require_array(self.kwargs.get('thresholds', None), dtype=self.dtype)
        self.diff_error_args = self.kwargs.get('diff_error_args', ())
        self.diff_error_kwargs = self.kwargs.get('diff_error_kwargs', kwarg(niter=50))
        self.diff_ave_args = self.kwargs.get('diff_ave_args', ())
        self.diff_ave_kwargs = self.kwargs.get('diff_ave_kwargs', kwarg(Dt=0))
        self.diff_sdf_args = self.kwargs.get('diff_sdf_args', ())
        self.diff_sdf_kwargs = self.kwargs.get('diff_sdf_kwargs', kwarg(niter=5))
        self.reinit_sdf_args = self.kwargs.get('reinit_sdf_args', ())
        self.reinit_sdf_kwargs = self.kwargs.get('reinit_sdf_kwargs', kwarg(niter=10, delta_t=0.05))
        self.init_reinit_sdf_args = self.kwargs.get('init_reinit_sdf_args', ())
        self.init_reinit_sdf_kwargs = self.kwargs.get('init_reinit_sdf_kwargs', kwarg(niter=20, delta_t=0.1))
        self.beta = self.kwargs.get('beta', 0.5)

        print(self.kwargs)

    def init_settings(self):
        self.debug = self.kwargs.pop('debug', False)
        self.verbose = self.kwargs.pop('verbose', False)
        self.out_path = self.kwargs.pop('out_path', None)
        print(self.out_path)
        if (self.out_path is not None) and (not os.path.exists(self.out_path)):
            os.makedirs(self.out_path)
        self.im_slice = self.kwargs.pop('im_slice', None)

    def init_vars(self, thresholds=None):
        self.regions = []
        if thresholds is not None:
            self.thresholds = require_array(thresholds, dtype=self.dtype)
        self.num_regions = np.ndim(self.thresholds) + 1
        self.sdf = np.empty((self.num_regions - 1, ) + self.im.shape, dtype=self.dtype)
        self.im_ave = np.empty_like(self.im, dtype=self.dtype)
        self.im_error = np.empty_like(self.im, dtype=self.dtype)
        self.iter_count = 0

    def init_regions(self):
        init_regions(self.im, self.thresholds, self.im_ave, self.im_error, self.sdf)

    def initialize(self, thresholds=None):
        """
        Initialize the segmentation.
        :param thresholds:
        :return:
        """
        self.init_vars(thresholds)
        print(self.thresholds)
        self.init_regions()
        # self.init_regions_noerr()
        # self.diffuse_ave()
        # self.calc_error()
        self.reinit_sdf()

    def update_regions(self):
        update_regions(self.im, self.sdf, self.im_ave, self.im_error)

    def reinit_sdf(self):
        if self.verbose:
            print('Reinitializing SDF ...')
        for i in xrange(self.num_regions - 1):
            im3D.sdf.inplace.reinit(self.sdf[i], self.sdf[i], **self.reinit_sdf_kwargs)

    def reinitialize(self):
        if self.verbose:
            print('Begin reinitialization ...')
        self.reinit_sdf()
        self.update_regions()
        # self.update_regions_noerr()
        # self.diffuse_ave()
        # self.calc_error()

    def update_sdf(self):
        if self.verbose:
            print('Updating SDF using im_error ...')
        update_sdf(self.sdf, self.im_error, self.beta)

    def assertion(self):
        assert self.im_error.shape == self.im.shape
        assert self.im_ave.shape == self.im.shape
        assert self.sdf.shape == (self.num_regions - 1,) + self.im.shape

    def diffuse_error(self):
        self.diff_func(self.im_error, self.im_error, **self.diff_error_kwargs)

    def diffuse_ave(self):
        im3D.smoothing.inplace.ds(self.im_ave, self.im_ave, **self.diff_ave_kwargs)

    def diffuse_sdf(self):
        for i in xrange(self.num_regions - 1):
            self.diff_func(self.sdf[i], self.sdf[i], **self.diff_sdf_kwargs)

    def iterate(self, niter):
        for i in xrange(niter):
            print('Iteration %d' % (self.iter_count + 1))
            if self.verbose:
                print('Smoothing im_error by diffusion ...')
            self.diffuse_error()
            self.update_sdf()
            if self.verbose:
                print('Smoothing SDF by diffusion ...')
            self.diffuse_sdf()
            self.reinitialize()
            self.iter_count += 1
            if self.debug and ((self.iter_count % self.debug == 0) or (self.iter_count == 1)):
                self.show(save_path=self.out_path)

    def show_contour(self, im_slice=Ellipsis, save_path=None, show=True):
        for i in xrange(self.num_regions - 1):
            plt.figure('contour-%d_iter_%d' % (i+1, self.iter_count))
            plt.title('contour-%d, iteration %d' % (i+1, self.iter_count))
            plt.imshow(self.im[im_slice], interpolation='nearest', cmap='gray')
            plt.contour(self.sdf[i][im_slice], levels=[0], colors='r')
            if save_path is not None:
                plt.savefig(save_path + 'contour-%d_iter_%d' % (i+1, self.iter_count))
        if show is True:
            if save_path is None:
                plt.show()

    def show(self, im_slice=None, save_path=None):
        if self.verbose:
            print('Plotting data ...')
        ###
        if im_slice is None:
            if self.im_slice is not None:
                im_slice = self.im_slice
            else:
                im_slice = Ellipsis
        ###
        if save_path == -1:
            save_path = self.out_path
        plt.figure('im_ave')
        plt.title('im_ave, iteration %d' % self.iter_count)
        plt.imshow(self.im_ave[im_slice])
        plt.colorbar()
        if save_path is not None:
            plt.savefig(save_path + 'im_ave_iter_%d' % self.iter_count)
        plt.figure('im_error')
        plt.title('im_error, iteration %d' % self.iter_count)
        plt.imshow(self.im_error[im_slice])
        plt.colorbar()
        if save_path is not None:
            plt.savefig(save_path + 'im_error_iter_%d' % self.iter_count)
        for i in xrange(self.num_regions - 1):
            plt.figure('SDF-%d' % (i + 1))
            plt.title('SDF-%d, iteration %d' % (i + 1, self.iter_count))
            plt.imshow(self.sdf[i][im_slice])
            plt.colorbar()
            if save_path is not None:
                plt.savefig(save_path + 'SDF-%d_iter_%d' % (i + 1, self.iter_count))
        self.show_contour(im_slice=im_slice, save_path=save_path, show=False)
        if save_path is None:
            plt.show()
        else:
            plt.close('all')

    def write_log(self):
        if self.out_path is not None:
            if self.verbose:
                print('Writing log file ...')
            f = open(self.out_path + '/log.txt', 'w')
            f.write('thresholds: %s\n' % self.thresholds)
            f.writelines(_dump_dict(self.kwargs))
            f.close()

    def save(self, filename=None, save_path=None):
        if save_path is None:
            save_path = self.out_path
        elif not os.path.exists(save_path):
            os.makedirs(save_path)
        if filename is None:
            filename = 'imseg_iter_%d.npz' % self.iter_count
        np.savez(save_path + '/' + filename,
                 thresholds=self.thresholds,
                 sdf=self.sdf,
                 iter_count=self.iter_count)

    def load(self, path2file):
        a = np.load(path2file)
        self.thresholds = a['thresholds']
        self.init_vars(self.thresholds)
        self.iter_count = a['iter_count']
        self.sdf = a['sdf']
        self.update_regions()
        # self.update_regions_noerr()
        # self.diffuse_ave()
        # self.calc_error()


def _dump_dict(d, indent=0):
    out = ''
    for key, value in d.iteritems():
        out += '\t' * indent + str(key) + '\n'
        if isinstance(value, dict):
            out += _dump_dict(value, indent + 1)
        else:
            out += '\t' * (indent + 1) + str(value) + '\n'
    return out


del absolute_import
del print_function
del division



