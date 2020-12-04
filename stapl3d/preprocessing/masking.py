#!/usr/bin/env python

"""Generate a mask that covers the tissue.

    # TODO: logging
    # logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)

    # TODO: calculated parameters (thresholds) to yml instead of pickle
    # TODO: report: thresholds to values, not indices.
    # TODO: plot thresholds as contours
    # TODO: plot gradient image as background?
    # TODO: add otsu auto-threshold; as in equalization

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

import numpy as np

from copy import copy

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import distance_transform_edt

from stapl3d import parse_args, Stapl3r, Image, MaskImage, get_paths, get_imageprops
from stapl3d.imarisfiles import find_resolution_level
from stapl3d.reporting import gen_orthoplot_with_colorbar, get_centreslices

logger = logging.getLogger(__name__)


def main(argv):
    """Generate a mask that covers the tissue."""

    steps = ['estimate']
    args = parse_args('mask', steps, *argv)

    masker = Masker(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        dataset=args.dataset,
        suffix=args.suffix,
        n_workers=args.n_workers,
    )

    for step in args.steps:
        masker._fun_selector[step]()


class Masker(Stapl3r):
    """Generate a mask that covers the tissue."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Masker, self).__init__(
            image_in, parameter_file,
            module_id='mask',
            **kwargs,
            )

        self._fun_selector = {
            'estimate': self.estimate,
            }

        default_attr = {
            'step_id': 'mask',
            'inputpath': True,
            'resolution_level': -1,
            'sigma': 48.0,
            'use_median_thresholds': True,
            'median_factor': 3,
            'abs_threshold': 0,
            'thresholds': [],
            'thresholds_slicewise': [],
            'distance_to_edge': True,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        # TODO?
        # self._image_rl = None
        # self._image_mean = None
        # self._image_smooth = None
        # self._image_mask = None
        # self._image_dist = None

        self._parsets = {'estimate': {
            'fpar': self._FPAR_NAMES + ('inputpath', 'resolution_level'),
            'ppar': ('sigma', 'use_median_thresholds', 'median_factor',
                     'abs_threshold', 'thresholds', 'thresholds_slicewise',
                     'distance_to_edge'),
            'spar': ('n_workers',),
        }}

        # TODO: merge with parsets
        self._partable = {
            'resolution_level': 'Image pyramid resolution level',
            'sigma': 'Smoothing sigma',
            'use_median_thresholds': 'Use median threshold method',
            'abs_threshold': 'Absolute threshold',
            }

    def estimate(self, **kwargs):
        """Generate a mask that covers the tissue.

        # resolution_level=-1,
        # sigma=48.0,
        # use_median_thresholds=True,
        # median_factor=3,
        # abs_threshold=0,
        # thresholds=[],
        # distance_to_edge=True,
        # inputpath=True,
        """

        self.set_parameters('estimate', kwargs)
        self._set_inputpath('stitching', 'fuse', '{}.bdv', fallback=self.image_in)
        self.dump_parameters(step=self.step)
        self._generate_dataset_mask()

    def _generate_dataset_mask(self):
        """Generate a mask that covers the tissue."""

        outstem = os.path.join(self.directory, self.format_())
        outputpat = '{}.h5/{}'.format(outstem, '{}')

        if ('.ims' in self.inputpath or '.bdv' in self.inputpath) and self.resolution_level == -1:
            self.resolution_level = find_resolution_level(self.inputpath)
            print('Using resolution level {}'.format(self.resolution_level))

        im_data = extract_resolution_level(self.inputpath, self.resolution_level)

        im_mean, meds = extract_mean(im_data, 'c', True, outputpat)

        im_smooth = extract_smooth(im_mean, self.sigma, True, outputpat)

        self.set_thresholds(self.thresholds, meds, im_data.dims[im_data.axlab.index('z')], im_data.dtype)

        im_mask = postproc_masks(im_smooth, self.thresholds_slicewise, True, outputpat)

        if self.distance_to_edge:
            im_edt = calculate_distance_to_edge(im_mask, outputpat)
            im_edt.close()

        im_data.close()
        im_mean.close()
        im_smooth.close()
        im_mask.close()

        self.dump_parameters(step=self.step)
        self.report()

    def set_thresholds(self, thrs, meds, n_planes, dtype, facs=[0.01, 0.1, 0.2, 0.5, 1.0, 2.0]):

        # TODO: otsu?

        if not thrs:
            thrs = [max(meds) * fac for fac in facs]
        if self.abs_threshold:
            thrs = sorted(list(set(thrs) | set([self.abs_threshold])))

        self.thresholds = [float(t) for t in thrs]

        if self.use_median_thresholds:
            self.thresholds_slicewise = [max(self.abs_threshold, m / self.median_factor) for m in meds]
        else:
            self.thresholds_slicewise = [self.abs_threshold for _ in range(0, n_planes)]

        self.thresholds_slicewise = [float(t) for t in self.thresholds_slicewise]

    def _get_info_dict(self, filestem, info_dict={}, channel=None):

        info_dict['parameters'] = self.load_dumped_pars()

        filepath = '{}.h5/mean'.format(filestem)
        info_dict['props'] = get_imageprops(filepath)
        info_dict['paths'] = get_paths(filepath)
        info_dict['centreslices'] = get_centreslices(info_dict)

        return info_dict

    def _gen_subgrid(self, f, gs, channel=None):
        """3rows-2 columns: 3 image-triplet left, three plots right"""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(3, 2, wspace=0.9)
        # FIXME: this wspace=0.9 is because the axis sharing isnt working properly

        idss = ['mean', 'smooth', 'contours', 'z', 'mask', 'dist2edge']
        axdict.update({k: gen_orthoplot_with_colorbar(f, gs01[i], idx=i)
                       for i, k in enumerate(idss)})

        for ax in axdict['z']: ax.axis('off')
        axdict['z'] = f.add_subplot(gs01[1, 1])
        axdict['z'].tick_params(axis='both', direction='in')
        for l in ['top', 'right']:
            axdict['z'].spines[l].set_visible(False)

        # Axes titles
        titles = {
            'mean': ('Mean over channels', 'lcm', 0),
            'smooth': ('Smoothed mean', 'rcm', 2),
            'contours': ('Thresholds', 'lcm', 0),
            'z': ('Z thresholds', 'rc', 0),
            'mask': ('Mask', 'lcm', 0),
            'dist2edge': ('Distance to mask', 'rcm', 2),
            }
        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict={}):
        """Show images in report."""

        # def plot_colorbar(ax, im, cax=None, loc='right'):
        #     cbar = f.colorbar(im, cax=cax, extend='both', shrink=0.9, ax=ax)
        #     cbar.ax.tick_params(labelsize=7)
        #     cax.yaxis.set_ticks_position(loc)
        #     return cbar

        # def get_extent(img, elsize, x_idx=2, y_idx=1):
        #     w = elsize[x_idx] * img.shape[1]
        #     h = elsize[y_idx] * img.shape[0]
        #     extent = [0, w, 0, h]
        #     return extent

        # def get_clim(cslc):
        #     c_min = np.amin([np.quantile(cslc[d], 0.05) for d in 'xyz'])
        #     c_max = np.amax([np.quantile(cslc[d], 0.95) for d in 'xyz'])
        #     c_min = self._power_rounder(c_min)
        #     c_max = self._power_rounder(c_max)
        #     # TODO: c_min should never be c_max
        #     return [c_min, c_max]

        cslcs = info_dict['centreslices']
        clim_mean = self._get_clim(cslcs['mean'])
        if 'dist2edge' in cslcs.keys():
            clim_d2e = self._get_clim(cslcs['dist2edge'])
        # print([clim_mean, clim_d2e])

        for k, v in cslcs.items():
            cslcs[k]['x'] = cslcs[k]['x'].transpose()

        palette = copy(plt.cm.gray)
        palette.set_bad('g', 1.0)
        vol_dict = {
            'mean':      ('right', 'gray',  clim_mean),
            'smooth':    ('left',  'jet',   clim_mean),
            'contours':  ('right', 'gray',  clim_mean),
            'mask':      ('right', palette, clim_mean),
            }
        if 'dist2edge' in cslcs.keys():
            vol_dict['dist2edge'] = ('left',  'rainbow', clim_d2e)

        for k, (loc, cmap, clim) in vol_dict.items():

            for d, aspect, ax_idx in zip('xyz', ['auto', 'auto', 'equal'], [2, 1, 0]):

                ax = axdict[k][ax_idx]

                extent = self._get_extent(cslcs['mean'][d], info_dict['props']['elsize'])

                if k == 'contours':
                    thrs = info_dict['parameters']['thresholds']
                    im = ax.imshow(cslcs['mean'][d], cmap=cmap, aspect=aspect)
                    cs = ax.contour(cslcs['smooth'][d], thrs, cmap=plt.cm.rainbow)
                    if d == 'z':  # add labels to xy-slice
                        ax.clabel(cs, inline=1, fontsize=7)
                        labels = ['thr={}'.format(thr) for thr in thrs]
                        for i in range(len(labels)):
                            cs.collections[i].set_label(labels[i])
                elif k == 'mask':
                    # clipped = self._clipped_colormap(clip_threshold, clip_color, n_colors)
                    im = ax.imshow(
                        np.ma.masked_where(~cslcs['mask'][d].astype('bool'), cslcs['mean'][d]),
                        interpolation='bilinear', cmap=cmap,
                        norm=colors.Normalize(vmin=-1.0, vmax=1.0),
                        aspect=aspect, extent=extent,
                        )
                else:
                    im = ax.imshow(cslcs[k][d], cmap=cmap, extent=extent, aspect=aspect)

                im.set_clim(clim[0], clim[1])
                ax.axis('off')

                if ax_idx == 0:
                    self._plot_colorbar(f, ax, im, cax=axdict[k][3], loc=loc)
                    if k == 'mean':
                        self._add_scalebar(ax, extent[1])

    def _plot_profiles(self, f, axdict, info_dict={}):
        """Show profiles in report."""

        ax = axdict['z']
        ax.plot(info_dict['parameters']['thresholds_slicewise'],
                color='k', linewidth=1, linestyle='-')
        ax.set_xlabel('section index', fontsize=7, loc='right')


def extract_resolution_level(image_in, resolution_level=-1):
    """Extract data from a 5D Imaris image pyramid and create 4D image."""

    # TODO: handle this in Image class when providing reslev as __init__ argument
    if resolution_level != -1:
        # FIXME: this is only ims, TODO bdv etc
        image_in = '{}/DataSet/ResolutionLevel {}'.format(image_in, resolution_level)

    im = Image(image_in, permission='r', reslev=resolution_level)
    im.load(load_data=False)

    return im


def extract_mean(im, dim='c', keep_dtype=True, output=''):
    """Calculate mean over channels."""

    ods = 'mean'

    im.load(load_data=False)
    props = im.get_props()
    if len(im.dims) > 4:
        props = im.squeeze_props(props, dim=4)
    if len(im.dims) > 3:
        props = im.squeeze_props(props, dim=3)

    mo = Image(output.format(ods), **props)
    mo.create()

    zdim = im.axlab.index('z')
    if im.chunks is not None:
        nslcs = im.chunks[zdim]
    else:
        nslcs = im.dims[zdim]
    slc_thrs = []
    for zstart in range(0, im.dims[zdim], nslcs):
        zstop = min(im.dims[zdim], zstart + nslcs)
        im.slices[zdim] = mo.slices[zdim] = slice(zstart, zstop, None)
        data = im.slice_dataset()
        if im.slices[zdim].stop - im.slices[zdim].start == 1:
            data = np.expand_dims(data, 0)
        data_mean = np.mean(data, axis=im.axlab.index(dim))
        if keep_dtype:
            data_mean = data_mean.astype(im.dtype)
        mo.write(data_mean)
        slc_thrs += list(np.median(np.reshape(data_mean, [data_mean.shape[0], -1]), axis=1))

    mo.slices = None
    mo.set_slices()

    im.close()

    return mo, slc_thrs


def extract_smooth(im, sigma=48.0, keep_dtype=True, output=''):
    """Smooth the image in-plane."""

    def smooth(data, sigma, elsize):
        """Smooth data with Gaussian kernel."""

        if len(sigma) == 1:
            sigma = sigma * len(elsize)
        elif len(sigma) != len(elsize):
            raise Exception('sigma does not match dimensions')
        sigma = [sig / es for sig, es in zip(sigma, elsize)]

        data_smoothed = gaussian_filter(data, sigma)

        return data_smoothed

    ods = 'smooth'

    if not isinstance(sigma, list):
        sigma = [sigma] * 3
        sigma[im.axlab.index('z')] = 0.0

    im.load(load_data=False)
    data_smoothed = smooth(im.slice_dataset(), sigma, im.elsize)
    if keep_dtype:
        data_smoothed = data_smoothed.astype(im.dtype)
    im.close()

    props = im.get_props()
    mo = write_data(data_smoothed, props, output, ods)

    return mo


def postproc_masks(im, thrs=[], fill_holes=True, output=''):
    """Apply slicewise thresholds to data and fill holes.

    NOTE: zyx assumed
    """

    ods = 'mask'.format(0)

    im.load(load_data=False)
    if thrs:
        data = im.slice_dataset()
        mask = np.zeros(im.dims[:3], dtype='bool')
        for slc in range(0, mask.shape[0]):
            mask[slc, :, :] = data[slc, :, :] > thrs[slc]
    else:
        mask = im.slice_dataset()
    im.close()

    if fill_holes:
        for slc in range(0, mask.shape[0]):
            mask[slc, :, :] = binary_fill_holes(mask[slc, :, :])

    props = im.get_props()
    mo = write_data(mask, props, output, ods)

    return mo


def calculate_distance_to_edge(im, output=''):
    """"Calculate the euclidian distance transform of the mask."""

    ods = 'dist2edge'

    im.load(load_data=False)
    elsize = np.absolute(im.elsize)
    dt = np.zeros(im.ds.shape, dtype='float')
    for i, slc in enumerate(im.ds[:]):
        dt[i, :, :] = distance_transform_edt(slc, sampling=elsize[1:])

    props = im.get_props()
    mo = write_data(dt, props, output, ods)

    return mo


def write_data(data, props, out_template='', ods=''):
    """Create an Image object and optionally write to file."""

    outputpath = ''
    if out_template:
        outputpath = out_template.format(ods)

    props['dtype'] = data.dtype
    mo = Image(outputpath, **props)
    mo.create()
    mo.write(data)
    # mo.close()

    return mo


if __name__ == "__main__":
    main(sys.argv[1:])
