#!/usr/bin/env python

"""Generate a mask that covers the tissue.

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

    steps = ['extract_mean', 'extract_smooth', 'extract_mask', 'postprocess']
    args = parse_args('mask', steps, *argv)

    mask3r = Mask3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        mask3r._fun_selector[step]()


class Mask3r(Stapl3r):
    """Generate a mask that covers the tissue."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Mask3r, self).__init__(
            image_in, parameter_file,
            module_id='mask',
            **kwargs,
            )

        self._fun_selector = {
            'extract_mean': self.extract_mean,
            'extract_smooth': self.extract_smooth,
            'extract_mask': self.extract_mask,
            'postprocess': self.postprocess,
            }

        self._parallelization = {
            'extract_mean': [],
            'extract_smooth': [],
            'extract_mask': [],
            'postprocess': [],
            }

        self._parameter_sets = {
            'extract_mean': {
                'fpar': self._FPAR_NAMES + ('resolution_level',),
                'ppar': (),
                'spar': ('_n_workers',),
                },
            'extract_smooth': {
                'fpar': self._FPAR_NAMES + ('resolution_level',),
                'ppar': ('sigma',),
                'spar': ('_n_workers',),
                },
            'extract_mask': {
                'fpar': self._FPAR_NAMES + ('resolution_level',),
                'ppar': ('use_median_thresholds', 'median_factor',
                         'abs_threshold', 'thresholds', 'thresholds_slicewise'),
                'spar': ('_n_workers',),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES + ('resolution_level',),
                'ppar': ('distance_to_edge', 'thresholds_slicewise'),
                'spar': ('_n_workers',),
                },
            }

        self._parameter_table = {
            'resolution_level': 'Image pyramid resolution level',
            'sigma': 'Smoothing sigma',
            'use_median_thresholds': 'Use median threshold method',
            'median_factor': 'Median multiplication factor',
            'abs_threshold': 'Absolute threshold / minimum threshold',
            }

        default_attr = {
            'resolution_level': -1,
            'channels': [],
            'keep_dtype': True,
            'sigma': 48.0,
            'use_median_thresholds': True,
            'median_factor': 3,
            'abs_threshold': 0,
            'fill_holes': True,
            'distance_to_edge': False,
            'thresholds': [],
            'thresholds_slicewise': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._images = ['mean', 'smooth']
        self._labels = ['mask']

    def _init_paths(self):

        # FIXME: moduledir (=step_id?) can vary
        prev_path = {
            'moduledir': 'stitching', 'module_id': 'stitching',
            'step_id': 'stitching', 'step': 'postprocess',
            'ioitem': 'outputs', 'output': 'aggregate',
            }
        datapath = self._get_inpath(prev_path)
        if datapath == 'default':
            datapath = self._build_path(
                moduledir=prev_path['moduledir'],
                prefixes=[self.prefix, prev_path['module_id']],
                ext='bdv',
                )

        stem = self._build_basename()
        self._paths = {
            'extract_mean': {
                'inputs': {
                    'data': datapath,
                    },
                'outputs': {
                    'file': f'{stem}.h5',
                    'mean': f'{stem}.h5/mean',
                    'meds': f'{stem}_medians.npy',
                    },
                },
            'extract_smooth': {
                'inputs': {
                    'mean': f'{stem}.h5/mean',
                    },
                'outputs': {
                    'file': f'{stem}.h5',
                    'smooth': f'{stem}.h5/smooth',
                    },
                },
            'extract_mask': {
                'inputs': {
                    'data': datapath,
                    'meds': f'{stem}_medians.npy',
                    'smooth': f'{stem}.h5/smooth',
                    },
                'outputs': {
                    'file': f'{stem}.h5',
                    'mask': f'{stem}.h5/mask',
                    },
                },
            'postprocess': {
                'inputs': {
                    'data': datapath,
                    'meds': f'{stem}_medians.npy',
                    'mask': f'{stem}.h5/mask',
                    },
                'outputs': {
                    'file': f'{stem}.h5',
                    'dist2edge': f'{stem}.h5/dist2edge',
                    'report': os.path.join(self._logdir, f'{stem}.pdf'),
                    },
                },
            }
        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def extract_mean(self, **kwargs):
        """Calculate mean over channels and slicewise medians."""

        self._set_resolution_level(self.inputpaths['extract_mean']['data'])

        arglist = self._prep_step('extract_mean', kwargs)
        self._extract_mean()
        self.dump_parameters(step='extract_mean')

    def _extract_mean(self):
        """Calculate mean over channels and slicewise medians."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        meds = calculate_mean(inputs['data'],
                              self.resolution_level,
                              self.keep_dtype,
                              self.channels,
                              outputpath=outputs['mean'])

        np.save(outputs['meds'], np.array(meds))

    def extract_smooth(self, **kwargs):
        """Smooth image."""

        self._set_resolution_level(self.inputpaths['extract_mean']['data'])

        arglist = self._prep_step('extract_smooth', kwargs)
        self._extract_smooth()
        self.dump_parameters(step='extract_smooth')

    def _extract_smooth(self):
        """Smooth image."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        smooth_image(inputs['mean'], self.sigma, self.keep_dtype, outputs['smooth'])

    def extract_mask(self, **kwargs):
        """Generate a mask that covers the tissue."""

        self._set_resolution_level(self.inputpaths['extract_mean']['data'])

        arglist = self._prep_step('extract_mask', kwargs)
        self._extract_mask()
        self.dump_parameters(step='extract_mask')

    def _extract_mask(self):
        """Generate a mask that covers the tissue."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        self.set_thresholds(self.thresholds, inputs)
        postproc_masks(inputs['smooth'], self.thresholds_slicewise, self.fill_holes, outputs['mask'])

    def postprocess(self, **kwargs):
        """Calculate distance to edge and generate report."""

        self._set_resolution_level(self.inputpaths['extract_mean']['data'])

        arglist = self._prep_step('postprocess', kwargs)
        self._postprocess()
        self.dump_parameters(step='postprocess')

    def _postprocess(self):
        """Calculate distance to edge and generate report."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        if not self.thresholds_slicewise:
            self.set_thresholds(self.thresholds, inputs)

        if self.distance_to_edge:
            im_edt = calculate_distance_to_edge(inputs['mask'], outputs['dist2edge'])
            im_edt.close()

        self.report(outputpath=outputs['report'], inputs=inputs, outputs=outputs)

    def _set_resolution_level(self, datapath):

        if ('.ims' in datapath or '.bdv' in datapath):
            if self.resolution_level == -1:
                self.resolution_level = find_resolution_level(datapath)

    def set_thresholds(self, thrs, inputs, facs=[0.01, 0.1, 0.2, 0.5, 1.0, 2.0]):
        """Set thresholds for each dataset slice."""
        # TODO: otsu? (as in equalization module)

        meds = np.load(inputs['meds'])
        props = get_image_props(inputs['data'], self.resolution_level)
        n_planes = props['shape'][props['axlab'].index('z')]

        if not thrs:
            thrs = [max(meds) * fac for fac in facs]
        if self.abs_threshold:
            thrs = sorted(list(set(thrs) | set([self.abs_threshold])))

        self.thresholds = [float(t) for t in thrs]

        if self.use_median_thresholds:
            thrs = [max(self.abs_threshold, m / self.median_factor) for m in meds]
        else:
            thrs = [self.abs_threshold for _ in range(0, n_planes)]

        self.thresholds_slicewise = [float(t) for t in thrs]

    def _get_info_dict(self, **kwargs):

        # self._prep_step('estimate', kwargs)  # TODO: independent of estimate
        self.step = 'postprocess'
        self._set_paths_step()
        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        kwargs['parameters']['thresholds'] = self.thresholds
        kwargs['parameters']['thresholds_slicewise'] = self.thresholds_slicewise

        if 'outputs' in kwargs.keys():
            filepath = self._abs(kwargs['inputs']['mask'])
        else:
            filepath = self._abs(kwargs['parameters']['inputs']['mask'])

        kwargs['props'] = get_imageprops(filepath)
        kwargs['paths'] = get_paths(filepath)
        kwargs['centreslices'] = get_centreslices(kwargs)

        return kwargs

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
                    # cs = ax.contour(cslcs['mean'][d], thrs, cmap=plt.cm.rainbow)
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

    def view(self, input=[], images=[], labels=[], settings={}):
        # TODO: specifying eg labels as empty will reset the labels to default: undesired
        # solve eg with None

        images = images or self._images
        labels = labels or self._labels

        if not input:
            input = self._abs(self.outputpaths['extract_mean']['file'])

        super().view(input, images, labels, settings)


def calculate_mean(image_in, resolution_level=-1, keep_dtype=True, channels=[], outputpath=''):
    """Calculate mean over channels."""

    def slice_dataset(im, cdim, channels):
        # TODO: implement dirty slicing in Image class
        d = []
        for ch in channels:
            im.slices[cdim] = slice(ch, ch + 1)
            d.append(im.slice_dataset(squeeze=False))
        return np.squeeze(np.stack(d, axis=cdim))

    # Prepare output.
    props = get_image_props(image_in, resolution_level, squeeze_dims=True)
    if not keep_dtype:
        props['dtype'] = 'float'
    mo = Image(outputpath, **props)
    mo.create()

    im = Image(image_in, reslev=resolution_level, permission='r')
    im.load(load_data=False)
    zdim = im.axlab.index('z')
    cdim = im.axlab.index('c')

    # Read in chunks or full image.
    if im.chunks is not None:
        nslcs = im.chunks[zdim]
    else:
        nslcs = im.dims[zdim]

    slc_thrs = []
    for zstart in range(0, im.dims[zdim], nslcs):
        zstop = min(im.dims[zdim], zstart + nslcs)
        im.slices[zdim] = mo.slices[zdim] = slice(zstart, zstop, None)

        if channels:
            data = slice_dataset(im, cdim, channels)  # for dirty slices
        else:
            channels = list(range(im.dims[cdim]))
            data = im.slice_dataset()

        if im.slices[zdim].stop - im.slices[zdim].start == 1:
            data = np.expand_dims(data, 0)

        if len(channels) == 1:
            data = np.expand_dims(data, cdim)

        data_mean = np.mean(data, axis=cdim)

        if keep_dtype:
            data_mean = data_mean.astype(im.dtype)

        mo.write(data_mean)

        slc_thrs += list(np.median(np.reshape(data_mean, [data_mean.shape[0], -1]), axis=1))

    mo.close()
    im.close()

    return slc_thrs


def smooth_image(image_in, sigma=48.0, keep_dtype=True, outputpath=''):
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

    im = Image(image_in, permission='r+')
    im.load(load_data=False)

    if not isinstance(sigma, list):
        sigma = [sigma] * 3
        sigma[im.axlab.index('z')] = 0.0

    data_smoothed = smooth(im.slice_dataset(), sigma, im.elsize)

    props = im.get_props()
    im.close()

    if keep_dtype:
        data_smoothed = data_smoothed.astype(im.dtype)

    mo = write_data(data_smoothed, props, outputpath)

    return mo


def postproc_masks(image_in, thrs=[], fill_holes=True, outputpath=''):
    """Apply slicewise thresholds to data and fill holes."""
    # FIXME: assumes zyx

    im = Image(image_in, permission='r+')
    im.load(load_data=False)
    props = im.get_props()
    if thrs:
        data = im.slice_dataset()
        dims = [d for d, al in zip(im.dims, im.axlab) if al in 'zyx']
        mask = np.zeros(dims, dtype='bool')
        for slc in range(0, mask.shape[0]):

            mask[slc, :, :] = data[slc, :, :] > thrs[slc]
    else:
        mask = im.slice_dataset()
    im.close()

    if fill_holes:
        for slc in range(0, mask.shape[0]):
            mask[slc, :, :] = binary_fill_holes(mask[slc, :, :])

    mo = write_data(mask, props, outputpath)

    return mo


def calculate_distance_to_edge(image_in, outputpath=''):
    """"Calculate the euclidian distance transform of the mask."""
    # FIXME: assumes zyx

    im = Image(image_in, permission='r+')
    im.load(load_data=False)
    elsize = np.absolute(im.elsize)
    dt = np.zeros(im.ds.shape, dtype='float')

    for i, slc in enumerate(im.ds[:]):
        dt[i, :, :] = distance_transform_edt(slc, sampling=elsize[1:])

    props = im.get_props()
    im.close()

    mo = write_data(dt, props, outputpath)

    return mo


def write_data(data, props, outputpath=''):
    """Create an Image object and optionally write to file."""

    props['dtype'] = data.dtype
    mo = Image(outputpath, **props)
    mo.create()
    mo.write(data)
    mo.close()

    return mo


def get_image_props(image_in, resolution_level, squeeze_dims=False):

    im = Image(image_in, reslev=resolution_level, permission='r')
    im.load(load_data=False)
    props = im.get_props()

    if squeeze_dims:
        if len(im.dims) > 4:
            props = im.squeeze_props(props, dim=4)
        if len(im.dims) > 3:
            props = im.squeeze_props(props, dim=3)

    im.close()

    return props


if __name__ == "__main__":
    main(sys.argv[1:])
