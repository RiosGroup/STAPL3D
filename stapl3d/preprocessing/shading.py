#!/usr/bin/env python

"""Correct z-stack shading.

    # TODO: generalize to other formats than czi
    # TODO: automate threshold estimate
    # TODO: implement more metrics
    # TODO: implement logging throughout.
    # TODO: write corrected output directly to bdv for Fiji BigStitcher.
    # TODO: try getting median over 4D concatenation directly (forego xy)
    # TODO: make file deletion optional?
    # TODO: corrected vs non-corrected images in report
    # TODO: clipping mask is not automatically included in the tileoffsets for the stitching procedure
    # TODO: tif input option
    # TODO: clean <>_X.npz and <>_Y.npz files
    # TODO: change step_id to module_id throughout?

    # TODO: maybe dump_pars from within functions to be useful for HCP calls too

    # TODO: merge pdfs
    # TODO: generalize read-write in 'apply'

    # TODO: logging system
    # logstem = os.path.join(outputdir, '{}_{}'.format(dataset, step_id))
    # logging.basicConfig(filename='{}.log'.format(logstem), level=logging.INFO)
    # msg = 'Processing ch{:02d}:plane{:03d}'.format(channel, plane)
    # logger.info(msg)  # print(msg)

    # # Save derived parameters.
    # dpars = {'z_sel': [v.item() for v in z_sel]}
    # with open('{}.yml'.format(outstem), 'w') as f:
    #     yaml.dump(dpars, f, default_flow_style=False)

        # attrs = vars(self)
        # print(', '.join("%s: %s" % item for item in attrs.items()))

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import yaml

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from tifffile import create_output
from xml.etree import cElementTree as etree

from skimage import img_as_float
from skimage.io import imread, imsave

import czifile
from readlif.reader import LifFile

from stapl3d import parse_args, Stapl3r, Image, format_

logger = logging.getLogger(__name__)


def main(argv):
    """Correct z-stack shading."""

    steps = ['estimate', 'postprocess', 'apply']
    args = parse_args('shading', steps, *argv)

    deshader = Deshader(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        dataset=args.dataset,
        suffix=args.suffix,
        n_workers=args.n_workers,
    )

    for step in args.steps:
        deshader._fun_selector[step]()


class Deshader(Stapl3r):
    """Correct z-stack shading."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Deshader, self).__init__(
            image_in, parameter_file,
            module_id='shading',
            **kwargs,
            )

        self._fun_selector = {
            'estimate': self.estimate,
            'postprocess': self.postprocess,
            'apply': self.apply,
            }

        default_attr = {
            'step_id': 'shading',
            'channels': [],
            'planes': [],
            'noise_threshold': 0,
            'metric': 'median',
            'quantile_threshold': 0.8,
            'polynomial_order': 3,
            'stacks': [],
            'clipping_mask': False,
            'correct': True,
            'shadingpat': '',
            'write_to_tif': True,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._parsets = {
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('noise_threshold', 'metric'),
                'spar': ('n_workers', 'channels', 'planes'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('quantile_threshold', 'polynomial_order'),
                'spar': ('n_workers', 'channels'),
                },
            'apply': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('clipping_mask', 'correct', 'shadingpat', 'write_to_tif'),
                'spar': ('n_workers', 'stacks'),
                },
            }

        # TODO: merge with parsets
        self._partable = {
            'noise_threshold': 'Noise threshold',
            'metric': 'Metric',
            'quantile_threshold': 'Quantile threshold',
            'polynomial_order': 'Polynomial order',
            }

    def estimate(self, **kwargs):
        """Estimate z-stack shading.

        channels=[],
        planes=[],
        noise_threshold=0,
        metric='median',
        """

        self.set_parameters('estimate', kwargs)
        arglist = self._get_arglist(['channels', 'planes'])
        self.set_n_workers(len(arglist))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._estimate_plane, arglist)

    def _estimate_plane(self, channel, plane):
        """Estimate the x- and y-profiles for a channel in a czi/lif file."""

        postfix_ch = self._suffix_formats['c'].format(channel)
        postfix_st = self._suffix_formats['z'].format(plane)
        basename = self.format_([self.dataset, self.suffix, postfix_ch, postfix_st])
        outstem = os.path.join(self.directory, basename)

        # Compute median values per plane for X and Y concatenation.
        dstack = read_tiled_plane(self.image_in, channel, plane)
        out = {}
        for axis in [0, 1]:
            dstacked = np.concatenate(dstack, axis=0)
            ma_data = np.ma.masked_array(dstacked, dstacked < self.noise_threshold)
            if self.metric == 'median':
                out[axis] = np.ma.median(ma_data, axis=0)
            else:
                out[axis] = np.ma.mean(ma_data, axis=0)

        np.savez('{}.npz'.format(outstem), X=out[1], Y=out[0])

    def postprocess(self, **kwargs):
        """Fit shading profiles for all channels.

        channels=[],
        quantile_threshold=0.8,
        polynomial_order=3,
        """

        self.set_parameters('postprocess', kwargs)
        arglist = self._get_arglist(['channels'])
        self.set_n_workers(len(arglist))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._postprocess_channel, arglist)

    def _postprocess_channel(self, channel):
        """Fit shading profiles for a channel."""

        postfix_ch = self._suffix_formats['c'].format(channel)
        basename = self.format_([self.dataset, self.suffix, postfix_ch])
        outstem = os.path.join(self.directory, basename)

        iminfo = get_image_info(self.image_in)
        img_fitted = np.ones(iminfo['tilesize'], dtype='float')

        # Process the medians for the channel.
        suffix_pat = self._suffix_formats['z'].format(0).replace('0', '?')
        plane_pat = self.format_([basename, suffix_pat])
        npzs = glob(os.path.join(self.directory, '{}.npz'.format(plane_pat)))
        npzs.sort()

        def process_medians(meds, q_thr, order, outstem, dim):
            """Select, normalize and fit profiles; and save for report."""

            sel, dm, z_sel = select_z_range(meds, q_thr)
            m, n, nm = normalize_profile(sel)
            f, fn = fit_profile(nm, order)
            np.savez(
                '{}_{}.npz'.format(outstem, dim),
                sel=sel, dm=dm, z_sel=z_sel,
                mean=m, norm=n, norm_mean=nm,
                fitp=f, fitp_norm=fn,
                )

            return fn

        for dim in 'XY':

            meds = []
            for npzfile in npzs:
                medians = np.load(npzfile)
                meds.append(medians[dim])
                medians.close()

            fitp_norm = process_medians(
                np.stack(meds, axis=0),
                self.quantile_threshold, self.polynomial_order,
                outstem, dim,
                )

            if dim == 'X':
                img_fitted *= fitp_norm
            elif dim == 'Y':
                img_fitted *= fitp_norm[:, np.newaxis]

        # Save estimated shading image.  # FIXME: handle floats here
        img = np.array(img_fitted * np.iinfo(iminfo['dtype']).max, dtype=iminfo['dtype'])
        imsave('{}.tif'.format(outstem), img)

        # Print a report page to pdf.
        self.report(channel=channel)

        # Delete planes
        for npzfile in npzs:
            os.remove(npzfile)

    def apply(self, **kwargs):
        """Apply zstack shading correction for all channels.

        stacks=[],
        clipping_mask=False,
        correct=True,
        shadingpat='',
        write_to_tif=True,
        """

        self.set_parameters('apply', kwargs)
        arglist = self._get_arglist(['stacks'])
        self.set_n_workers(len(arglist))

        # Set derived parameter defaults.  # TODO: move to method: derive_defaults
        if not self.shadingpat:
            stem = self.format_([self.dataset, self.suffix, self._suffix_formats['c']])
            self.shadingpat = os.path.join(self.directory, '{}.tif'.format(stem))

        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._apply_stack, arglist)

    def _apply_stack(self, stack=0):
        """Apply zstack shading correction for a channel."""

        postfix_st = self._suffix_formats['s'].format(stack)
        basename = self.format_([self.dataset, self.suffix, postfix_st])
        outstem = os.path.join(self.directory, basename)

        iminfo = get_image_info(self.image_in)

        out = create_output(None, iminfo['zstack_shape'], iminfo['dtype'])

        # Get stack data (CZYX)  # TODO/FIXME: timepoint / assumes T is squeezed
        c_axis = 0
        data = np.squeeze(read_zstack(self.image_in, stack, out))

        if self.clipping_mask:
            clip_mask = create_clipping_mask(data, axis=c_axis)

        if self.correct:
            data = correct_zstack(data, c_axis, self.shadingpat)

        if self.clipping_mask:
            data = np.append(data, clip_mask.astype(data.dtype), axis=c_axis)

        props = {
            'axlab': 'zyxc',
            'shape': iminfo['dims_zyxc'],
            'elsize': iminfo['elsize_zyxc'],
            'dtype': iminfo['dtype'],
            }

        if self.write_to_tif:

            props['axlab'] = props['axlab'][:3]
            props['shape'] = props['shape'][:3]
            props['elsize'] = props['elsize'][:3]

            for ch in range(0, data.shape[c_axis]):
                postfix_ch = self._suffix_formats['c'].format(ch)
                basename = self.format_([self.dataset, self.suffix, postfix_ch, postfix_st])
                outputpath = os.path.join(self.directory, '{}.tif'.format(basename))

                mo = Image(outputpath, **props)
                mo.create()
                slcs = [slc for slc in mo.slices]
                slcs.insert(c_axis, slice(ch, ch+1))
                mo.write(data[tuple(slcs)])
                mo.close()

        else:

            idxs = [1, 2, 3, 0]  # czyx to zyxc
            data = np.transpose(data, axes=idxs)
            mo = Image("{}{}.h5/data".format(outstem, postfix_st), **props)
            mo.create()
            mo.write(data)
            mo.close()

    def _get_info_dict(self, filestem, info_dict={}, channel=None):

        info_dict['parameters'] = self.load_dumped_pars()
        info_dict['filestem'] = filestem
        info_dict['clip_threshold'] = 0.75
        info_dict['res'] = 10000

        return info_dict

    def _gen_subgrid(self, f, gs, channel=None):
        """Generate the axes for printing the report."""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(3, 2)

        dimdict = {
            'X': {'row': 0, 'xlab': 'Y'},
            'Y': {'row': 1, 'xlab': 'X'},
            'Z': {'row': 2, 'xlab': 'Z'},
            }
        p = {'m': ({'title': '{m} over {d}', 'loc': 'lc'},
                   {'col': 0, 'slc': slice(None, -2),
                   'spines': ['top', 'left'],
                   'ticks': [], 'ticks_position': 'right'}),
             'n': ({'title': '{d} profile', 'loc': 'rc'},
                   {'col': 1, 'slc': slice( 2, None),
                    'spines': ['top', 'right'],
                    'ticks': [0.6, 1.0, 1.4], 'ticks_position': 'left'})}

        def gen_axes(f, sg, sg_width=9, row=0, col=0, xlab='', slc=None, spines=[], ticks=[], ticks_position='left'):
            gs_sp = gs[row, col].subgridspec(1, sg_width)
            ax = f.add_subplot(gs_sp[slc])
            for l in spines:
                ax.spines[l].set_visible(False)
            if ticks:
                ax.yaxis.set_ticks(ticks)
            ax.yaxis.set_ticks_position(ticks_position)
            ax.tick_params(axis='both', direction='in')
            ax.set_xlabel('{} [px]'.format(xlab), loc='center', labelpad=-7)
            return ax

        # Profile axes.
        sg_width = 9
        for k, dd in dimdict.items():
            for j, (td, pd) in p.items():
                axname = j + k
                t = td['title'].format(m=self.metric, d=k)
                titles[axname] = (t, td['loc'], 0)
                axdict[axname] = self._gen_axes(f, gs01, sg_width=sg_width, **dd, **pd)

        titles['mZ'] = ('Z plane selections', 'lc', 0)

        del titles['nZ']
        axdict['nZ'].remove()

        # Image axes.
        titles['I'] = ('Shading image', 'rc', 0)
        gs_sp = gs01[2, 1].subgridspec(2*sg_width, sg_width)
        ax = f.add_subplot(gs_sp[p['n'][1]['slc']])
        ax.set_xlabel('X', labelpad=-7)
        ax.set_ylabel('Y', labelpad=-7)
        ax.tick_params(axis='both', direction='in')
        ax.invert_yaxis()
        for l in ['top', 'right', 'bottom', 'left']:
            ax.spines[l].set_visible(False)
        axdict['I'] = ax
        # colorbar axes
        ax = f.add_subplot(gs_sp[-2:-1, 3:-1])
        axdict['Ic'] = ax

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict, clip_color=[1, 0, 0, 1], n_colors=100):
        """Plot graph with shading image."""

        ax = axdict['I']

        clip_threshold = info_dict['clip_threshold']
        clipped = self._clipped_colormap(clip_threshold, clip_color, n_colors)

        img = imread('{}.tif'.format(info_dict['filestem']))
        infun = np.iinfo if img.dtype.kind in 'ui' else np.finfo
        img = img.transpose() / int(infun(img.dtype).max)  # TODO: img_as_float?

        im = ax.imshow(img, cmap=clipped, vmin=0, vmax=1, origin='lower')
        ax.xaxis.set_ticks([0, img.shape[0] - 1])
        ax.yaxis.set_ticks([0, img.shape[1] - 1])

        cbar = self._plot_colorbar(
            f, ax, im, cax=axdict['Ic'],
            orientation='horizontal', loc='left',
            clip_threshold=clip_threshold,
            )
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

    def _plot_profiles(self, f, axdict, info_dict, clip_threshold=0.75, res=10000, fac=0.05):
        """Plot graphs with profiles."""

        for dim, c in {'X': 'g', 'Y': 'b'}.items():
            npzfile = '{}_{}.npz'.format(info_dict['filestem'], dim)
            ddict = np.load(npzfile)
            self._plot_profile(ddict, axdict, dim, c, info_dict['clip_threshold'], fac)

    def _plot_profile(self, ddict, axdict, dim='X', c='r', clip_threshold=0.75, fac=0.05):
        """Plot graphs with profiles."""

        # TODO: legends

        data_sel = ddict['sel']
        z_sel = ddict['z_sel']

        x = np.linspace(0, data_sel.shape[1] - 1, data_sel.shape[1])

        y_min = self._power_rounder(0)  # FIXME
        y_max = self._power_rounder(np.amax(data_sel))
        yticks = [y_min, y_min + 0.5 * y_max, y_max]

        # Plot selected profiles.
        ax = axdict['m'+dim]
        colors = matplotlib.cm.jet(np.linspace(0, 1, data_sel.shape[0]))
        # TODO: colorbar (legend)
        for i in range(0, data_sel.shape[0]):
            ax.plot(x, data_sel[i, :], color=colors[i], linewidth=0.5)
        ax.set_ylim([y_min, y_max])
        ax.xaxis.set_ticks([0, len(x) - 1])
        ax.yaxis.set_ticks(yticks)

        # Plot mean with confidence interval, normalized fit and threshold.
        ax = axdict['n'+dim]
        dn = ddict['norm']
        dnm = ddict['norm_mean'].transpose()
        df = ddict['fitp'].transpose()
        dfn = ddict['fitp_norm'].transpose()
        ax.plot(dnm, color='k', linewidth=1, linestyle='-')
        ax.set_ylim([0.5, 1.5])
        ax.xaxis.set_ticks([0, len(dnm) - 1])
        ci = 1.96 * np.std(dn, axis=0) / np.mean(dn, axis=0)
        ax.fill_between(x, dnm - ci, dnm + ci, color='b', alpha=.1)
        ax.plot(df, color='k', linewidth=1, linestyle='--')
        ax.plot(dfn, color=c, linewidth=2, linestyle='--')
        ax.axhline(1, xmin=0, xmax=1, color='k', linewidth=0.5, linestyle=':')
        ax.axhline(clip_threshold, xmin=0, xmax=1, color='r', linewidth=0.5, linestyle=':')

        # Plot Z profile and and ticks to indicate selected planes.
        ax = axdict['mZ']
        dm_Z = ddict['dm']
        ax.plot(dm_Z, color='k', linewidth=1, linestyle='-')
        ax.set_ylim([y_min, y_max])
        ax.xaxis.set_ticks([0, len(dm_Z) - 1])
        ax.yaxis.set_ticks(yticks)
        yd = {'X': [1-fac, 1], 'Y': [0, fac]}
        for i in z_sel:
            ax.axvline(i, ymin=yd[dim][0], ymax=yd[dim][1], color=c, linewidth=0.5, linestyle='-')
        ax.axvspan(min(z_sel), max(z_sel), alpha=0.1, color='k')

def read_tiled_plane(image_in, channel, plane):
    """Read a plane of a tiled czi/lif file."""

    dstack = []

    if image_in.endswith('.czi'):

        # order of subblock_directory: CZM
        czi = czifile.CziFile(image_in)
        n_planes = czi.shape[czi.axes.index('Z')]
        n_channels = czi.shape[czi.axes.index('C')]
        sbd_channel = [sbd for sbd in czi.subblock_directory[channel::n_channels]]

        for sbd in sbd_channel[plane::n_planes]:
            dstack.append(np.squeeze(sbd.data_segment().data()))

    elif image_in.endswith('.lif'):

        m_idx = 3

        lif = LifFile(image_in)
        lim = lif.get_image(0)
        n_tiles = lim.dims[m_idx]

        for m in range(n_tiles):
            data = lim.get_frame(z=plane, c=channel, t=0, m=m, return_as_np=True)
            dstack.append(data)

    else:

        print('Sorry, only czi and lif implemented for now...')
        return

    return dstack


def select_z_range(data, quant_thr=0.8, z_range=[]):
    """Select planes of a defined range or with the highest medians."""

    if not z_range:
        dm = np.median(data, axis=1)
        dq = np.quantile(dm, quant_thr)
        z_sel = np.where(dm>dq)[0]
    else:
        z_sel = np.array(range(z_range[0], z_range[1]))
    data = data[z_sel, :]

    return data, dm, z_sel


def normalize_profile(data):
    """Normalize data between 0 and 1."""

    data_mean = np.mean(data, axis=0)
    data_norm = data / np.tile(data[:, -1], [data.shape[1], 1]).transpose()
    data_norm_mean = np.mean(data_norm, axis=0)

    return data_mean, data_norm, data_norm_mean


def fit_profile(data, order=3):
    """Fit a polynomial to data."""

    x = np.linspace(0, data.shape[0] - 1, data.shape[0])

    p = np.poly1d(np.polyfit(x, data, order))
    data_fitted = p(x)
    data_fitted_norm = data_fitted / np.amax(data_fitted)

    return data_fitted, data_fitted_norm


def create_clipping_mask(data, axis=0):
    """Create a mask of voxels at the ends of the datarange."""

    try:
        dmax = np.iinfo(data.dtype).max
    except ValueError:
        dmax = np.finfo(data.dtype).max

    mask = np.any(data==dmax, axis=axis)

    if data.ndim == 4:
        mask = np.expand_dims(mask, axis=0)

    return mask


def correct_zstack(data, c_axis=0, shadingpat=''):
    """Divide each plane in a zstack by the shading image."""

    if not shadingpat:
        return data

    shadingdata = []

    for ch in range(data.shape[c_axis]):
        filepath_shading = shadingpat.format(ch)
        shading_img = img_as_float(imread(filepath_shading))
        shadingdata.append(np.tile(shading_img, (data.shape[1], 1, 1)))

    data = data / np.stack(shadingdata, axis=c_axis)

#    TODO: efficiency tradeoff in cpu and memory (30GB per stack)
#        dtype = data.dtype
#        data = data.astype('float')
#        for z in range(data.shape[1]):
#            data[ch, z, :, :] /= shading_img
#    return data.astype(dtype)

    return data


def get_image_info(image_in):
    """Retrieve selected image metadata."""

    # TODO: move to __init__.py function or Image class

    iminfo = {}

    if image_in.endswith('.czi'):

        czi = czifile.CziFile(image_in)

        iminfo['dtype'] = czi.dtype

        iminfo['nchannels'] = czi.shape[czi.axes.index('C')]
        iminfo['ntimepoints'] = czi.shape[czi.axes.index('T')]
        iminfo['nplanes'] = czi.shape[czi.axes.index('Z')]
        iminfo['ncols'] = czi.shape[czi.axes.index('Y')]
        iminfo['nrows'] = czi.shape[czi.axes.index('X')]
        n = iminfo['nchannels'] * iminfo['ntimepoints'] * iminfo['nplanes']
        iminfo['nstacks'] = len(czi.filtered_subblock_directory) // n

        zstack_shape = list(czi.filtered_subblock_directory[0].shape)
        zstack_shape[czi.axes.index('C')] = iminfo['nchannels']
        zstack_shape[czi.axes.index('T')] = iminfo['ntimepoints']
        zstack_shape[czi.axes.index('Z')] = iminfo['nplanes']
        iminfo['zstack_shape'] = zstack_shape
        iminfo['tilesize'] = zstack_shape[-3:-1]

        zyxc_idxs = [czi.axes.index(dim) for dim in 'ZYXC']
        # zyxc_idxs = [8, 9, 10, 6]  # TODO: check with zstack
        iminfo['dims_zyxc'] = [iminfo['zstack_shape'][idx] for idx in zyxc_idxs]
        iminfo['elsize_zyxc'] = czi_get_elsize(czi) + [1]

        iminfo['stack_offsets'] = czi_tile_offsets(czi, iminfo)

    elif image_in.endswith('.lif'):

        lim = LifFile(image_in).get_image(0)  # FIXME: choice of image / series

        iminfo['dtype'] = lim.dtype

        iminfo['nchannels'] = lim.dims[0]
        iminfo['nplanes'] = lim.dims[1]
        iminfo['ntimepoints'] = lim.dims[2]
        iminfo['nstacks'] = lim.dims[3]
        iminfo['ncols'] = lim.dims[4]  # FIXME: this merged shape for czi, tilesize for lif
        iminfo['nrows'] = lim.dims[5]

        m_idx = 3
        iminfo['zstack_shape'] = lim.dims[:m_idx] + lim.dims[m_idx+1:]
        iminfo['tilesize'] = [iminfo['ncols'], iminfo['nrows']]

        zyxc_idxs = [1, 4, 5, 0]
        iminfo['dims_zyxc'] = [lim.dims[idx] for idx in zyxc_idxs]
        iminfo['elsize_zyxc'] = [1./lim.scale[idx] for idx in zyxc_idxs]

        v_offsets = np.zeros([iminfo['nstacks'], 4])
        for i in range(iminfo['nstacks']):
            v_offsets[i, :] = [
                i,
                lim.tile_positions[0, i] / iminfo['elsize_zyxc'][2],
                lim.tile_positions[1, i] / iminfo['elsize_zyxc'][1],
                0]

        iminfo['stack_offsets'] = v_offsets

    elif image_in.endswith('.ims'):

        from stapl3d import get_imageprops
        props = get_imageprops(image_in)
        iminfo = {
            'nchannels': props['shape'][props['axlab'].index('c')],
            'ntimepoints': props['shape'][props['axlab'].index('t')],
            'nplanes': props['shape'][props['axlab'].index('z')],
            'nstacks': 1,
            }

    else:

        print('Sorry, only czi and lif implemented for now...')
        return

    iminfo['channels'] = list(range(iminfo['nchannels']))
    iminfo['timepoints'] = list(range(iminfo['ntimepoints']))
    iminfo['planes'] = list(range(iminfo['nplanes']))
    iminfo['stacks'] = list(range(iminfo['nstacks']))

    return iminfo


def czi_get_elsize(czi):
    """Get the zyx resolutions from the czi metadata."""

    segment = czifile.Segment(czi._fh, czi.header.metadata_position)
    data = segment.data().data()
    md = etree.fromstring(data.encode('utf-8'))

    # FIXME: use find() method on etree items
    try:
        elsize_x = float(md[0][3][0][0][0].text) * 1e6
        elsize_y = float(md[0][3][0][1][0].text) * 1e6
        elsize_z = float(md[0][3][0][2][0].text) * 1e6
    except IndexError:
        elsize_x, elsize_y, elsize_z = 1, 1, 1

    return [elsize_z, elsize_y, elsize_x]


def czi_tile_offsets(czi, iminfo):
    """Get the offset coordinates of a czi zstack."""

    # first dir of eacxh zstack: C[8]Z[84]M[286]
    stack_stride = iminfo['nchannels'] * iminfo['ntimepoints'] * iminfo['nplanes']
    sbd_zstacks0 = [sbd for sbd in czi.subblock_directory[::stack_stride]]
    v_offsets = np.zeros([iminfo['nstacks'], 4])
    for i, directory_entry in zip(range(iminfo['nstacks']), sbd_zstacks0):
        subblock = directory_entry.data_segment()
        for sbdim in subblock.dimension_entries:
            if sbdim.dimension == 'X':
                x_osv = sbdim.start
                x_osc = sbdim.start_coordinate
            if sbdim.dimension == 'Y':
                y_osv = sbdim.start
                y_osc = sbdim.start_coordinate
            if sbdim.dimension == 'Z':
                z_osv = sbdim.start
                z_osc = sbdim.start_coordinate
        v_offsets[i, :] = [i, x_osv, y_osv, z_osv]

    return v_offsets


def read_zstack(image_in, zstack_idx, out=None):
    """Read the zstack data."""

    if out is None:
        iminfo = get_image_info(image_in)
        out = create_output(None, iminfo['zstack_shape'], iminfo['dtype'])

    if image_in.endswith('.czi'):

        czi = czifile.CziFile(image_in)

        start = czi.start
        n = czi.shape[czi.axes.index('C')] * czi.shape[czi.axes.index('Z')]
        start_idx = n * zstack_idx
        stop_idx = start_idx + n
        zstack = czi.filtered_subblock_directory[start_idx:stop_idx]

        for directory_entry in zstack:
            subblock = directory_entry.data_segment()
            tile = subblock.data(resize=False, order=0)
            index = [slice(i-j, i-j+k) for i, j, k in zip(directory_entry.start, start, tile.shape)]
            index[czi.axes.index('Y')] = slice(0, tile.shape[czi.axes.index('Y')], None)
            index[czi.axes.index('X')] = slice(0, tile.shape[czi.axes.index('X')], None)
            out[tuple(index)] = tile

    elif image_in.endswith('.lif'):

        lim = LifFile(image_in).get_image(0)  # FIXME: choice of image / series

        out = lim.get_stack_np(zstack_idx)  # CZTYX

    else:

        print('Sorry, only czi and lif implemented for now...')
        return

    return out


if __name__ == "__main__":
    main(sys.argv[1:])
