#!/usr/bin/env python

"""Correct z-stack shading.

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

from stapl3d import parse_args, Stapl3r, Image

logger = logging.getLogger(__name__)


def main(argv):
    """Correct z-stack shading."""

    steps = ['estimate', 'process', 'postprocess', 'apply']
    args = parse_args('shading', steps, *argv)

    deshad3r = Deshad3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        deshad3r._fun_selector[step]()


class Deshad3r(Stapl3r):
    """Correct z-stack shading.

    Parameters
    ----------
    image_in : string
        Path to dataset.
    parameter_file : string
        Path to yaml parameter file.
    module_id : string
        Name of the STAPL3D module.
    step_id: string
        Identifier of the yaml parameterfile entry.
    directory : string
        Name of output subdirectory.
    prefix : string
        Output prefix.
    max_workers : int
        Maximal number of cores to use for processing.

    Attributes
    ----------
    channels : list []
        List of channels indices to process, default 'all'.
    planes : list []
        List of plane indices to process, default 'all'.
    noise_threshold : 1000
        Threshold to discard background, default 1000.
    metric : string 'median'
        Metric for creating profiles, default 'median', else 'mean'.
    quantile_threshold : float 0.8
        Quantile at which planes are discarded, default 0.8.
    polynomial_order : int 3
        Order of the polynomial to fit the profile, default 3.
    stacks : list []
        List of stack indices to correct.
    clipping_mask': bool False
        Create an additional mask of pixels that clip outside the datarange,
        default False.
    correct : bool True
        Apply the shading correction to the z-stacks, default True.
    write_to_tif : bool True
        Write the corrected z-stacks to tifs, default True.

    Examples
    --------
    # TODO


    """

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Deshad3r, self).__init__(
            image_in, parameter_file,
            module_id='shading',
            **kwargs,
            )

        self._fun_selector = {
            'estimate': self.estimate,
            'process': self.process,
            'postprocess': self.postprocess,
            'apply': self.apply,
            }

        self._parallelization = {
            'estimate': ['channels', 'planes'],
            'process': ['channels'],
            'postprocess': [],
            'apply': ['stacks'],
            }

        self._parameter_sets = {
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('noise_threshold', 'metric'),
                'spar': ('_n_workers', 'channels', 'planes'),
                },
            'process': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('quantile_threshold', 'polynomial_order'),
                'spar': ('_n_workers', 'channels'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': (),
                },
            'apply': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('clipping_mask', 'correct', 'write_to_tif'),
                'spar': ('_n_workers', 'stacks'),
                },
            }

        self._parameter_table = {  # FIXME: causes error in report generation
#            'noise_threshold': 'Noise threshold',
#            'metric': 'Metric',
#            'quantile_threshold': 'Quantile threshold',
#            'polynomial_order': 'Polynomial order',
            }

        default_attr = {
            'channels': [],
            'planes': [],
            'noise_threshold': 1000,  # FIXME: automate
            'metric': 'median',
            'quantile_threshold': 0.8,
            'polynomial_order': 3,
            'stacks': [],
            'clipping_mask': False,
            'correct': True,
            'write_to_tif': True,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

    def _init_paths(self):

        stem = self._build_path()
        ppat = self._build_path(suffixes=[{'c': 'p', 'z': 'p'}])
        pmat = self._build_path(suffixes=[{'c': 'p', 'z': '?'}])
        cpat = self._build_path(suffixes=[{'c': 'p'}])
        cmat = self._build_path(suffixes=[{'c': '?'}])
        spat = self._build_path(suffixes=[{'c': 'p', 's': 'p'}])

        self._paths = {
            'estimate': {
                'inputs': {
                    'data': self.image_in,
                    },
                'outputs': {
                    'metrics': f'{ppat}.npz',
                    },
                },
            'process': {
                'inputs': {
                    'metrics': f'{pmat}.npz',
                    },
                'outputs': {
                    'profile_X': f'{cpat}_X.npz',
                    'profile_Y': f'{cpat}_Y.npz',
                    'shading': f'{cpat}.tif',
                    'report': f'{cpat}.pdf',
                    },
                },
            'postprocess': {
                'inputs': {
                    'report': f'{cmat}.pdf',
                    },
                'outputs': {
                    'report': f'{stem}.pdf',
                    },
                },
            'apply': {
                'inputs': {
                    'data': self.image_in,
                    'shading': f'{cpat}.tif',
                    },
                'outputs': {
                    'stacks': f'{spat}.tif',
                    },
            },
        }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def estimate(self, **kwargs):
        """Estimate z-stack shading.

        channels=[],
        planes=[],
        noise_threshold=0,
        metric='median',
        """

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_plane, arglist)

    def _estimate_plane(self, channel, plane):
        """Estimate the x- and y-profiles for a channel in a czi/lif file."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs, reps={'c': channel, 'z': plane})

        # Compute median values per plane for X and Y concatenation.
        fp = inputs['data']    # for testing stacks in files '{f}.czi'  #
        dstack = read_tiled_plane(fp, channel, plane)
        out = {}
        for axis in [0, 1]:
            dstacked = np.concatenate(dstack, axis=0)
            ma_data = np.ma.masked_array(dstacked, dstacked < self.noise_threshold)
            if self.metric == 'median':
                out[axis] = np.ma.median(ma_data, axis=0)
            else:
                out[axis] = np.ma.mean(ma_data, axis=0)

        np.savez(outputs['metrics'], X=out[1], Y=out[0])

    def process(self, **kwargs):
        """Fit shading profiles for all channels.

        channels=[],
        quantile_threshold=0.8,
        polynomial_order=3,
        """

        arglist = self._prep_step('process', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._process_channel, arglist)

    def _process_channel(self, channel):
        """Fit shading profiles for a channel."""

        inputs = self._prep_paths(self.inputs, reps={'c': channel})
        outputs = self._prep_paths(self.outputs, reps={'c': channel})

        npzs = glob(inputs['metrics'])
        npzs.sort()

        iminfo = get_image_info(self.image_in)
        img_fitted = np.ones(iminfo['tilesize'], dtype='float')

        def process_medians(meds, q_thr, order, outputpath):
            """Select, normalize and fit profiles; and save for report."""

            sel, dm, z_sel = select_z_range(meds, q_thr)
            m, n, nm = normalize_profile(sel)
            f, fn = fit_profile(nm, order)

            np.savez(outputpath,
                sel=sel, dm=dm, z_sel=z_sel,
                mean=m, norm=n, norm_mean=nm,
                fitp=f, fitp_norm=fn,
                )

            return fn

        # TODO: write to single npz file per channel
        for dim in 'XY':

            meds = []
            for npzfile in npzs:
                medians = np.load(npzfile, allow_pickle=True)
                meds.append(medians[dim])
                medians.close()

            fitp_norm = process_medians(
                np.stack(meds, axis=0),
                self.quantile_threshold, self.polynomial_order,
                outputs[f'profile_{dim}'],
                )

            if dim == 'X':
                img_fitted *= fitp_norm
            elif dim == 'Y':
                img_fitted *= fitp_norm[:, np.newaxis]

        # Save estimated shading image.  # FIXME: handle floats here
        img = np.array(img_fitted * np.iinfo(iminfo['dtype']).max,
                       dtype=iminfo['dtype'])
        imsave(outputs['shading'], img)

        # Print a report page to pdf.
        self.report(outputpath=outputs['report'],
                    channel=channel,
                    inputs=inputs, outputs=outputs)

        # Delete planes
        for npzfile in npzs:
            os.remove(npzfile)

    def postprocess(self, **kwargs):
        """Merge reports."""

        arglist = self._prep_step('postprocess', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._postprocess, arglist)

    def _postprocess(self, foo=0):
        """Merge reports."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        pdfs = glob(inputs['report'])
        pdfs.sort()
        self._merge_reports(pdfs, outputs['report'])

    def apply(self, **kwargs):
        """Apply zstack shading correction for all channels.

        stacks=[],
        clipping_mask=False,
        correct=True,
        shadingpat='',
        write_to_tif=True,
        """

        arglist = self._prep_step('apply', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._apply_stack, arglist)

    def _apply_stack(self, stack=0):
        """Apply zstack shading correction for a channel."""

        inputs = self.inputs
        outputs = self.outputs

        iminfo = get_image_info(self.image_in)
        out = create_output(None, iminfo['zstack_shape'], iminfo['dtype'])

        # Get stack data (CZYX)  # TODO/FIXME: timepoint / assumes T is squeezed
        c_axis = 0
        data = np.squeeze(read_zstack(inputs['data'], stack, out))

        if self.clipping_mask:
            clip_mask = create_clipping_mask(data, axis=c_axis)

        if self.correct:
            data = correct_zstack(data, c_axis, inputs['shading'])

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

            for channel in range(0, data.shape[c_axis]):

                outpath = outputs['stacks'].format(c=channel, s=stack)
                mo = Image(outpath, **props)
                mo.create()
                slcs = [slc for slc in mo.slices]
                slcs.insert(c_axis, slice(channel, channel + 1))
                mo.write(data[tuple(slcs)])
                mo.close()

        # else:
        #
        #     outpath = self._outputs[0].format(stack)
        #
        #     idxs = [1, 2, 3, 0]  # czyx to zyxc
        #     data = np.transpose(data, axes=idxs)
        #     mo = Image(outpath, **props)
        #     mo.create()
        #     mo.write(data)
        #     mo.close()

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        kwargs['clip_threshold'] = 0.75

        return kwargs

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

        img = imread(info_dict['outputs']['shading'])
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
            ddict = np.load(info_dict['outputs'][f'profile_{dim}'])
            self._plot_profile(ddict, axdict, dim, c, info_dict['clip_threshold'], fac)

    def _plot_profile(self, ddict, axdict, dim='X', c='r', clip_threshold=0.75, fac=0.05):
        """Plot graphs with profiles."""

        # TODO: legends

        data_sel = ddict['sel']
        z_sel = ddict['z_sel']

        x = np.linspace(0, data_sel.shape[1] - 1, data_sel.shape[1])

        y_min = self._power_rounder(0)  # FIXME
        y_max = self._power_rounder(np.amax(data_sel), roundfun=np.ceil)
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

        if '{f}' in image_in:
            import re
            mat = '*'
            pat = r"{[^{}]+}"
            foo = re.sub(pat, mat, image_in)
            fps = sorted(glob(os.path.abspath(foo)))
        else:
            fps = [image_in]

        for fp in fps:
            # order of subblock_directory: CZM
            czi = czifile.CziFile(fp)
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
        filepath_shading = shadingpat.format(c=ch)
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

        if '{f}' in image_in:
            import re
            mat = '*'
            pat = r"{[^{}]+}"
            foo = re.sub(pat, mat, image_in)
            fps = sorted(glob(os.path.abspath(foo)))
            image_in = fps[0]
            iminfo['nstacks'] = len(fps)

        czi = czifile.CziFile(image_in)

        iminfo['dtype'] = czi.dtype

        iminfo['nchannels'] = czi.shape[czi.axes.index('C')]
        iminfo['ntimepoints'] = czi.shape[czi.axes.index('T')]
        iminfo['nplanes'] = czi.shape[czi.axes.index('Z')]
        iminfo['ncols'] = czi.shape[czi.axes.index('Y')]
        iminfo['nrows'] = czi.shape[czi.axes.index('X')]
        n = iminfo['nchannels'] * iminfo['ntimepoints'] * iminfo['nplanes']
        if not '{f}' in image_in:
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

    elif image_in.endswith('.ims') or image_in.endswith('.bdv') or '.h5' in image_in:

        # TODO: take group of h5 channels as input
        from stapl3d import get_imageprops
        props = get_imageprops(image_in)

        def dimfinder(props, al):
            return props['shape'][props['axlab'].index(al)]  if al in props['axlab'] else 1

        iminfo = {
            'nchannels': dimfinder(props, 'c'),
            'ntimepoints': dimfinder(props, 't'),
            'nplanes': dimfinder(props, 'z'),
            'ncols': dimfinder(props, 'y'),
            'nrows': dimfinder(props, 'x'),
            'nstacks': dimfinder(props, 'm'),
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

    elsize_x = float(md.findall('.//ScalingX')[0].text) * 1e6
    elsize_y = float(md.findall('.//ScalingY')[0].text) * 1e6
    elsize_z = float(md.findall('.//ScalingZ')[0].text) * 1e6

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
