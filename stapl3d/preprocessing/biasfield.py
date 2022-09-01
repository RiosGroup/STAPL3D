#!/usr/bin/env python

"""Perform N4 bias field correction.

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

import yaml

import numpy as np

from glob import glob

from skimage.transform import resize, downscale_local_mean
from skimage.measure import block_reduce
from skimage.filters import threshold_otsu

import SimpleITK as sitk

from stapl3d import parse_args, Stapl3r, Image, wmeMPI
from stapl3d import get_paths, get_imageprops  # TODO: into Image/Stapl3r
from stapl3d.imarisfiles import (
    find_downsample_factors,
    find_resolution_level,
    ims_linked_channels,
    h5chs_to_virtual,
    create_ref,
    )
from stapl3d.reporting import (
    gen_orthoplot_with_colorbar,
    get_centreslices,
    get_zyx_medians,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Perform N4 bias field correction."""

    steps = ['estimate', 'apply', 'postprocess']
    args = parse_args('biasfield', steps, *argv)

    homogeniz3r = Homogeniz3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        homogeniz3r._fun_selector[step]()


class Homogeniz3r(Stapl3r):
    _doc_main = """Perform N4 bias field correction."""
    _doc_attr = """

    Homogeniz3r Attributes
    ----------
    method : str, default 'biasfield'
        Correction Method.
            'biasfield': N4 bias field correction [Tustison et al., 2010].
            'smooth_division': simple division by very smooth image.
            'attenuation': attenuation correction [REF]
    channels : list, default [0, 1, ..., N]
        List of channels indices to process.
    resolution_level : int, default 'smallest not downsampled in Z'
        Resolution level of the image pyramid to use.
    target_yx : float, default 20
        Target in-plane resolution [um] for the inhomogeneity estimation step.
        Used to calculate downsample_factors from resolution_level to target_yx.
    downsample_factors : dict, default {}
        Downsample factors applied on the image (at resolution_level).
        Supersedes target_yx.
    n_iterations : int, default 50
        Number of iterations for the N4 algorithm (passed to ITK filter).
    n_fitlevels : dict, default {'z': 4, 'y': 4, 'x': 4}
        Number of fitlevels for the N4 algorithm (passed to ITK filter).
    n_bspline_cps : dict, default {'z': 5, 'y': 5, 'x': 5}
        Number of B-splines for the N4 algorithm (passed to ITK filter).
    tasks : int, default 1
        Maximal concurrent threads for estimation (passed to ITK filter).
    blocksize_xy : int, default 1280
        In-plane size of the block used during the 'apply' step.
    """
    _doc_meth = """

    Homogeniz3r Methods
    --------
    run
        Run all steps in the Homogeniz3r module.
    estimate
        Estimate the inhomogeneity for each channel.
    apply:
        Correct the inhomogeneity for each channel.
    postprocess
        Aggregate channels into a symlinked h5 and merge the channel reports.
    view
        View volumes with napari.
    """
    _doc_exam = """

    Homogeniz3r Examples
    --------
    # TODO
    """
    __doc__ = f"{_doc_main}{Stapl3r.__doc__}{_doc_meth}{_doc_attr}{_doc_exam}"

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Homogeniz3r, self).__init__(
            image_in, parameter_file,
            module_id='biasfield',
            **kwargs,
            )

        self._fun_selector = {
            'estimate': self.estimate,
            'apply': self.apply,
            'postprocess': self.postprocess,
            }

        self._parallelization = {
            'estimate': ['channels'],
            'apply': ['channels'],
            'postprocess': [],
            }

        self._parameter_sets = {
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('method',
                         'resolution_level', 'target_yx', 'downsample_factors',
                         'n_iterations', 'n_fitlevels', 'n_bspline_cps'),
                'spar': ('_n_workers', 'channels', 'tasks'),
                },
            'apply': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('downsample_factors', 'blocksize_xy'),
                'spar': ('_n_workers', 'channels'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            }

        self._parameter_table = {
            'resolution_level': 'Resolution level image pyramid',
            'downsample_factors': 'Downsample factors',
            'n_iterations': 'N iterations',
            'n_fitlevels': 'N fitlevels',
            'n_bspline_cps': 'N b-spline components',
            }

        default_attr = {
            'method': 'biasfield',
            'channels': [],
            'resolution_level': -1,
            'target_yx': 20,
            'downsample_factors': {},
            '_downsample_factors_reslev': {},
            'n_iterations': 50,
            'n_fitlevels': {'z': 4, 'y': 4, 'x': 4},
            'n_bspline_cps': {'z': 5, 'y': 5, 'x': 5},
            'blocksize_xy': 1280,
            'tasks': 1,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._images = ['data', 'bias', 'corr']
        self._labels = []

    def _init_paths(self):

        if self.image_in.endswith('.ims'):
            ext = '.ims'
            ids = ''
        else:
            ext = '.h5'
            ids = '/data'

        vols = ['data', 'mask', 'bias', 'corr']
        stem = self._build_path()
        cpat = self._build_path(suffixes=[{'c': 'p'}])
        cmat = self._build_path(suffixes=[{'c': '?'}])

        self._paths = {
            'estimate': {
                'inputs': {
                    'data': self.image_in,
                    'mask': '',  # only from parameterfile for now
                    },
                'outputs': {
                    **{'file': f'{cpat}_ds.h5'},
                    **{ods: f'{cpat}_ds.h5/{ods}' for ods in vols},
                    **{'parameters': f'{cpat}_ds'},
                    **{'report': f'{cpat}_ds.pdf'},
                    },
                },
            'apply': {
                'inputs': {
                    'data': self.image_in,
                    'bias': f'{cpat}_ds.h5/bias',
                    'ims_ref': '',
                    },
                'outputs': {
                    'channels': f'{cpat}{ext}{ids}',
                    },
            },
            'postprocess': {
                'inputs': {
                    'channels': f'{cmat}{ext}',
                    'channels_ds': f'{cmat}_ds.h5',
                    'report': f'{cmat}_ds.pdf',
                    'ims_ref': '',
                    },
                'outputs': {
                    'aggregate': f'{stem}{ext}',
                    'aggregate_ds': f'{stem}_ds.h5',
                    'report': f'{stem}.pdf',
                    },
                },
        }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def estimate(self, **kwargs):
        """Perform inhomogeneity estimation."""

        arglist = self._prep_step('estimate', kwargs)

        self._set_downsampling_parameters(self.inputpaths['estimate']['data'])

        # NOTE: ITK is already multithreaded =>
        # n_workers = 1; pass number of threads to ITK filter
        if self.method == 'biasfield':
            self._n_threads = min(self.tasks, multiprocessing.cpu_count())
            self._n_workers = 1
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_channel, arglist)

    def _estimate_channel(self, channel):
        """Estimate the x- and y-profiles for a channel in a czi file."""

        inputs = self._prep_paths(self.inputs, reps={'c': channel})
        outputs = self._prep_paths(self.outputs, reps={'c': channel})

        ds_im = downsample_channel(
            inputs['data'], self.resolution_level, channel,
            self.downsample_factors, False,
            outputs['data'],
            )
        # FIXME: assuming here that mask is already created from resolution_level!
        ds_ma = downsample_channel(
            inputs['mask'], -1, channel,
            self.downsample_factors, True,
            outputs['mask'],
            )

        if self.method == 'smooth_division':
            ds_bf = smooth_division(
                ds_im, self.sigma, self.normalize_bias,
                outputs['bias'],
                )
        elif self.method == 'attenuation':
            ds_bf = attenuation_correction(
                ds_im, 2, None,
                outputs['bias'],
                )
        else:
            ds_bf = calculate_bias_field(
                ds_im, ds_ma,
                self.n_iterations,
                [self.n_fitlevels[dim] for dim in ds_im.axlab],
                [self.n_bspline_cps[dim] for dim in ds_im.axlab],
                self._n_threads,
                outputs['bias'],
                )
        ds_cr = divide_bias_field(
            ds_im, ds_bf,
            outputs['corr'],
            )

        for im in [ds_im, ds_ma, ds_bf, ds_cr]:
            try:
                im.close()
            except:
                pass

        self.dump_parameters(self.step, outputs['parameters'])
        pars = {k: getattr(self, k) for k in self._parameter_table.keys()}
        self.report(outputs['report'],
                    channel=channel,
                    parameters=pars,
                    inputs=inputs, outputs=outputs)

    def _set_downsampling_parameters(self, inputpath):
        """Set the downsample factors of the volume.

        downsampling of the input volume is composed of:
        1. the downsampling in the image pyramid
        2. additional downsampling to reach the target resolution
        """

        self._set_resolution_level(inputpath)
        self._set_downsample_factors_reslev(inputpath)
        self._set_downsample_factors(inputpath)

    def _set_resolution_level(self, inputpath):
        """Set the resolution level parameter.

        If the resolution_level is set to a negative value,
        find the resolution level in the image pyramid that
        has no downsampling in the z-dimension.
        """

        if self.resolution_level >= 0:
            return

        if ('.ims' in inputpath or '.bdv' in inputpath):
            self.resolution_level = find_resolution_level(inputpath)

    def _set_downsample_factors_reslev(self, inputpath):
        """Set the downsampling factors associated with the resolution level."""

        if self._downsample_factors_reslev:
            return

        if ('.ims' in inputpath or '.bdv' in inputpath):
            axlab = 'zyxct'
            dsfacs = find_downsample_factors(inputpath, 0, self.resolution_level)

            dsfacs = [d.item() for d in np.array(list(dsfacs) + [1, 1])]
            self._downsample_factors_reslev = dict(zip(axlab, dsfacs))

    def _set_downsample_factors(self, inputpath):
        """Set the resolution level parameter."""

        if self.downsample_factors:
            return

        # downsample factors to reach target resolution
        im = Image(inputpath, permission='r', reslev=self.resolution_level)
        im.load(load_data=False)
        im.close()

        target = {
            'z': im.elsize[im.axlab.index('z')],
            'y': self.target_yx,
            'x': self.target_yx,
            }

        axlab = 'zyxct'
        dsfacs = [target[dim] / im.elsize[im.axlab.index(dim)] for dim in 'zyx']
        dsfacs = [np.round(dsfac).astype('int') for dsfac in dsfacs]
        dsfacs[1] = dsfacs[2] = min(dsfacs[1], dsfacs[2])

        dsfacs = [d.item() for d in np.array(list(dsfacs) + [1, 1])]
        self.downsample_factors = dict(zip(axlab, dsfacs))

    def postprocess(self, **kwargs):
        """Merge bias field estimation files."""

        self._prep_step('postprocess', kwargs)

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        gather_4D(inputs['channels'], outputs['aggregate'],
                  ['data'])
        gather_4D(inputs['channels_ds'], outputs['aggregate_ds'],
                  ['data', 'bias', 'corr'])

        pdfs = glob(inputs['report'])
        pdfs.sort()
        self._merge(pdfs, outputs['report'], self._merge_reports)

    def apply(self, **kwargs):
        """Apply N4 bias field correction."""

        arglist = self._prep_step('apply', kwargs)

        self._set_downsampling_parameters(self.inputpaths['estimate']['data'])

        if self.outputpaths['apply']['channels'].endswith('.ims'):
            if not self.inputpaths['apply']['ims_ref']:
                filepath_ims = self.inputpaths['estimate']['data']
                filepath_ref = filepath_ims.replace('.ims', '_ref.ims')
                # TODO: protect existing files
                create_ref(filepath_ims)
                self.inputpaths['apply']['ims_ref'] = filepath_ref
                self.inputpaths['postprocess']['ims_ref'] = filepath_ref

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._apply_channel, arglist)

    def _apply_channel(self, channel=None):
        """Correct inhomogeneity of a channel."""

        inputs = self._prep_paths(self.inputs, reps={'c': channel})
        outputs = self._prep_paths(self.outputs, reps={'c': channel})

        self._set_downsampling_parameters(inputs['data'])

        # Create the image objects.
        in_place = inputs['data'] == outputs['channels']
        im = Image(inputs['data'], permission='r+' if in_place else 'r')
        im.load(load_data=False)

        if self.method == 'attenuation':

            filepath = inputs['bias'].replace('.h5/bias', '_att.npz')
            npzfile = np.load(filepath)
            means = npzfile['means']
            stds = npzfile['stds']

            if self.ref_idx is None:
                self.ref_idx = np.argmax(means)
            ref_mean = means[self.ref_idx]
            ref_std = stds[self.ref_idx]

            # im_ch = im.extract_channel(channel)
            # attenuation_correction_apply(im_ch, npzfile['means'], npzfile['stds'], ref_idx=None, outputpath=outputs['channels'])
            # im.close()
            # return

            # print(ref_mean, ref_std)
            # for m, s in zip(means, stds):
            #     print(m, s)

        bf = Image(inputs['bias'], permission='r')
        bf.load(load_data=False)

        # Create the output image.
        if inputs['ims_ref']:
            shutil.copy2(inputs['ims_ref'], outputs['channels'])
            mo = Image(outputs['channels'])
            mo.load()
        # TODO: write to bdv pyramid
        else:
            props = im.get_props()
            if len(im.dims) > 4:
                props = im.squeeze_props(props, dim=4)
            if len(im.dims) > 3:
                props = im.squeeze_props(props, dim=3)
            props['dtype'] = 'float'
            mo = Image(outputs['channels'], **props)
            mo.create()

        # Get the downsampling between full and bias images.
        p = self._load_dumped_step(self.directory, self._module_id, 'estimate')
        self.resolution_level = p['resolution_level']
        self._set_downsampling_parameters(p['inputs']['data'])

        downsample_factors = {}
        for dim, dsfac in self._downsample_factors_reslev.items():
            downsample_factors[dim] = dsfac * self.downsample_factors[dim]
        downsample_factors = tuple([downsample_factors[dim] for dim in im.axlab])

        # Channel selection for 4D inputs.
        if channel is not None:
            if 'c' in im.axlab:
                im.slices[im.axlab.index('c')] = slice(channel, channel + 1)
            if 'c' in bf.axlab:
                bf.slices[bf.axlab.index('c')] = slice(channel, channel + 1)

        # Set up the blocks
        blocksize = list(im.dims)
        blockmargin = [0] * len(blocksize)
        if self.blocksize_xy:
            for dim in 'xy':
                idx = im.axlab.index(dim)
                blocksize[idx] = self.blocksize_xy
                if im.chunks is not None:
                    blockmargin[idx] = im.chunks[idx]
                else:
                    blockmargin[idx] = 128

        mpi = wmeMPI(usempi=False)
        mpi.set_blocks(im, blocksize, blockmargin)
        mpi.scatter_series()
        mpi_nm = wmeMPI(usempi=False)  # no margin
        mpi_nm.set_blocks(im, blocksize)


        for i in mpi.series:

            block = mpi.blocks[i]

            data_shape = list(im.slices2shape(block['slices']))

            block_nm = mpi_nm.blocks[i]

            it = zip(block['slices'], block_nm['slices'], blocksize, data_shape)

            data_shape = list(im.slices2shape(block_nm['slices']))
            # ??? this does nothing ???

            data_slices = []
            for b_slc, n_slc, bs, ds in it:
                m_start = n_slc.start - b_slc.start
                m_stop = m_start + bs
                m_stop = min(m_stop, ds)
                data_slices.append(slice(m_start, m_stop, None))

            # Get the fullres image block.
            im.slices = block['slices']
            data = im.slice_dataset().astype('float')

            # Get the upsampled bias field.
            if self.method == 'attenuation':
                data_corr = np.zeros_like(data)
                for i, slc in enumerate(data):
                    data_corr[i, :, :] = ref_mean + ref_std * ((slc - means[i]) / stds[i])
                data = data_corr
            else:
                bias = get_bias_field_block(bf, im.slices, data.shape,
                                            downsample_factors)
                data /= bias
                data = np.nan_to_num(data, copy=False)

            # Write the corrected data.
            mo.slices = block_nm['slices']
            if 'c' in im.axlab:
                c_idx = im.axlab.index('c')
                mo.slices[c_idx] = slice(0, 1, 1)  # FIXME: assumes writing to 3D
                data_slices[c_idx] = block['slices'][c_idx]  # is this necessary?

            mo.write(data[tuple(data_slices[:data.ndim])].astype(mo.dtype))

        im.close()
        bf.close()
        mo.close()

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        filepath = kwargs['outputs']['data']
        kwargs['props'] = get_imageprops(filepath)
        kwargs['paths'] = get_paths(filepath)
        kwargs['centreslices'] = get_centreslices(kwargs)

        d = extract_zyx_profiles(kwargs['paths']['file'], kwargs['channel'])
        kwargs = {**kwargs, **d}

        return kwargs

    def _gen_subgrid(self, f, gs, channel=None):
        """3rows-2 columns: 3 image-triplet left, three plots right"""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(3, 2, wspace=0.9)

        pd = {'col': 1, 'slc': slice(None, None), 'spines': ['top', 'right'],
              'ticks': [], 'ticks_position': 'left'}
        dimdict = {'x': {'row': 2}, 'y': {'row': 1}, 'z': {'row': 0}}
        voldict = {
            'data': {'row': 0, 'title': 'downsampled'},
            'corr': {'row': 1, 'title': 'corrected'},
            'bias': {'row': 2, 'title': 'inhomogeneity field'},
            }

        sg_width = 9
        for k, dd in dimdict.items():
            titles[k] = (f'{k.upper()} profiles', 'rc', 0)
            axdict[k] = self._gen_axes(f, gs01, sg_width=sg_width, **dd, **pd)
        for k, vd in voldict.items():
            titles[k] = (vd['title'], 'lcm', 0)
            axdict[k] = gen_orthoplot_with_colorbar(f, gs01[vd['row'], 0], idx=0,
                                                    add_profile_insets=True)
            # titles[k] = (vd['title'], 'lc', 4)
            # axdict[k] = gen_orthoplot_with_profiles(f, gs01[vd['row'], 0])

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict={}):
        """Show images in report."""

        meds = info_dict['medians']
        cslcs = info_dict['centreslices']

        for k, v in cslcs.items():
            cslcs[k]['x'] = cslcs[k]['x'].transpose()

        clim_data = self._get_clim(cslcs['data'])
        clim_bias = self._get_clim(cslcs['bias'])
        clim_bias[0] = min(0, clim_bias[0])
        clim_bias[1] = max(2, clim_bias[1])
        vol_dict = {
            'data': ('right', 'gray', clim_data, 'r'),
            'corr': ('right', 'gray', clim_data, 'g'),
            'bias': ('right', 'rainbow', clim_bias, 'b'),
            }

        for k, (loc, cmap, clim, c) in vol_dict.items():

            for d, aspect, ax_idx in zip('xyz', ['auto', 'auto', 'equal'], [2, 1, 0]):

                ax = axdict[k][ax_idx]

                extent = self._get_extent(cslcs['data'][d], info_dict['props']['elsize'])

                im = ax.imshow(cslcs[k][d], cmap=cmap, extent=extent, aspect=aspect)

                im.set_clim(clim[0], clim[1])
                ax.axis('off')

                if ax_idx == 0:
                    self._plot_colorbar(f, ax, im, cax=axdict[k][3], loc=loc)
                    if k == 'data':
                        self._add_scalebar(ax, extent[1], color='r')

                if len(axdict[k]) > 4:
                    ax = axdict[k][ax_idx + 4]
                    A = meds[k][d]
                    B = list(range(len(meds['data'][d])))
                    ls, lw = '-', 2.0
                    if d == 'y':
                        ax.plot(A, B, color=c, linewidth=lw, linestyle=ls)
                        ax.set_xlim([np.amin(A), np.amax(A)])
                        ax.set_ylim([np.amin(B), np.amax(B)])
                    else:
                        ax.plot(B, A, color=c, linewidth=lw, linestyle=ls)
                        ax.set_xlim([np.amin(B), np.amax(B)])
                        ax.set_ylim([np.amin(A), np.amax(A)])

                    ax.axis('off')

    def _plot_profiles(self, f, axdict, info_dict={}, res=10000):
        """Show images in report."""

        # meds = info_dict['medians']
        means = info_dict['means']
        stds = info_dict['stds']

        data = {d: means['data'][d] + stds['data'][d] for d in 'xyz'}
        clim_data = self._get_clim(data, q=[0,1], roundfuns=[np.floor, np.ceil])

        for dim, ax_idx in zip('xyz', [2, 1, 0]):

            ax = axdict[dim]

            props = info_dict['props']
            es = props['elsize'][props['axlab'].index(dim)]
            sh = stds['data'][dim].shape[0]
            x_max = sh * es
            if dim != 'z':
                x_max /= 1000  # from um to mm
            x = np.linspace(0, x_max, sh)

            dm = means['data'][dim]
            ci = stds['data'][dim]
            ax.plot(x, dm, color='r', linewidth=1, linestyle='-')
            ax.fill_between(x, dm, dm + ci, color='r', alpha=.1)

            dm = means['corr'][dim]
            ci = stds['corr'][dim]
            ax.plot(x, dm, color='g', linewidth=1, linestyle='-')
            ax.fill_between(x, dm, dm + ci, color='g', alpha=.1)

            ax.set_xlim([0, x_max])
            ax.set_xticks([0, x_max])

            x_fmtr = matplotlib.ticker.StrMethodFormatter("{x:.1f}")
            ax.xaxis.set_major_formatter(x_fmtr)
            if dim != 'z':
                ax.set_xlabel('{} [mm]'.format(dim), labelpad=-7)
            else:
                ax.set_xlabel(r'{} [$\mu m]$'.format(dim), labelpad=-7)

            ax.set_ylim(clim_data)
            ax.set_yticks([
                clim_data[0],
                clim_data[0] + (clim_data[1] - clim_data[0]) / 2,
                clim_data[1],
                ])

            if dim == 'z':
                ax.legend(['original', 'SD', 'corrected', 'SD'], fontsize=7,
                          loc='upper center', frameon=False, ncol=2)

    def view(self, input=[], images=[], labels=[], settings={}):

        images = images or self._images
        labels = labels or self._labels

        if not input:
            filestem = os.path.join(self.directory, self.format_())
            input = f'{filestem}_ds.h5'

        super().view(input, images, labels, settings)


def downsample_channel(inputpath, resolution_level, channel,
                       downsample_factors, ismask, outputpath):

    if not inputpath:
        return

    im = Image(inputpath, permission='r', reslev=resolution_level)
    im.load(load_data=False)
    dsfacs_rl = dict(zip(im.axlab, im.find_downsample_factors()))
    im_ch = im.extract_channel(channel)
    im.close()
    if downsample_factors is not None:
        ds_im = im_ch.downsampled(downsample_factors, ismask, outputpath)
    else:
        ds_im = im

    im_ch.close()

    return ds_im


def smooth_division(im, sigma={}, normalize_bias=False, outputpath=''):

    from skimage.filters import gaussian
    vol = im.slice_dataset()
    s = [sigma[d] for d in im.axlab if d in 'zyx']
    bias = gaussian(vol, sigma=s)
    if normalize_bias:
        from stapl3d.segmentation import segment
        bias, dr = segment.normalize_data(bias, a=1.00, b=5.00)

    mo = write_image(im, outputpath, bias)

    return mo


def attenuation_correction(im, radius=2, ref_idx=None, outputpath=''):
    from skimage.morphology import opening, disk
    selem = disk(radius)
    vol = im.slice_dataset()
    vol_open = np.zeros_like(vol)
    for i, slc in enumerate(vol):
        vol_open[i, :, :] = opening(slc, selem)

    # TODO: write slicewise mean and std instead of bias
    means = np.mean(vol_open, axis=(1, 2))
    stds = np.std(vol_open, axis=(1, 2))
    filepath = outputpath.replace('.h5/bias', '_att.npz')
    with open(filepath, 'wb') as f:
        np.savez(f, means=means, stds=stds)

    if ref_idx is None:
        ref_idx = np.argmax(means)
    ref_mean = means[ref_idx]
    ref_std = stds[ref_idx]

    vol_corr = np.zeros_like(vol)
    for i, slc in enumerate(vol):
        vol_corr[i, :, :] = ref_mean + ref_std * ((slc - means[i]) / stds[i])

    bias = vol / vol_corr

    mo = write_image(im, outputpath, bias)

    return mo


def attenuation_correction_apply(im, means, stds, ref_idx=None):

    vol_corr = np.zeros_like(vol)
    for i, slc in enumerate(vol):
        vol_corr[i, :, :] = ref_mean + ref_std * ((slc - means[i]) / stds[i])

    mo = write_image(im, outputpath, vol_corr)

    return mo


def calculate_bias_field(im, mask=None, n_iter=50,
                         n_fitlev=[4, 4, 4], n_cps=[5, 5, 5],
                         n_threads=1, outputpath=''):
    """Calculate the bias field."""

    # wmem images to sitk images
    dsImage = sitk.GetImageFromArray(im.ds)
    dsImage.SetSpacing(np.array(im.elsize[::-1], dtype='float'))
    dsImage = sitk.Cast(dsImage, sitk.sitkFloat32)
    if mask is not None:
        dsMask = sitk.GetImageFromArray(mask.ds[:].astype('uint8'))
        dsMask.SetSpacing(np.array(im.elsize[::-1], dtype='float'))
        dsMask = sitk.Cast(dsMask, sitk.sitkUInt8)

    # run the N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    corrector.SetDebug(True)
    corrector.SetNumberOfThreads(n_threads)  # FIXME: seems to have no effect
    corrector.SetMaximumNumberOfIterations([n_iter] * n_fitlev[0])  # FIXME
    corrector.SetNumberOfControlPoints(n_cps)
    corrector.SetWienerFilterNoise(0.01)
    corrector.SetConvergenceThreshold(0.001)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrector.SetNumberOfHistogramBins(200)
    if mask is None:
        dsOut = corrector.Execute(dsImage)
    else:
        dsOut = corrector.Execute(dsImage, dsMask)

    # get the bias field at lowres (3D)
    # TODO: itkBSplineControlPointImageFilter
    data = np.stack(sitk.GetArrayFromImage(dsImage))
    data /= np.stack(sitk.GetArrayFromImage(dsOut))
    data = np.nan_to_num(data, copy=False).astype('float32')

    mo = write_image(im, outputpath, data)

    return mo


def divide_bias_field(im, bf, outputpath=''):
    """Apply bias field correction."""

    data = np.copy(im.ds[:]).astype('float32')
    data /= bf.ds[:]
    data = np.nan_to_num(data, copy=False)

    mo = write_image(im, outputpath, data)

    return mo


def write_image(im, outputpath, data):
    props = im.get_props2()
    props['path'] = outputpath
    props['permission'] = 'r+'
    props['dtype'] = data.dtype
    mo = Image(**props)
    mo.create()
    mo.write(data)
    # if 'downsample_factors' in im.ds.attrs:
    #     mo.ds.attrs['downsample_factors'] = im.ds.attrs['downsample_factors']
    return mo


def gather_4D(inputpat, outputfile, idss=['data']):
    """Merge 3D image stacks to virtual 4D datasets."""

    inputfiles = glob(inputpat)
    inputfiles.sort()
    if not inputfiles:
        return
    if outputfile.endswith('.ims'):
        ims_linked_channels(outputfile, inputfiles, inputs['ims_ref'])
    elif '.h5' in outputfile:
        for ids in idss:
            inputpaths = [f'{inputfile}/{ids}' for inputfile in inputfiles]
            h5chs_to_virtual(inputpaths, f'{outputfile}/{ids}')


def stack_bias(inputfiles, outputfile, idss=['data', 'bias', 'corr']):
    """Merge the downsampled biasfield images to 4D datasets."""

    for ids in idss:
        images_in = [f'{filepath}/{ids}' for filepath in inputfiles]
        outputpath = f'{outputfile}/{ids}'
        mo = stack_channels(images_in, outputpath=outputpath)


def stack_channels(images_in, axis=-1, outputpath=''):
    """Write a series of 3D images to a 4D stack."""

    im = Image(images_in[0], permission='r')
    im.load(load_data=True)
    props = im.get_props2()
    props['path'] = outputpath
    props['permission'] = 'r+'
    props['slices'] = None
    im.close()

    mo = Image(**props)

    # insert channel axis
    if axis == -1:
        axis = len(im.dims)
    C = len(images_in)
    mo.shape = mo.dims
    props = {'dims': C, 'shape': C, 'elsize': 1, 'chunks': 1, 'axlab': 'c'}
    for k, v in props.items():
        val = list(mo.__getattribute__(k))
        val.insert(axis, v)
        mo.__setattr__(k, val)

    mo.axlab = ''.join(mo.axlab)
    mo.create()

    for i, image_in in enumerate(images_in):
        im = Image(image_in, permission='r')
        im.load(load_data=True)
        mo.slices[mo.axlab.index('c')] = slice(i, i + 1, 1)
        mo.write(np.expand_dims(im.slice_dataset(), axis))
        im.close()

    mo.close()

    return mo


def get_bias_field_block(bf, slices, outdims, dsfacs):
    """Retrieve and upsample the biasfield for a datablock."""

    bf.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                 for slc, ds in zip(slices, dsfacs)]
    bf_block = bf.slice_dataset().astype('float32')
    bias = resize(bf_block, outdims, preserve_range=True)

    return bias


def get_data(h5_path, ids, ch=0, dim=''):

    im = Image(f'{h5_path}/{ids}')
    im.load(load_data=False)

    if dim:
        dim_idx = im.axlab.index(dim)
        cslc = int(im.dims[dim_idx] / 2)
        im.slices[dim_idx] = slice(cslc, cslc+1, 1)

    if len(im.dims) > 3:
        im.slices[im.axlab.index('c')] = slice(ch, ch+1, 1)

    data = im.slice_dataset()

    return data


def extract_zyx_profiles(filepath, ch=0):

    vols = ('data', 'corr', 'bias')
    metrics = ('median', 'mean', 'std')

    d = {f'{metric}s': {} for metric in metrics}

    try:
        mask = ~get_data(filepath, 'mask', ch)
    except KeyError:
        data = get_data(filepath, 'data', ch)
        mask = data < threshold_otsu(data)

    for k in vols:
        data = get_data(filepath, k, ch)
        for metric in metrics:
            d[f'{metric}s'][k] = get_zyx_medians(data, mask, metric=metric)

    d['n_samples'] = {dim: np.sum(mask, axis=i) for i, dim in enumerate('zyx')}

    return d

def create_ref_ims(filepath_ims, filepath_ref, outputpath):

    outputpath = self.outputpaths['apply']['channels']
    ims_ref = self.inputpaths['apply']['ims_ref']
    filepath_ims = self.inputpaths['estimate']['data']
    if outputpath.endswith('.ims'):
        if not ims_ref:
            filepath_ref = filepath_ims.replace('.ims', '_ref.ims')
            create_ref(filepath_ims)
    return filepath_ref

if __name__ == "__main__":
    main(sys.argv[1:])
