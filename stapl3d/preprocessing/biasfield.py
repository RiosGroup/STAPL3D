#!/usr/bin/env python

"""Perform N4 bias field correction.

    # TODO: biasfield report is poorly scaled for many datasets => implement autoscaling
    # TODO: use mask in plotted bias field image if it was used; make sure it is used for calculating medians/means
    # from ruamel import yaml

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

from stapl3d import parse_args, Stapl3r, Image, format_, wmeMPI
from stapl3d import get_paths, get_imageprops  # TODO: into Image/Stapl3r
from stapl3d import imarisfiles
from stapl3d.preprocessing.shading import get_image_info
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

    homogenizer = Homogenizer(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        homogenizer._fun_selector[step]()


class Homogenizer(Stapl3r):
    """Perform N4 bias field correction."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Homogenizer, self).__init__(
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
                'ppar': ('resolution_level', 'downsample_factors',
                         'n_iterations', 'n_fitlevels', 'n_bspline_cps'),
                'spar': ('_n_workers', 'channels', 'n_threads'),
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
            'channels': [],
            'resolution_level': -1,
            'target_yx': 20,
            'downsample_factors': {},
            '_downsample_factors_reslev': {},
            'tasks': 1,
            'n_iterations': 50,
            'n_fitlevels': 4,
            'n_bspline_cps': {'z': 5, 'y': 5, 'x': 5},
            'inputstem': True,
            'blocksize_xy': 1280,
            'n_threads': 1,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

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

        # FIXME: moduledir (=step_id?) can vary
        prev_path = {
            'moduledir': 'mask', 'module_id': 'mask',
            'step_id': 'mask', 'step': 'estimate',
            'ioitem': 'outputs', 'output': 'mask',
            }
        maskpath = self._get_inpath(prev_path)
        if maskpath == 'default':
            maskpath = self._build_path(
                moduledir=prev_path['moduledir'],
                prefixes=[self.prefix, prev_path['module_id']],
                ext='h5/mask',
                )

        vols = ['data', 'mask', 'bias', 'corr']
        stem = self._build_path()
        cpat = self._build_path(suffixes=[{'c': 'p'}])
        cmat = self._build_path(suffixes=[{'c': '?'}])

        self._paths = {
            'estimate': {
                'inputs': {
                    'data': datapath,
                    'mask': maskpath,
                    },
                'outputs': {
                    **{ods: f'{cpat}_ds.h5/{ods}' for ods in vols},
                    **{'parameters': f'{cpat}_ds'},
                    **{'report': f'{cpat}_ds.pdf'},
                    },
                },
            'apply': {
                'inputs': {
                    # 'cpat': "self.inputpaths['estimate']['data']",
                    'data': datapath,
                    'bias': f'{cpat}_ds.h5/bias',
                    },
                'outputs': {
                    'channels': f'{cpat}.h5/data',
                    },
            },
            'postprocess': {
                'inputs': {
                    'channels': f'{cmat}.h5',
                    'channels_ds': f'{cmat}_ds.h5',
                    'report': f'{cmat}_ds.pdf',
                    },
                'outputs': {
                    'aggregate': f'{stem}.h5',
                    'aggregate_ds': f'{stem}_ds.h5',
                    'report': f'{stem}.pdf',
                    },
                },
        }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

        self.set_downsample_factors(self.inputpaths['estimate']['data'])

    def estimate(self, **kwargs):
        """Perform N4 bias field correction.

        channels=[],
        resolution_level=-1,
        mask_in='',
        downsample_factors=[],
        n_iterations=50,
        n_fitlevels=4,
        n_bspline_cps={'z': 5, 'y': 5, 'x': 5},
        """

        arglist = self._prep_step('estimate', kwargs)
        # NOTE: ITK is already multithreaded => n_workers = 1
        self.n_threads = min(self.tasks, multiprocessing.cpu_count())
        self._n_workers = 1
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_channel, arglist)

    def _estimate_channel(self, channel):
        """Estimate the x- and y-profiles for a channel in a czi file.

        channel
        inputpath
        - 3D/4D
        - direct/derived[auto]/derived[specified]/image_in/...
        [mask_in]
        [downsample_factors]
        """

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
        ds_bf = calculate_bias_field(
            ds_im, ds_ma,
            self.n_iterations, self.n_fitlevels,
            [self.n_bspline_cps[dim] for dim in ds_im.axlab],
            self.n_threads,
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
        self.report(outputs['report'],
                    channel=channel,
                    inputs=inputs, outputs=outputs)

    def set_downsample_factors(self, inputpath, resolution_level=-1):

        self.set_resolution_level(inputpath)

        dsfacs = imarisfiles.find_downsample_factors(inputpath, 0, self.resolution_level)
        self._downsample_factors_reslev = {dim: int(d) for dim, d in zip('zyxct', list(dsfacs) + [1, 1])}

        if self.downsample_factors:
            return

        im = Image(inputpath, permission='r', reslev=self.resolution_level)
        im.load(load_data=False)
        im.close()
        target = {'z': im.elsize[im.axlab.index('z')], 'y': self.target_yx, 'x': self.target_yx}
        dsfacs = [target[dim] / im.elsize[im.axlab.index(dim)] for dim in 'zyx']
        dsfacs = [np.round(dsfac).astype('int') for dsfac in dsfacs]
        dsfacs[1] = dsfacs[2] = min(dsfacs[1], dsfacs[2])

        self.downsample_factors = {dim: int(d) for dim, d in zip('zyxct', dsfacs + [1, 1])}

    def set_resolution_level(self, inputpath):

        if ('.ims' in inputpath or '.bdv' in inputpath) and self.resolution_level < 0:
            self.resolution_level = imarisfiles.find_resolution_level(inputpath)

    def postprocess(self, **kwargs):
        """Merge bias field estimation files.

        kwargs:
        """

        self._prep_step('postprocess', kwargs)

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        """
        # TODO: skip if inputfiles not found, throwing warning
        inputfiles = glob(f'{inpath}.h5')
        inputfiles.sort()
        self.stack_bias(inputfiles, f'{outpath}.h5')
        for filepath in inputfiles:
            os.remove(filepath)
        """
        imarisfiles.h5chs_to_virtual(outputs['aggregate'], inputs['channels'], ids='data')
        for ids in ['data', 'bias', 'corr']:
            imarisfiles.h5chs_to_virtual(outputs['aggregate_ds'], inputs['channels_ds'], ids=ids)

        pdfs = glob(inputs['report'])
        pdfs.sort()
        self._merge_reports(pdfs, outputs['report'])

        # # Replace fields.  # FIXME: assumed all the same: 4D inputfile
        # ymls = glob(f'{inpath}.yml')
        # ymls.sort()
        # bname = self.format_([self.prefix, self._module_id, 'estimate'])
        # ymlstem = os.path.join(self.datadir, self.directory, bname)
        # steps = {
        #     'estimate': [
        #         ['biasfield', 'estimate', 'files', 'inputpath'],
        #         ['biasfield', 'estimate', 'files', 'resolution_level'],
        #         ['biasfield', 'estimate', 'files', 'mask_in'],
        #     ],
        # }
        # for step, trees in steps.items():
        #     # NOTE: will replace to the values from ymls[-1]
        #     # FIXME: may aggregate submit:channels here
        #     self._merge_parameters(ymls, trees, ymlstem, aggregate=False)
        # for yml in ymls:
        #     os.remove(yml)

    def stack_bias(self, inputfiles, outputfile, idss=['data', 'bias', 'corr']):
        """Merge the downsampled biasfield images to 4D datasets."""

        for ids in idss:
            images_in = [f'{filepath}/{ids}' for filepath in inputfiles]
            outputpath = f'{outputfile}/{ids}'
            mo = stack_channels(images_in, outputpath=outputpath)

    def apply(self, **kwargs):
        """Apply N4 bias field correction.

        channels=[],
        blocksize_xy=1280,
        """

        arglist = self._prep_step('apply', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._apply_channel, arglist)

    def _apply_channel(self, channel=None):
        """Correct inhomogeneity of a channel."""

        inputs = self._prep_paths(self.inputs, reps={'c': channel})
        outputs = self._prep_paths(self.outputs, reps={'c': channel})

        self.set_downsample_factors(inputs['data'])

        # Create the image objects.
        in_place = inputs['data'] == outputs['channels']
        im = Image(inputs['data'], permission='r+' if in_place else 'r')
        im.load(load_data=False)
        bf = Image(inputs['bias'], permission='r')
        bf.load(load_data=False)

        # Create the output image.
        # if self.image_ref:
        #     shutil.copy2(self.image_ref, self.outputpath)
        #     mo = Image(self.outputpath)
        #     mo.load()
        #     # TODO: write to bdv pyramid
        # else:
        props = im.get_props()
        if len(im.dims) > 4:
            props = im.squeeze_props(props, dim=4)
        if len(im.dims) > 3:
            props = im.squeeze_props(props, dim=3)
        mo = Image(outputs['channels'], **props)
        mo.create()

        # Get the downsampling between full and bias images.
        p = self._load_dumped_step(self.directory, self._module_id, 'estimate')
        self.set_downsample_factors(p['inputs']['data'], p['resolution_level'])

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
        blocksize = im.dims
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

            data_shape = list(im.slices2shape(block_nm['slices']))  # ??? this does nothing ???

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
            bias = get_bias_field_block(bf, im.slices, data.shape, downsample_factors)
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
            axdict[k] = gen_orthoplot_with_colorbar(f, gs01[vd['row'], 0], idx=0, add_profile_insets=True)
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
                ax.legend(['original', 'corrected'], fontsize=7,
                          loc='lower center', frameon=False)

    def view_with_napari(self, filepath='', idss=['data', 'bias', 'corr'], ldss=[]):

        if not filepath:
            filestem = os.path.join(self.directory, self.format_())
            filepath = f'{filestem}_ds.h5'

        super().view_with_napari(filepath, idss, ldss=[])


def downsample_channel(inputpath, resolution_level, channel, downsample_factors, ismask, outputpath):

    if not inputpath:
        return

    im = Image(inputpath, permission='r', reslev=resolution_level)
    im.load(load_data=False)
    dsfacs_rl = dict(zip(im.axlab, im.find_downsample_factors()))
    im_ch = im.extract_channel(channel)
    im.close()
    ds_im = im_ch.downsampled(downsample_factors, ismask, outputpath)

    # downsample_factors = downsample_factors.copy()
    # for dim, dsfac in dsfacs_rl.items():
    #     downsample_factors[dim] = dsfac * downsample_factors[dim]
    # downsample_factors = tuple([downsample_factors[dim] for dim in ds_im.axlab])
    # ds_im.ds.attrs['downsample_factors'] = downsample_factors

    im_ch.close()

    return ds_im


def calculate_bias_field(im, mask=None, n_iter=50, n_fitlev=4, n_cps=[5, 5, 5], n_threads=1, outputpath=''):
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
    corrector.SetMaximumNumberOfIterations([n_iter] * n_fitlev)
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


if __name__ == "__main__":
    main(sys.argv[1:])
