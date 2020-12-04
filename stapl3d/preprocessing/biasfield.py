#!/usr/bin/env python

"""Perform N4 bias field correction.

    # TODO: biasfield report is poorly scaled for many datasets => implement autoscaling

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
# from ruamel import yaml

import numpy as np

from glob import glob

from skimage.transform import resize, downscale_local_mean
from skimage.measure import block_reduce

import SimpleITK as sitk

from stapl3d import parse_args, Stapl3r, Image, format_, wmeMPI
from stapl3d.imarisfiles import find_downsample_factors, find_resolution_level
from stapl3d.preprocessing.shading import get_image_info
from stapl3d.reporting import (
    gen_orthoplot_with_profiles,
    gen_orthoplot_with_colorbar,
    get_centreslices,
    get_zyx_medians,
    merge_reports,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Perform N4 bias field correction."""

    steps = ['estimate', 'postprocess', 'apply']
    args = parse_args('biasfield', steps, *argv)

    homogenizer = Homogenizer(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        dataset=args.dataset,
        suffix=args.suffix,
        n_workers=args.n_workers,
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
            'postprocess': self.postprocess,
            'apply': self.apply,
            }

        default_attr = {
            'step_id': 'biasfield',
            'channels': [],
            'inputpath': True,
            'resolution_level': -1,
            'mask_in': False,
            'downsample_factors': {},
            '_downsample_factors_reslev': {},
            'tasks': 1,
            'n_iterations': 50,
            'n_fitlevels': 4,
            'n_bspline_cps': {'z': 5, 'y': 5, 'x': 5},
            'inputstem': True,
            'blocksize_xy': 1280,
            'bias_in': True,
            'output_format': '.h5',
            'n_threads': 1,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self.inputpath = self._get_input(self.inputpath,
            'stitching', 'postprocess', '{}.bdv', fallback=self.image_in,
            )
        self.mask_in = self._get_input(self.mask_in, 'mask', 'estimate', '{}.h5/mask')

        self._parsets = {
            'estimate': {
                'fpar': self._FPAR_NAMES + ('inputpath', 'resolution_level', 'mask_in'),
                'ppar': ('downsample_factors',
                         'n_iterations', 'n_fitlevels', 'n_bspline_cps'),
                'spar': ('n_workers', 'channels', 'n_threads'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES + ('inputstem',),
                'ppar': (),
                'spar': ('n_workers',),
                },
            'apply': {
                'fpar': self._FPAR_NAMES + ('inputstem', 'bias_in', 'resolution_level'),
                'ppar': ('downsample_factors', 'blocksize_xy', 'output_format'),
                'spar': ('n_workers', 'channels'),
                },
            }

        # TODO: merge with parsets?
        self._partable = {
            'resolution_level': 'Resolution level image pyramid',
            'downsample_factors': 'Downsample factors',
            'n_iterations': 'N iterations',
            'n_fitlevels': 'N fitlevels',
            'n_bspline_cps': 'N b-spline components',
            }

    def estimate(self, **kwargs):
        """Perform N4 bias field correction.

        channels=[],
        inputpath=True,
        resolution_level=-1,
        mask_in='',
        downsample_factors=[],
        n_iterations=50,
        n_fitlevels=4,
        n_bspline_cps={'z': 5, 'y': 5, 'x': 5},
        """

        self.set_parameters('estimate', kwargs)
        arglist = self._get_arglist(['channels'])
        self.set_n_workers(len(arglist))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        # NOTE: ITK is already multithreaded => n_workers = 1
        self.n_threads = min(self.tasks, multiprocessing.cpu_count())
        self.n_workers = 1
        with multiprocessing.Pool(processes=self.n_workers) as pool:
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

        postfix_ch = self._suffix_formats['c'].format(channel)
        basename = self.format_([self.dataset, self.suffix, postfix_ch])
        outstem = os.path.join(self.directory, basename)
        outputpat = '{}_ds.h5/{}'.format(outstem, '{}')

        # Set derived parameter defaults.
        self.inputpath = self._get_input(self.inputpath,
            'stitching', 'postprocess', '{}.bdv', fallback=self.image_in)
        self.mask_in = self._get_input(self.mask_in, 'mask', 'estimate', '{}.h5/mask')

        self.set_downsample_factors()
        self.set_directory()  # ???

        ds_im = get_downsampled_channel(
            self.inputpath, self.resolution_level, channel,
            self.downsample_factors, False, outputpat.format('data'),
            )
        # FIXME: assuming here that mask is already created from resolution_level!
        ds_ma = get_downsampled_channel(
            self.mask_in, -1, channel,
            self.downsample_factors, True, outputpat.format('mask'),
            )
        ds_bf = calculate_bias_field(
            ds_im, ds_ma,
            self.n_iterations, self.n_fitlevels,
            [self.n_bspline_cps[dim] for dim in ds_im.axlab],
            self.n_threads, outputpat.format('bias'),
            )
        ds_cr = divide_bias_field(ds_im, ds_bf, outputpat.format('corr'))

        for im in [ds_im, ds_ma, ds_bf, ds_cr]:
            try:
                im.close()
            except:
                pass

        self.dump_parameters(step=self.step, filestem=outstem)
        self.report(channel=channel)

    def set_downsample_factors(self, target_yx=20, inputpath='', resolution_level=-1):

        if not inputpath:
            inputpath = self.inputpath
        if resolution_level == -1:
            self.set_resolution_level()

        dsfacs = find_downsample_factors(inputpath, 0, resolution_level)
        self._downsample_factors_reslev = {dim: int(d) for dim, d in zip('zyxct', list(dsfacs) + [1, 1])}

        if self.downsample_factors:
            return

        im = Image(inputpath, permission='r', reslev=resolution_level)
        im.load(load_data=False)
        im.close()
        target = {'z': im.elsize[im.axlab.index('z')], 'y': target_yx, 'x': target_yx}
        dsfacs = [target[dim] / im.elsize[im.axlab.index(dim)] for dim in 'zyx']
        dsfacs = [np.round(dsfac).astype('int') for dsfac in dsfacs]
        dsfacs[1] = dsfacs[2] = min(dsfacs[1], dsfacs[2])

        self.downsample_factors = {dim: int(d) for dim, d in zip('zyxct', dsfacs + [1, 1])}

    def set_resolution_level(self):

        if self.resolution_level >= 0:
            return

        if '.ims' in self.inputpath or '.bdv' in self.inputpath:
            if self.resolution_level == -1:
                self.resolution_level = find_resolution_level(self.inputpath)

    def postprocess(self, **kwargs):
        """Merge bias field estimation files.

        kwargs:
        """

        self.set_parameters('postprocess', kwargs)

        outstem = os.path.join(self.directory, self.format_())
        self.inputstem = self._get_input(self.inputstem, 'biasfield', 'estimate', '{}', fallback=self.image_in)
        suffix_pat = self._suffix_formats['c'].format(0).replace('0', '?')
        inputpat = '{}_{}'.format(self.inputstem, suffix_pat)

        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        # TODO: skip if inputfiles not found, throwing warning
        inputfiles = glob('{}_ds.h5'.format(inputpat))
        inputfiles.sort()
        outputfile = '{}_ds.h5'.format(outstem)
        self.stack_bias(inputfiles, outputfile)
        for filepath in inputfiles:
            os.remove(filepath)

        pdfs = glob('{}.pdf'.format(inputpat))
        pdfs.sort()
        merge_reports(pdfs, '{}.pdf'.format(outstem))

        # Replace fields.  # FIXME: assumed all the same: 4D inputfile
        ymls = glob('{}.yml'.format(inputpat))
        ymls.sort()
        bname = self.format_([self.dataset, self._module_id, step])
        ymlstem = os.path.join(self.directory, bname)
        steps = {
            'estimate': [
                ['biasfield', 'estimate', 'files', 'inputpath'],
                ['biasfield', 'estimate', 'files', 'resolution_level'],
                ['biasfield', 'estimate', 'files', 'mask_in'],
            ],
        }
        for step, trees in steps.items():
            # will replace to the values from ymls[-1]
            self._merge_parameters(ymls, trees, ymlstem, aggregate=False)
        for yml in ymls:
            os.remove(yml)

    def stack_bias(self, inputfiles, outputfile, idss=['data', 'bias', 'corr']):
        """Merge the downsampled biasfield images to 4D datasets."""

        for ids in idss:
            images_in = ['{}/{}'.format(filepath, ids) for filepath in inputfiles]
            outputpath = '{}/{}'.format(outputfile, ids)
            mo = stack_channels(images_in, outputpath=outputpath)

    def apply(self, **kwargs):
        """Apply N4 bias field correction.

        channels=[],
        inputpath=True,
        bias_in=True,
        output_format='.h5',
        blocksize_xy=1280,
        """

        self.set_parameters('apply', kwargs)
        arglist = self._get_arglist(['channels'])
        self.set_n_workers(len(arglist))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._apply_channel, arglist)

    def _apply_channel(self, channel=None):
        """Correct inhomogeneity of a channel."""

        postfix_ch = self._suffix_formats['c'].format(channel)
        basename = self.format_([self.dataset, self.suffix, postfix_ch])
        outstem = os.path.join(self.directory, basename)

        # TODO: setters for these pars
        self.image_out = '{}.h5/data'.format(outstem)
        # Set derived parameter defaults.
        self.inputpath = self._get_input(self.inputpath, 'stitching', 'fuse', '{}.bdv', fallback=self.image_in)
        self.bias_in = self._get_input(self.bias_in, 'biasfield', 'postprocess', '{}_ds.h5/bias')
        self.set_downsample_factors()

        # Create the image objects.
        in_place = self.inputpath == self.image_out
        im = Image(self.inputpath, permission='r+' if in_place else 'r')
        im.load(load_data=False)
        bf = Image(self.bias_in, permission='r')
        bf.load(load_data=False)

        # Create the output image.
        # if self.image_ref:
        #     shutil.copy2(self.image_ref, self.image_out)
        #     mo = Image(self.image_out)
        #     mo.load()
        #     # TODO: write to bdv pyramid
        # else:
        props = im.get_props()
        if len(im.dims) > 4:
            props = im.squeeze_props(props, dim=4)
        if len(im.dims) > 3:
            props = im.squeeze_props(props, dim=3)
        mo = Image(self.image_out, **props)
        mo.create()

        # Get the downsampling between full and bias images.
        p = self.load_dumped_step('estimate')
        self.set_downsample_factors(inputpath=p['inputpath'],
                                    resolution_level=p['resolution_level'])

        downsample_factors = {}
        for dim, dsfac in self._downsample_factors_reslev.items():
            downsample_factors[dim] = dsfac * self.downsample_factors[dim]
        downsample_factors = tuple([downsample_factors[dim] for dim in im.axlab])
        print(downsample_factors)

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

    def _get_info_dict(self, filestem, info_dict={}, channel=None):

        from stapl3d import get_paths, get_imageprops  # TODO: into Image/Stapl3r

        info_dict['parameters'] = self.load_dumped_pars()

        filepath = '{}_ds.h5/data'.format(filestem)
        info_dict['props'] = get_imageprops(filepath)
        info_dict['paths'] = get_paths(filepath)
        info_dict['centreslices'] = get_centreslices(info_dict)

        info_dict['medians'], info_dict['means'], info_dict['stds'], info_dict['n_samples'] = get_means_and_stds(info_dict, channel, thr=1000)  # TODO thr flexible or informed

        return info_dict

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
            titles[k] = ('{} profiles'.format(k.upper()), 'rc', 0)
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
            outstem = os.path.join(self.directory, self.format_())
            filepath = '{}_ds.h5'.format(outstem)

        super().view_with_napari(filepath, idss, ldss=[])


def get_downsampled_channel(inputpath, resolution_level, channel, downsample_factors, ismask, outputpath):

    if not inputpath:
        return

    im = Image(inputpath, permission='r', reslev=resolution_level)
    im.load(load_data=False)
    im_ch = im.extract_channel(channel)
    im.close()
    ds_im = im_ch.downsampled(downsample_factors, ismask, outputpath)
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

    im = Image('{}/{}'.format(h5_path, ids))
    im.load(load_data=False)

    if dim:
        dim_idx = im.axlab.index(dim)
        cslc = int(im.dims[dim_idx] / 2)
        im.slices[dim_idx] = slice(cslc, cslc+1, 1)

    if len(im.dims) > 3:
        im.slices[im.axlab.index('c')] = slice(ch, ch+1, 1)

    data = im.slice_dataset()

    return data


def get_medians(info_dict, ch=0, thr=1000):

    h5_path = info_dict['paths']['file']
    meds = {}

    mask = get_data(h5_path, 'mask', ch)

    for k in ['data', 'corr', 'bias']:
        data = get_data(h5_path, k, ch)
        meds[k] = get_zyx_medians(data, mask)

    return meds


def get_means_and_stds(info_dict, ch=0, thr=1000):

    h5_path = info_dict['paths']['file']
    meds, means, stds, n_samples = {}, {}, {}, {}

    try:
        mask = ~get_data(h5_path, 'mask', ch)
    except KeyError:
        data = get_data(h5_path, 'data', ch)
        mask = data < thr
    n_samples = {dim: np.sum(mask, axis=i) for i, dim in enumerate('zyx')}

    for k in ['data', 'corr', 'bias']:
        data = get_data(h5_path, k, ch)
        meds[k] = get_zyx_medians(data, mask, metric='median')
        means[k] = get_zyx_medians(data, mask, metric='mean')
        stds[k] = get_zyx_medians(data, mask, metric='std')

    return meds, means, stds, n_samples


if __name__ == "__main__":
    main(sys.argv[1:])
