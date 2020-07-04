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

import numpy as np

from glob import glob

from skimage.transform import resize, downscale_local_mean
from skimage.measure import block_reduce

from stapl3d import (
    get_n_workers,
    get_outputdir,
    prep_outputdir,
    get_imageprops,
    get_params,
    get_paths,
    Image,
    wmeMPI,
    )

from stapl3d.preprocessing.masking import (write_data)
from stapl3d.reporting import (
    gen_orthoplot,
    gen_orthoplot_with_profiles,
    get_centreslice,
    get_centreslices,
    get_zyx_medians,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Perform N4 bias field correction."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-i', '--image_in',
        required=True,
        help='path to image file',
        )
    parser.add_argument(
        '-p', '--parameter_file',
        required=True,
        help='path to yaml parameter file',
        )
    parser.add_argument(
        '-o', '--outputdir',
        required=False,
        help='path to output directory',
        )

    args = parser.parse_args()

    estimate(args.image_in, args.parameter_file, args.outputdir)


def estimate(
    image_in,
    parameter_file,
    outputdir='',
    channels=[],
    mask_in='',
    resolution_level=-1,
    downsample_factors={'z': 1, 'y': 1, 'x': 1, 'c': 1, 't': 1},
    n_iterations=50,
    n_fitlevels=4,
    n_bspline_cps={'z': 5, 'y': 5, 'x': 5},
    ):
    """Perform N4 bias field correction."""

    step_id = 'biasfield'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, step_id)

    params = get_params(locals(), parameter_file, step_id)

    if not params['channels']:
        props = get_imageprops(image_in)
        n_channels = props['shape'][props['axlab'].index('c')]
        params['channels'] = list(range(n_channels))

    n_workers = get_n_workers(len(params['channels']), params)

    if isinstance(params['mask_in'], bool):
        if params['mask_in']:
            filestem = os.path.splitext(get_paths(image_in)['fname'])[0]
            mpars = get_params(dict(), parameter_file, 'mask')
            maskfile = '{}{}.h5/mask'.format(filestem, mpars['postfix'])
            maskdir = get_outputdir(image_in, parameter_file, '', 'mask')
            params['mask_in'] = os.path.join(maskdir, maskfile)

    arglist = [
        (
            image_in,
            ch,
            params['submit']['tasks'],
            params['mask_in'],
            params['resolution_level'],
            [params['downsample_factors'][dim] for dim in 'zyxct'],
            params['n_iterations'],
            params['n_fitlevels'],
            [params['n_bspline_cps'][dim] for dim in 'zyx'],
            outputdir,
        )
        for ch in params['channels']]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(estimate_channel, arglist)


def estimate_channel(
    image_in,
    channel,
    n_threads=1,
    mask_in='',
    resolution_level=-1,
    downsample_factors=[1, 1, 1, 1, 1],
    n_iterations=50,
    n_fitlevels=4,
    n_bspline_cps=[5, 5, 5],
    outputdir='',
    ):
    """Estimate the x- and y-profiles for a channel in a czi file.

    """

    # Prepare the output.
    step_id = 'biasfield'
    postfix = 'ch{:02d}_{}'.format(channel, step_id)

    outputdir = prep_outputdir(outputdir, image_in, subdir=step_id)

    paths = get_paths(image_in, resolution_level, channel)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)

    filestem = '{}_{}'.format(dataset, postfix)
    outputstem = os.path.join(outputdir, filestem)
    outputpat = '{}.h5/{}'.format(outputstem, '{}')

    logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)
    report = {'parameters': locals()}


    ds_im = downsample_channel(image_in, channel, resolution_level,
                               downsample_factors, False, outputpat)

    ds_ma = downsample_channel(mask_in, channel, resolution_level,
                               downsample_factors, True, outputpat)

    ds_bf = calculate_bias_field(ds_im, ds_ma,
                                 n_iterations, n_fitlevels, n_bspline_cps,
                                 n_threads, outputpat)

    ds_cr = divide_bias_field(ds_im, ds_bf, outputpat)


    # Save medians and parameters.
    with open('{}.pickle'.format(outputstem), 'wb') as f:
        pickle.dump(report['parameters'], f, pickle.HIGHEST_PROTOCOL)

    # Print a report page to pdf.
    generate_report(outputdir, dataset, channel=channel, ioff=True)

    ds_im.close()
    ds_bf.close()
    ds_cr.close()
    if mask_in:
        ds_ma.close()


def downsample_channel(image_in, ch, resolution_level=-1,
                       dsfacs=[1, 4, 4, 1, 1], ismask=False, output=''):
    """Downsample an image."""

    ods = 'data' if not ismask else 'mask'

    # return in case no mask provided
    if not image_in:
        return None

    if resolution_level != -1 and not ismask:  # we should have an Imaris pyramid
        image_in = '{}/DataSet/ResolutionLevel {}'.format(image_in, resolution_level)

    # load data
    im = Image(image_in, permission='r')
    im.load(load_data=False)
    props = im.get_props()
    if len(im.dims) > 4:
        im.slices[im.axlab.index('t')] = slice(0, 1, 1)
        props = im.squeeze_props(props, dim=4)
    if len(im.dims) > 3:
        im.slices[im.axlab.index('c')] = slice(ch, ch+1, 1)
        props = im.squeeze_props(props, dim=3)
    data = im.slice_dataset()
    im.close()

    # downsample
    dsfac = tuple(dsfacs[:len(data.shape)])
    if not ismask:
        data = downscale_local_mean(data, dsfac).astype('float32')
    else:
        data = block_reduce(data, dsfac, np.max)

    # generate output
    props['axlab'] = 'zyx'  # FIXME: axlab returns as string-list
    props['shape'] = data.shape
    props['elsize'] = [es*ds for es, ds in zip(im.elsize[:3], dsfac)]
    props['slices'] = None
    mo = write_data(data, props, output, ods)

    return mo


def calculate_bias_field(im, mask=None,
                         n_iter=50, n_fitlev=4, n_cps=[5, 5, 5],
                         n_threads=1, output=''):
    """Calculate the bias field."""

    import SimpleITK as sitk

    ods = 'bias'

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
    corrector.SetDebug(False)
    #corrector.SetNumberOfThreads(n_threads)  # FIXME: seems to have no effect
    corrector.SetMaximumNumberOfIterations([n_iter] * n_fitlev)
    corrector.SetNumberOfControlPoints(n_cps)
    if mask is None:
        dsOut = corrector.Execute(dsImage)
    else:
        dsOut = corrector.Execute(dsImage, dsMask)

    # get the bias field at lowres (3D)
    data = np.stack(sitk.GetArrayFromImage(dsImage))
    data /= np.stack(sitk.GetArrayFromImage(dsOut))
    data = np.nan_to_num(data, copy=False).astype('float32')

    # generate output
    props = im.get_props()
    mo = write_data(data, props, output, ods)

    return mo


def divide_bias_field(im, bf, output=''):
    """Apply bias field correction."""

    ods = 'corr'

    data = np.copy(im.ds[:]).astype('float32')
    data /= bf.ds[:]
    data = np.nan_to_num(data, copy=False)

    props = im.get_props()
    mo = write_data(data, props, output, ods)

    return mo


def apply_bias_field(filestem, channels=[], outputpath=''):

    def apply_field(image_in, bias_in, outputpath):

        im = Image(image_in, permission='r+')
        im.load(load_data=True)
        bf = Image(bias_in, permission='r+')
        bf.load(load_data=True)

        corr = divide_bias_field(im, bf, outputpath)

        im.close()
        bf.close()
        corr.close()

    if channels:
        for ch in channels:
            image_in = '{}_bfc_ch{:02d}.h5/data'.format(filestem, ch)
            bias_in = '{}_bfc_ch{:02d}.h5/bias'.format(filestem, ch)
            outputpath = outputpath or '{}_bfc_ch{:02d}.h5/corr'.format(filestem, ch)
            apply_field(image_in, bias_in, outputpath)
    else:
        image_in = '{}_bfc.h5/data'.format(filestem)
        bias_in = '{}_bfc.h5/bias'.format(filestem)
        outputpath = outputpath or '{}_bfc.h5/corr'.format(filestem)
        apply_field(image_in, bias_in, outputpath)


def stack_channels(images_in, outputpath=''):

    channels =[]
    for image_in in images_in:
        im = Image(image_in, permission='r')
        im.load(load_data=True)
        channels.append(im.ds[:])
    data = np.stack(channels, axis=3)

    mo = Image(outputpath)
    mo.elsize = list(im.elsize) + [1]
    mo.axlab = im.axlab + 'c'
    mo.dims = data.shape
    mo.chunks = list(im.chunks) + [1]
    mo.dtype = im.dtype
    mo.set_slices()
    if outputpath:
        mo.create()
        mo.write(data)
        mo.close()

    return mo


def stack_bias(inputfiles, outputstem, idss=['data', 'bias', 'corr']):

    for ids in idss:
        images_in = ['{}.h5/{}'.format(filepath, ids) for filepath in inputfiles]
        outputpath = '{}.h5/{}'.format(outputstem, ids)
        outputpath_nii = '{}_{}.nii.gz'.format(outputstem, ids)
        stack_channels(images_in, outputpath)


def apply(
    image_in,
    parameter_file,
    outputdir='',
    channels=[],
    downsample_factors={'z': 1, 'y': 1, 'x': 1, 'c': 1, 't': 1},
    blocksize_xy=1280,
    ):
    """Perform N4 bias field correction."""

    step_id = 'biasfield'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, 'biasfield', 'biasfield')

    params = get_params(locals(), parameter_file, 'biasfield_apply')

    if not params['channels']:
        im = Image(image_in, permission='r')
        im.load()
        n_channels = im.dims[im.axlab.index('c')]
        params['channels'] = list(range(n_channels))
        im.close()

    n_workers = get_n_workers(len(params['channels']), params)

    try:
        subdir = dirs['datadir']['channels'] or ''
    except KeyError:
        subdir = 'channels'
    channeldir = prep_outputdir('', image_in, subdir)

    paths = get_paths(image_in)
    datadir, filename = os.path.split(paths['base'])
    filestem = os.path.splitext(filename)[0]
    dataset = filestem.split('_shading')[0]
    postfix = filestem.split(dataset)[-1]
    chstem_pat = '{}{}{}'.format(dataset, '_ch{:02d}', postfix)

    arglist = []
    for ch in params['channels']:
        chstem = chstem_pat.format(ch)
        os.path.join(channeldir, chstem)
        bias_fname = '{}{}_ch{:02d}_{}.h5/bias'.format(dataset, postfix, ch, step_id)
        arglist.append(
            (
                image_in,
                os.path.join(outputdir, bias_fname),
                os.path.join(channeldir, '{}_{}.ims'.format(chstem, step_id)),
                ch,
                [params['downsample_factors'][dim] for dim in 'zyxct'],
                params['blocksize_xy'],
            )
        )

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(apply_channel, arglist)


def apply_channel(
    image_in,
    bias_in,
    image_out,
    channel=None,
    downsample_factors=[1, 1, 1, 1, 1],
    blocksize_xy=1280,
    ):
    """Correct inhomogeneity of a channel."""

    im = Image(image_in, permission='r')
    im.load(load_data=False)

    bf = Image(bias_in, permission='r')
    bf.load(load_data=False)

    mo = Image(image_out)
    mo.load()

    if channel is not None:
        if len(im.dims) > 3:
            im.slices[3] = slice(channel, channel + 1)
        if len(bf.dims) > 3:
            bf.slices[3] = slice(channel, channel + 1)

    mpi = wmeMPI(usempi=False)
    mpi_nm = wmeMPI(usempi=False)
    if blocksize_xy:
        blocksize = [im.dims[0], blocksize_xy, blocksize_xy, 1, 1]
        blockmargin = [0, im.chunks[1], im.chunks[2], 0, 0]
    else:
        blocksize = im.dims[:3] +  [1, 1]
        blockmargin = [0] * len(im.dims)
    mpi.set_blocks(im, blocksize, blockmargin)
    mpi_nm.set_blocks(im, blocksize)
    mpi.scatter_series()

    for i in mpi.series:
        print(i)
        block = mpi.blocks[i]
        data_shape = list(im.slices2shape(block['slices']))
        block_nm = mpi_nm.blocks[i]
        it = zip(block['slices'], block_nm['slices'], blocksize, data_shape)
        data_shape = list(im.slices2shape(block_nm['slices']))
        data_slices = []
        for b_slc, n_slc, bs, ds in it:
            m_start = n_slc.start - b_slc.start
            m_stop = m_start + bs
            m_stop = min(m_stop, ds)
            data_slices.append(slice(m_start, m_stop, None))
        data_slices[3] = block['slices'][3]
        data_shape = list(im.slices2shape(data_slices))

        # get the fullres image block
        im.slices = block['slices']
        data = im.slice_dataset().astype('float')

        # get the upsampled bias field
        bias = get_bias_field_block(bf, im.slices, data.shape, downsample_factors)
        data /= bias
        data = np.nan_to_num(data, copy=False)

        mo.slices = block_nm['slices']
        mo.slices[3] = slice(0, 1, 1)
        data = data[tuple(data_slices[:3])].astype(mo.dtype)
        mo.write(data)

    im.close()
    bf.close()
    mo.close()


def get_bias_field_block(bf, slices, outdims, dsfacs):

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


def plot_params(f, axdict, info_dict={}):
    """Show images in report."""

    pars = {'downsample_factors': 'Downsample factors',
            'n_iterations': 'N iterations',
            'n_fitlevels': 'N fitlevels',
            'n_bspline_cps': 'N b-spline components'}

    cellText = []
    for par, name in pars.items():
        v = info_dict['parameters'][par]
        if not isinstance(v, list):
            v = [v]
        cellText.append([name, ', '.join(str(x) for x in v)])

    axdict['p'].table(cellText, loc='bottom')
    axdict['p'].axis('off')


def plot_profiles(f, axdict, info_dict={}, res=10000):
    """Show images in report."""

    meds = info_dict['medians']
    means = info_dict['means']
    stds = info_dict['stds']

    y_min, y_max = 0, 20000  # FIXME

    for dim, ax_idx in zip('xyz', [2, 1, 0]):

        es = info_dict['elsize'][dim]
        sh = stds['data'][dim].shape[0]

        x_max = sh * es
        if dim != 'z':
            x_max /= 1000

        x = np.linspace(0, x_max, sh)

        n_samples = sh  # FIXME: this is not correct
        dm = means['data'][dim]
        ci = stds['data'][dim]
        # ci = 1.96 * stds['data'][dim] / np.sqrt(n_samples)
        axdict[dim].plot(x, dm, color='r', linewidth=1, linestyle='-')
        axdict[dim].fill_between(x, dm, dm + ci, color='r', alpha=.1)

        dm = means['corr'][dim]
        ci = stds['corr'][dim]
        # ci = 1.96 * stds['corr'][dim] / np.sqrt(n_samples)
        axdict[dim].plot(x, dm, color='g', linewidth=1, linestyle='-')
        axdict[dim].fill_between(x, dm, dm + ci, color='g', alpha=.1)

        axdict[dim].set_ylim([y_min, y_max])
        axdict[dim].set_yticks([y_min, y_max])
        axdict[dim].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        # y_fmtr = matplotlib.ticker.ScalarFormatter()  # FIXME
        # y_fmtr.set_scientific(True)
        # axdict[dim].yaxis.set_major_formatter(y_fmtr)

        axdict[dim].set_xlim([0, x_max])
        axdict[dim].set_xticks([0, x_max])
        x_fmtr = matplotlib.ticker.StrMethodFormatter("{x:.1f}")
        axdict[dim].xaxis.set_major_formatter(x_fmtr)
        if dim != 'z':
            axdict[dim].set_xlabel('{} [mm]'.format(dim), labelpad=10)
        else:
            axdict[dim].set_xlabel(r'{} [$\mu m]$'.format(dim), labelpad=10)


def plot_images(f, axdict, info_dict={}, res=10000, add_profiles=True):
    """Show images in report."""

    def get_img(cslsc, ax_idx, dims):
        img = cslsc
        if ax_idx == 2:  # zy-image
            dims[0] = img.shape[0]
            img = img.transpose()
        if ax_idx == 0:  # yx-image
            dims[1:] = img.shape
        return img, dims

    centreslices = info_dict['centreslices']
    meds = info_dict['medians']

    y_min, y_max = 0, 0
    dims = [0, 0, 0]

    bf_dict = {'data': (0, 'r'), 'corr': (1, 'g'), 'bias': (2, 'b')}

    aspects = ['auto', 'auto', 'auto']
    for dim, aspect, ax_idx in zip('xyz', aspects, [2, 1, 0]):
        for k, v in bf_dict.items():

            img, dims = get_img(centreslices[k][dim], ax_idx, dims)
            im = axdict[k][ax_idx].imshow(img, cmap='gray', aspect=aspect)

            axdict[k][ax_idx].axis('off')
            y_min = min(y_min, np.floor(np.amin(img) / res) * res)
            y_max = max(y_max, np.ceil(np.amax(img) / res) * res)

            if add_profiles:
                ax = axdict[k][ax_idx+3]
                X = list(range(len(meds['data'][dim])))
                ls, lw = '-', 1
                if dim == 'y':
                    ax.plot(meds[k][dim], X, color=v[1], linewidth=lw, linestyle=ls)
                else:
                    ax.plot(X, meds[k][dim], color=v[1], linewidth=lw, linestyle=ls)
                ax.axis('off')

        #img, dims = get_img(centreslices['mask'][dim], ax_idx, dims)
        #axdict['bias'][ax_idx].imshow(img, cmap='inferno', vmin=0, vmax=5, aspect=aspect, alpha=0.6)

            # old = False
            # if old:
            #     if dim == 'y':
            #         if k != 'corr':
            #             ax.plot(meds['data'][dim], X, color=bfdict['data'][1], linewidth=lw, linestyle=ls)
            #         if k != 'data':
            #             ax.plot(meds['corr'][dim], X, color=bfdict['corr'][1], linewidth=lw, linestyle=ls)
            #     else:
            #         if k != 'corr':
            #             ax.plot(X, meds['data'][dim], color=bfdict['data'][1], linewidth=lw, linestyle=ls)
            #         if k != 'data':
            #             ax.plot(X, meds['corr'][dim], color=bfdict['corr'][1], linewidth=lw, linestyle=ls)
            # else:


    y_min, y_max = 0, 10000  # FIXME

    # set brightness [data and corr equal]
    for i in [0, 1, 2]:
        for k in ['data', 'corr']:
            for im in axdict[k][i].get_images():
                im.set_clim(y_min, y_max)
        for im in axdict['bias'][i].get_images():
            im.set_clim(0, 3)


def add_titles(axs, info_dict):
    """Add plot titles to upper row of plot grid."""

    try:
        params = info_dict['parameters']
    except KeyError:
        params = {'channel': '???',
                  'downsample_factors': '???',
                  'n_iterations': '???',
                  'n_fitlevels': '???',
                  'n_bspline_cps': '???',
                  }

    ch = params['channel']
    titles = []

    l1 = 'uncorrected: ch{}'.format(ch)
    # l2 = 'downsampling: {}'.format(params['downsample_factors'])
    titles.append('{} \n {}'.format(l1, l2))

    l1 = 'corrected: ch{}'.format(ch)
    # l2 = ''
    titles.append('{} \n {}'.format(l1, l2))

    l1 = 'bias field: ch{}'.format(ch)
    # l2 = 'ni = {}; nf = {}; nb = {}'.format(
    #     params['n_iterations'],
    #     params['n_fitlevels'],
    #     params['n_bspline_cps'],
    #     )
    titles.append('{} \n {}'.format(l1, l2))

    # l1 = 'profiles: ch{}'.format(ch)
    # l2 = ''
    # titles.append('{} \n {}'.format(l1, l2))

    for j, title in enumerate(titles):
        axs[j][0].set_title(title)


def gen_subgrid_with_profiles(f, gs, fsize=7, channel=None, metric='median'):
    """3rows-2 columns: 3 image-triplet left, three plots right"""

    fdict = {'fontsize': fsize,
     'fontweight' : matplotlib.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])

    gs00 = gs0[0].subgridspec(10, 1)
    gs01 = gs0[1].subgridspec(3, 2)

    axdict = {k: gen_orthoplot_with_profiles(f, gs01[i, 0])
              for i, k in enumerate(['data', 'corr', 'bias'])}

    axdict['p'] = f.add_subplot(gs00[0, 0])
    axdict['p'].set_title('parameters', fdict, fontweight='bold')
    axdict['p'].tick_params(axis='both', labelsize=fsize, direction='in')
    # axdict['p'].imshow(np.zeros([100, 100]))  # test dummy for constrained layout

    axdict['z'] = f.add_subplot(gs01[0, 1])
    axdict['z'].set_title('Z profiles', fdict, fontweight='bold', loc='right')
    axdict['z'].tick_params(axis='both', labelsize=fsize, direction='in')

    axdict['y'] = f.add_subplot(gs01[1, 1])
    axdict['y'].set_title('Y profiles', fdict, fontweight='bold', loc='right')
    axdict['y'].tick_params(axis='both', labelsize=fsize, direction='in')

    axdict['x'] = f.add_subplot(gs01[2, 1])
    axdict['x'].set_title('X profiles', fdict, fontweight='bold', loc='right')
    axdict['x'].tick_params(axis='both', labelsize=fsize, direction='in')

    return axdict


def gen_subgrid(f, gs, fsize=7, channel=None, metric='median'):
    """3rows-2 columns: 3 image-triplet left, three plots right"""

    fdict = {'fontsize': fsize,
     'fontweight' : matplotlib.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    gs00 = gs.subgridspec(3, 2)

    axdict = {k: gen_orthoplot(f, gs00[i, 0])
              for i, k in enumerate(['data', 'corr', 'bias'])}

    ax1 = f.add_subplot(gs00[0, 1])
    title = 'Z'
    ax1.set_title(title, fdict, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=fsize, direction='in')
    ax2 = f.add_subplot(gs00[1, 1])
    title = 'Y'
    ax2.set_title(title, fdict, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=fsize, direction='in')
    ax3 = f.add_subplot(gs00[2, 1])
    title = 'X'
    ax3.set_title(title, fdict, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=fsize, direction='in')

    axdict['z'] = ax1
    axdict['y'] = ax2
    axdict['x'] = ax3

    return axdict


def get_info_dict(image_in, info_dict={}, report_type='bfc', channel=0):

    im = Image(image_in)
    im.load(load_data=False)
    info_dict['elsize'] = {dim: im.elsize[i] for i, dim in enumerate(im.axlab)}
    info_dict['paths'] = im.split_path()  # FIXME: out_base
    info_dict['paths']['out_base'] = info_dict['paths']['base']
    im.close()

    ppath = '{}.pickle'.format(info_dict['paths']['base'])
    with open(ppath, 'rb') as f:
        info_dict['parameters'] = pickle.load(f)
    info_dict['centreslices'] = get_centreslices(info_dict)

    # info_dict['medians'] = get_medians(info_dict)
    info_dict['medians'], info_dict['means'], info_dict['stds'], info_dict['n_samples'] = get_means_and_stds(info_dict, ch=channel, thr=1000)

    return info_dict


def generate_report(outputdir, dataset, channel=None, ioff=False):
    """Generate a QC report of the bias field correction process."""

    # chsize = [12, 7]  # FIXME:affect aspect of images
    chsize = (11.69, 8.27)  # A4 portrait
    figtitle = 'STAPL-3D bias field correction report'
    filestem = os.path.join(outputdir, dataset)
    outputstem = filestem
    if channel is None:  # plot all channels on a single page.
        figtitle = '{}: {} \n'.format(figtitle, dataset)
        n_channels = 8 # TODO: get n_channels from yml or glob -- or make channel argument a list of channels
        subplots = [2, int(n_channels / 2)]
        channels = list(range(n_channels))
    else:  # plot a single channel
        figtitle = '{} \n {}: channel {:02d} \n'.format(figtitle, dataset, channel)
        n_channels = 1
        subplots = [1, 1]
        channels = [channel]
        outputstem = '{}_ch{:02d}'.format(outputstem, channel)

    figsize = (chsize[1]*subplots[1], chsize[0]*subplots[0])
    f = plt.figure(figsize=figsize, constrained_layout=False)
    gs = gridspec.GridSpec(subplots[0], subplots[1], figure=f)

    for idx, ch in enumerate(channels):
        chstem = '{}_ch{:02d}'.format(filestem, ch)
        # axdict = gen_subgrid(f, gs[idx], channel=ch)
        axdict = gen_subgrid_with_profiles(f, gs[idx], fsize=10, channel=ch)
        # image_in = '{}_bfc.h5/data'.format(filestem)  # TODO use chstem, get it from the non-merged bfc's
        image_in = '{}_biasfield.h5/data'.format(chstem)
        info_dict = get_info_dict(image_in, channel=ch)
        info_dict['parameters']['channel'] = ch
        plot_params(f, axdict, info_dict)
        plot_images(f, axdict, info_dict)
        plot_profiles(f, axdict, info_dict)
        info_dict.clear()

    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_biasfield.pdf'.format(outputstem))
    if ioff:
        plt.close(f)


if __name__ == "__main__":
    main(sys.argv[1:])
