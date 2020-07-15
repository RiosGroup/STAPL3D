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

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from tifffile import create_output
from xml.etree import cElementTree as etree

from skimage.io import imread, imsave

import czifile

from stapl3d import (
    get_n_workers,
    get_outputdir,
    prep_outputdir,
    get_params,
    get_paths,
    Image,
)

logger = logging.getLogger(__name__)


def main(argv):
    """Correct z-stack shading."""

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
    n_workers=0,
    channels=[],
    planes=[],
    metric='median',
    noise_threshold=None,
    z_range=[],
    quantile_threshold=0.8,
    polynomial_order=3,
    postfix='',
    ):
    """Correct z-stack shading."""

    step_id = 'shading'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, step_id)

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    if not subparams['channels']:
        if image_in.endswith('.czi'):
            czi = czifile.CziFile(image_in)
            subparams['channels'] = list(range(czi.shape[czi.axes.index('C')]))
        else:
            print('Sorry, only czi implemented for now...')
            return

    if not subparams['planes']:
        if image_in.endswith('.czi'):
            czi = czifile.CziFile(image_in)
            subparams['planes'] = list(range(czi.shape[czi.axes.index('Z')]))

    nested_arglist = [[
        (
            image_in,
            ch,
            pl,
            params['noise_threshold'],
            params['metric'],
            params['z_range'],
            params['quantile_threshold'],
            params['polynomial_order'],
            params['postfix'],
            outputdir,
        )
        for ch in subparams['channels']]
        for pl in subparams['planes']]

    arglist = [item for sublist in nested_arglist for item in sublist]

    n_jobs = len(subparams['channels']) * len(subparams['planes'])
    n_workers = get_n_workers(n_jobs, subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(estimate_plane, arglist)


def estimate_plane(
    image_in,
    channel,
    plane,
    noise_threshold=None,
    metric='median',
    z_range=[],
    quantile_threshold=0.8,
    polynomial_order=3,
    postfix='',
    outputdir='',
    ):
    """Estimate the x- and y-profiles for a channel in a czi file.

    # TODO: generalize to other formats than czi
    # TODO: automate threshold estimate
    # TODO: implement more metrics
    # TODO: implement logging throughout.
    # TODO: write corrected output for Fiji BigStitcher.
    # TODO: try getting median over 4D concatenation directly (forego xy)
    """

    # Prepare the output.
    step_id = 'shading'
    postfix = postfix or '_{}'.format(step_id)
    postfix = 'ch{:02d}_Z{:03d}{}'.format(channel, plane, postfix)

    outputdir = get_outputdir(image_in, '', outputdir, step_id, step_id)

    paths = get_paths(image_in, channel=channel)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)

    filestem = '{}_{}'.format(dataset, postfix)
    outputstem = os.path.join(outputdir, filestem)
    outputpat = '{}.h5/{}'.format(outputstem, '{}')

    # logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)
    report = {'parameters': locals()}

    # Get the list of subblocks for the channel.
    if image_in.endswith('.czi'):  # order of subblock_directory: CZM
        czi = czifile.CziFile(image_in)
        zstack0 = czi.subblock_directory[0]
        dtype = zstack0.dtype
        tilesize = [zstack0.shape[zstack0.axes.index('Y')],
                    zstack0.shape[zstack0.axes.index('X')]]
        n_planes = czi.shape[czi.axes.index('Z')]
        n_chs = czi.shape[czi.axes.index('C')]
        sbd_channel = [sbd for sbd in czi.subblock_directory[channel::n_chs]]
    else:
        print('Sorry, only czi implemented for now...')
        return

    # Compute median values per plane for X and Y concatenation.
    meds_X, meds_Y = [], []

    msg = 'Processing ch{:02d}:plane{:03d}'.format(channel, plane)
    # logger.info(msg)
    print(msg)

    dstack = []
    for sbd in sbd_channel[plane::n_planes]:
        dstack.append(np.squeeze(sbd.data_segment().data()))

    dstacked = np.concatenate(dstack, axis=0)
    ma_data = np.ma.masked_array(dstacked, dstacked < noise_threshold)
    meds_X = np.ma.median(ma_data, axis=0)

    dstacked = np.concatenate(dstack, axis=1)
    ma_data = np.ma.masked_array(dstacked, dstacked < noise_threshold)
    meds_Y = np.ma.median(ma_data, axis=1)

    out = [noise_threshold, metric, meds_X, meds_Y]
    with open('{}.pickle'.format(outputstem), 'wb') as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)


def postprocess(
    image_in,
    parameter_file,
    outputdir='',
    n_workers=0,
    channels=[],
    metric='median',
    z_range=[],
    quantile_threshold=0.8,
    polynomial_order=3,
    postfix='',
    ):

    step_id = 'shading_postproc'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, 'shading', 'shading')

    params = get_params(locals().copy(), parameter_file, 'shading')
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    if not subparams['channels']:
        if image_in.endswith('.czi'):
            czi = czifile.CziFile(image_in)
            subparams['channels'] = list(range(czi.shape[czi.axes.index('C')]))
        else:
            print('Sorry, only czi implemented for now...')
            return

    arglist = [
        (
            image_in,
            ch,
            params['metric'],
            params['z_range'],
            params['quantile_threshold'],
            params['polynomial_order'],
            params['postfix'],
            outputdir,
        )
        for ch in subparams['channels']]

    n_workers = get_n_workers(len(subparams['channels']), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(postprocess_channel, arglist)


def postprocess_channel(
    image_in,
    channel,
    metric='median',
    z_range=[],
    quantile_threshold=0.8,
    polynomial_order=3,
    postfix='',
    outputdir='',
    ):

    step_id = 'shading'
    postfix = postfix or '_{}'.format(step_id)
    pf = postfix
    postfix = 'ch{:02d}{}'.format(channel, postfix)

    outputdir = get_outputdir(image_in, '', outputdir, step_id, step_id)

    paths = get_paths(image_in, channel=channel)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)

    filestem = '{}_{}'.format(dataset, postfix)
    outputstem = os.path.join(outputdir, filestem)
    outputpat = '{}.h5/{}'.format(outputstem, '{}')

    # Get the list of subblocks for the channel.
    if image_in.endswith('.czi'):  # order of subblock_directory: CZM
        czi = czifile.CziFile(image_in)
        zstack0 = czi.subblock_directory[0]
        dtype = zstack0.dtype
        tilesize = [zstack0.shape[zstack0.axes.index('Y')],
                    zstack0.shape[zstack0.axes.index('X')]]
        n_planes = czi.shape[czi.axes.index('Z')]
        n_chs = czi.shape[czi.axes.index('C')]
        sbd_channel = [sbd for sbd in czi.subblock_directory[channel::n_chs]]
    else:
        print('Sorry, only czi implemented for now...')
        return

    # glob all planes of a channel
    plane_pat = '{}_ch{:02d}_Z???{}'.format(dataset, channel, pf)
    pickles = glob(os.path.join(outputdir, '{}.pickle'.format(plane_pat)))
    pickles.sort()
    meds_X, meds_Y = [], []
    for pfile in pickles:
        with open(pfile, 'rb') as f:
            out = pickle.load(f)
        meds_X.append(out[2])
        meds_Y.append(out[3])

    img_fitted = np.ones(tilesize, dtype='float')

    # Estimate the profiles from the medians.
    out = [metric, z_range, quantile_threshold, polynomial_order]
    for dim, meds in {'X': meds_X, 'Y': meds_Y}.items():

        out.append(np.stack(meds, axis=0))
        data_sel, dm, z_sel = select_z_range(out[-1], z_range, quantile_threshold)
        data_mean, data_norm, data_norm_mean = normalize_profile(data_sel)
        data_fitted, data_fitted_norm = fit_profile(data_norm_mean, polynomial_order)

        if dim == 'X':
            img_fitted *= data_fitted_norm
        elif dim == 'Y':
            img_fitted *= data_fitted_norm[:, np.newaxis]

    # Save estimated shading image.
    img = np.array(img_fitted * np.iinfo(dtype).max, dtype=dtype)
    imsave('{}.tif'.format(outputstem), img)

    # Save parameters.
    with open('{}.pickle'.format(outputstem), 'wb') as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    # Print a report page to pdf.
    generate_report(outputdir, dataset, channel=channel, ioff=True)

    # Delete plane pickles
    for pfile in pickles:
        os.remove(pfile)


def select_z_range(data, z_range=[], quant_thr=0.8):
    """Select planes of in a defined range or with the highest medians."""

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


def czi_split_zstacks(filepath, offset=0, nstacks=0, clipping_mask=False, correct=False, write_to_tif=True):
    """Split czi file in zstacks."""

    filestem = os.path.splitext(filepath)[0]
    datadir, fstem = os.path.split(filestem)

    czi = czifile.CziFile(filepath)
    zss, n = get_zstack_shape(czi)
    out = create_output(None, zss, czi.dtype)
    nstacks = nstacks or n

    props = {
        'shape': [zss[8], zss[9], zss[10], zss[6]],
        'elsize': get_elsize_zyx(czi) + [1],
        'axlab': 'zyxc',
        'dtype': czi.dtype,
        }
    props3D = {
        'shape': [zss[8], zss[9], zss[10]],
        'elsize': get_elsize_zyx(czi),
        'axlab': 'zyx',
        'dtype': czi.dtype,
        }

    for stack_idx in range(offset, offset + nstacks):

        if stack_idx == nstacks:
            return

        read_zstack(czi, zstack_idx=stack_idx, out=out)
        data = np.transpose(np.squeeze(out), axes=[1, 2, 3, 0])

        if clipping_mask:
            mask = create_clipping_mask(data)
            data = np.append(data, mask.astype(data.dtype), axis=3)

        if correct:
            shadingstem = os.path.join(datadir, 'shading', fstem)
            shading = []
            for ch in range(data.shape[3]):
                filepath_shading = '{}_ch{:02d}_shading.tif'.format(shadingstem, ch)
                shading_img = imread(filepath_shading)
                shading.append(np.tile(shading_img, (data.shape[0], 1, 1)))
            data /= np.stack(shading, axis=3)

        if write_to_tif:
            outputstem = os.path.join(datadir, 'stacks', "{}_stack{:03d}".format(fstem, stack_idx))
            stack2tifs(data, outputstem, props3D)
        else:
            outpath = "{}_stack{:03d}.h5/data".format(filestem, stack_idx)
            # TODO blockdir
            mo = Image(outpath, **props)
            mo.create()
            mo.write(data)
            mo.close()


def  create_clipping_mask(data):

    try:
        dmax = np.iinfo(data.dtype).max
    except ValueError:
        dmax = np.finfo(data.dtype).max

    mask = np.any(data==dmax, axis=3)

    return mask


def stack2tifs(data, outputstem, props):

    for ch in range(0, data.shape[3]):
        outputpath = "{}_ch{:02d}.tif".format(outputstem, ch)
        mo = Image(outputpath, **props)
        mo.create()
        mo.write(data[:, :, :, ch])
        mo.close()


def get_elsize_zyx(czi):
    """Get the zyx resolutions from the czi metadata."""

    segment = czifile.Segment(czi._fh, czi.header.metadata_position)
    data = segment.data().data()
    md = etree.fromstring(data.encode('utf-8'))

    elsize_x = float(md[0][3][0][0][0].text) * 1e6
    elsize_y = float(md[0][3][0][1][0].text) * 1e6
    elsize_z = float(md[0][3][0][2][0].text) * 1e6

    return [elsize_z, elsize_y, elsize_x]


def get_offsets(czi):
    """Get the zstack positions and shapes from the czi subblocks."""

    nchannels = czi.shape[czi.axes.index('C')]
    ntimepoints = czi.shape[czi.axes.index('T')]
    nslices = czi.shape[czi.axes.index('Z')]

    sbs = czi.filtered_subblock_directory[::nchannels * ntimepoints * nslices]

    starts, shapes = [], []
    for sb in sbs:
        starts.append(sb.start)
        shapes.append(sb.shape)

    return starts, shapes


def get_zstack_shape(czi):
    """Get the shape and amount of the zstacks."""

    nchannels = czi.shape[czi.axes.index('C')]
    ntimepoints = czi.shape[czi.axes.index('T')]
    nslices = czi.shape[czi.axes.index('Z')]
    ncols = czi.shape[czi.axes.index('Y')]
    nrows = czi.shape[czi.axes.index('X')]

    zstack_shape = list(czi.filtered_subblock_directory[0].shape)
    zstack_shape[czi.axes.index('C')] = nchannels
    zstack_shape[czi.axes.index('T')] = ntimepoints
    zstack_shape[czi.axes.index('Z')] = nslices
    #zstack_shape[czi.axes.index('Y')] = ncols
    #zstack_shape[czi.axes.index('X')] = nrows

    n = nchannels * ntimepoints * nslices
    nzstacks = len(czi.filtered_subblock_directory) // n

    return zstack_shape, nzstacks


def read_zstack(czi, zstack_idx=0, out=None):
    """Read the zstack data."""

    if out is None:
        zss = get_zstack_shape(czi)
        out = create_output(None, zstack_shape, czi.dtype)

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

    return out


def find_stack_offsets(czi, conffile=''):
    ### get offsets of the zstack in XYZ
    nchannels = czi.shape[czi.axes.index('C')]
    ntimepoints = czi.shape[czi.axes.index('T')]
    nslices = czi.shape[czi.axes.index('Z')]
    ncols = czi.shape[czi.axes.index('Y')]
    nrows = czi.shape[czi.axes.index('X')]
    # first dir of eacxh zstack: C[8]Z[84]M[286]
    stack_stride = nchannels * ntimepoints * nslices
    sbd_zstacks0 = [sbd for sbd in czi.subblock_directory[::stack_stride]]
    nstacks = len(sbd_zstacks0)
    stack_idxs = list(range(0, nstacks))
    v_offsets = np.zeros([nstacks, 4])
    c_offsets = np.zeros([nstacks, 4])
    for i, directory_entry in zip(stack_idxs, sbd_zstacks0):
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
        c_offsets[i, :] = [i, x_osc, y_osc, z_osc]
    # print(v_offsets, c_offsets)

    if conffile:
        # conffile = '{}_stitch.conf'.format(filestem)
        # entry per channelxtile
        #vo = np.repeat(v_offsets, nchannels, axis=0)
        vo = np.tile(v_offsets, [nchannels, 1])
        vo[:, 0] = list(range(0, vo.shape[0]))
        np.savetxt(conffile, vo,
                   fmt='%d;;(%10.5f, %10.5f, %10.5f)',
                   header='dim=3', comments='')

    return v_offsets, c_offsets


def plot_profiles(f, axdict, filestem, clip_threshold=0.75, res=10000, fac=0.05):

    with open('{}.pickle'.format(filestem), 'rb') as f:
        metric, z_range, quant_thr, order, data_X, data_Y = pickle.load(f)

    for dim, vals in {'X': [data_X, 'g'], 'Y': [data_Y, 'b']}.items():
        plot_profile(vals[0], z_range, quant_thr, axdict, dim, vals[1], clip_threshold, res, fac)


def plot_profile(data, z_range=[], quant_thr=0.8, axdict={}, ax='X', c='r', clip_threshold=0.75, res=10000, fac=0.05):
    """Plot graphs with profiles."""

    if not axdict:
        fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    else:
        axs = axdict[ax] + [axdict['Z']]

    data_sel, dm_Z, z_sel = select_z_range(data, z_range, quant_thr)
    data_mean, data_norm, data_norm_mean = normalize_profile(data_sel)
    data_fitted, data_fitted_norm = fit_profile(data_norm_mean)

    x = np.linspace(0, data_sel.shape[1] - 1, data_sel.shape[1])

    # Plot selected profiles.
    colors = matplotlib.cm.jet(np.linspace(0, 1, data_sel.shape[0]))
    for i in range(0, data_sel.shape[0]):
        axs[0].plot(x, data_sel[i,:], color=colors[i], linewidth=0.5)

    # Plot mean with confidence interval, normalized fit and threshold.
    dm = data_norm_mean.transpose()
    ci = 1.96 * np.std(data_norm, axis=0) / np.mean(data_norm, axis=0)
    axs[1].plot(dm,
                color='k', linewidth=1, linestyle=':')
    axs[1].fill_between(x, dm - ci, dm + ci,
                        color='b', alpha=.1)
    axs[1].plot(data_fitted_norm.transpose(),
                color=c, linewidth=1, linestyle='-')
    axs[1].plot([0, len(x)], [clip_threshold] * 2,
                color='r', linewidth=1, linestyle='--')

    # Format axes.
    y_min = np.floor(np.amin(data_sel) / res) * res
    y_max = np.ceil(np.amax(data_sel) / res) * res
    axs[0].set_ylim([y_min, y_max])
    axs[0].yaxis.set_ticks([y_min, y_min + 0.5 * y_max, y_max])
    axs[1].set_ylim([0.5, 1.5])

    # Plot Z profile and and ticks to indicate selected planes.
    axs[2].plot(dm_Z, color=c, linewidth=1, linestyle='-')

    if ax == 'X':  # plot X-selection on bottom
        y = [y_min, y_max * fac]
    elif ax == 'Y':  # plot Y-selection on top
        y = [y_min + (1 - fac) * (y_max - y_min), y_max]
    for i in z_sel:
        axs[2].plot([z_sel, z_sel], y,
                    color=c, linewidth=0.5, linestyle='-')

    # Format axes.
    axs[2].set_ylim([y_min, y_max])
    axs[2].yaxis.set_ticks([y_min, y_min + 0.5 * y_max, y_max])


def clip_colormap(threshold, clip_color, n_colors=100):
    """Create clipping colormap."""

    colors = matplotlib.cm.viridis(np.linspace(0, 1, n_colors))
    n = int(threshold * n_colors)
    for i in range(n):
        colors[i, :] = clip_color

    return matplotlib.colors.ListedColormap(colors)


def plot_images(f, ax, filestem, clip_threshold=0.75, clip_color=[1, 0, 0, 1], n_colors=100):
    """Plot graph with shading image."""

    clipped = clip_colormap(clip_threshold, clip_color, n_colors)

    img = imread('{}.tif'.format(filestem))
    img = img.transpose() / int(np.iinfo(img.dtype).max)
    im = ax.imshow(img, cmap=clipped, vmin=0, vmax=1, origin='lower')

    # PLot colorbar.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = f.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0, clip_threshold, 1])


def gen_subgrid(f, gs, fsize=7, channel=None, metric='median'):

    fdict = {'fontsize': fsize,
     'fontweight' : matplotlib.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    # TODO
    n_planes = 106
    ts = [1024, 1024]

    gs00 = gs.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1])
    gs000 = gs00[0].subgridspec(3, 2)

    ax1 = f.add_subplot(gs000[0, 0])
    title = '{} over X'.format(metric)
    ax1.set_title(title, fdict, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=fsize, direction='in')
    ax1.xaxis.set_ticks([0, ts[1] - 1])
    ax1.set_xlabel('Y [px]', fdict, labelpad=-3)

    ax3 = f.add_subplot(gs000[1, 0])
    title = '{} over Y'.format(metric)
    ax3.set_title(title, fdict, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=fsize, direction='in')
    ax3.xaxis.set_ticks([0, ts[0] - 1])
    ax3.set_xlabel('X [px]', fdict, labelpad=-3)


    ax2 = f.add_subplot(gs000[0, 1])
    title = 'normalized X profile'
    ax2.set_title(title, fdict, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=fsize, direction='in')
    ax2.xaxis.set_ticks([0, ts[1] - 1])
    ax2.set_xlabel('Y [px]', fdict, labelpad=-3)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks([0.6, 1.0, 1.4])

    ax4 = f.add_subplot(gs000[1, 1])
    title = 'normalized Y profile'
    ax4.set_title(title, fdict, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=fsize, direction='in')
    ax4.xaxis.set_ticks([0, ts[0] - 1])
    ax4.set_xlabel('X [px]', fdict, labelpad=-3)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_ticks([0.6, 1.0, 1.4])


    ax5 = f.add_subplot(gs000[2, 0])
    title = 'Z plane selections'
    ax5.set_title(title, fdict, fontweight='bold')
    ax5.tick_params(axis='both', labelsize=fsize, direction='in')
    ax5.xaxis.set_ticks([0, n_planes - 1])
    ax5.set_xlabel('Z [px]', fdict, labelpad=-3)
    # ax5.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    ax6 = f.add_subplot(gs000[2, 1])  # image
    title = '2D shading profile'.format(channel)
    ax6.set_title(title, fdict, fontweight='bold')
    ax6.set_xlabel('X', fdict, labelpad=-3)
    ax6.set_ylabel('Y', fdict, labelpad=-3)
    ax6.xaxis.set_ticks([0, ts[0] - 1])
    ax6.yaxis.set_ticks([0, ts[1] - 1])
    ax6.tick_params(axis='both', labelsize=fsize, direction='in')
    ax6.invert_yaxis()


    ax3.set_ylabel('channel {}'.format(channel), fontsize=14, fontweight='bold')
    ax3.yaxis.label.set_color('k')

    return {'X': [ax1, ax2], 'Y': [ax3, ax4], 'Z': ax5, 'I': ax6}


def old2new(datadir, dataset, metric='medians', z_range=[], quant_thr=0.8, order=3):
    """Legacy format conversion."""

    inputdir = os.path.join(datadir, 'bias')
    outputdir = os.path.join(datadir, 'shading')
    for ch in range(8):
        chstem = os.path.join(inputdir, '{}_ch{:02d}'.format(dataset, ch))
        data_X = np.load('{}_medians_axisX.npy'.format(chstem))
        data_Y = np.load('{}_medians_axisY.npy'.format(chstem))
        out = [metric, z_range, quant_thr, order, data_X, data_Y]

        os.makedirs(outputdir, exist_ok=True)
        outstem = os.path.join(outputdir, '{}_ch{:02d}'.format(dataset, ch))
        with open('{}.pickle'.format(outstem), 'wb') as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

        import shutil
        inpath = '{}_medians_bias.tif'.format(chstem)
        outpath = '{}_shading.tif'.format(outstem)
        shutil.copy2(inpath, outpath)


def generate_report(outputdir, dataset, channel=None, res=10000, clip_threshold=0.75, ioff=False):

    chsize = (11.69, 8.27)  # A4 portrait
    # chsize = [6, 6]
    figtitle = 'STAPL-3D Z-stack shading report'
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
        chstem = '{}_ch{:02d}_shading'.format(filestem, ch)
        axdict = gen_subgrid(f, gs[idx], channel=ch)
        plot_images(f, axdict['I'], chstem, clip_threshold)
        plot_profiles(f, axdict, chstem, clip_threshold)

    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_shading.pdf'.format(outputstem))
    if ioff:
        plt.close(f)


if __name__ == "__main__":
    main(sys.argv[1:])
