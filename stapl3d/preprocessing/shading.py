#!/usr/bin/env python

"""Correct z-stack shading.

"""

import sys
import argparse

import os
import numpy as np
import pickle
import czifile

from types import SimpleNamespace
import multiprocessing

from skimage.io import imread, imsave

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import logging

from stapl3d import get_params
from stapl3d.reporting import generate_report_page
logger = logging.getLogger(__name__)


def main(argv):
    """Correct z-stack shading."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        'inputfile',
        help='path to czi file',
        )
    parser.add_argument(
        'parameter_file',
        help='path to yaml parameter file',
        )
    parser.add_argument(
        '-o', '--outputdir',
        help='output directory',
        )

    args = parser.parse_args()

    shading_correction(
        args.inputfile,
        args.parameter_file,
        args.outputdir,
        )


def shading_correction(
    filepath,
    parameter_file='',
    outputdir='',
    channels=[],
    metric='median',
    noise_threshold=None,
    z_range=[],
    quantile_threshold=0.8,
    polynomial_order=3,
    ):
    """Correct z-stack shading."""

    params = get_params(locals(), parameter_file, 'shading_correction')

    if 'channels' not in params.keys():
        czi = czifile.CziFile(filepath)
        params['channels'] = list(range(czi.shape[czi.axes.index('C')]))

    n_workers = get_n_workers(len(params['channels']), params)

    arglist = [
        (
            filepath,
            ch,
            params['noise_threshold'],
            params['metric'],
            params['z_range'],
            params['quantile_threshold'],
            params['polynomial_order'],
            outputdir,
        )
        for ch in params['channels']]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(estimate_channel_shading, arglist)


def estimate_channel_shading(
    filepath,
    channel,
    noise_threshold=None,
    metric='median',
    z_range=[],
    quantile_threshold=0.8,
    polynomial_order=3,
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

    # Get the list of subblocks for the channel.
    if filepath.endswith('.czi'):  # order of subblock_directory: CZM
        czi = czifile.CziFile(filepath)
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

    # Prepare the output directory.
    datadir, filename = os.path.split(filepath)
    dataset, ext = os.path.splitext(filename)
    outputdir = outputdir or os.path.join(datadir, 'shading')
    os.makedirs(outputdir, exist_ok=True)
    fstem = '{}_ch{:02d}_{}'.format(dataset, channel, metric)
    filepath = os.path.join(outputdir, '{}.log'.format(fstem))
    logging.basicConfig(filename=filepath, level=logging.INFO)

    # Compute median values per plane for X and Y concatenation.
    meds_X, meds_Y = [], []
    img_fitted = np.ones(tilesize, dtype='float')
    for plane in range(n_planes):
        msg = 'Processing ch{:02d}:plane{:03d}'.format(channel, plane)
        logger.info(msg)
        #print(msg)

        dstack = []
        for sbd in sbd_channel[plane::n_planes]:
            dstack.append(np.squeeze(sbd.data_segment().data()))

        for ax, meds in {0: meds_X, 1: meds_Y}.items():
            dstacked = np.concatenate(dstack, axis=ax)
            ma_data = np.ma.masked_array(dstacked, dstacked < noise_threshold)
            meds.append(np.ma.median(ma_data, axis=ax))

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

    # Save medians and parameters.
    fstem = '{}_ch{:02d}'.format(dataset, channel)
    filepath = os.path.join(outputdir, '{}_shading.pickle'.format(fstem))
    with open(filepath, 'wb') as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    # Save estimated shading image.
    img = np.array(img_fitted * np.iinfo(dtype).max, dtype=dtype)
    filepath = os.path.join(outputdir, '{}_shading.tif'.format(fstem))
    imsave(filepath, img)

    # Print a report page to pdf.
    generate_report(outputdir, dataset, channel=channel)
    #generate_report_page(outputdir, dataset, 'shading', channel=channel)


def get_n_workers(n_workers, params):
    """Determine the number of workers."""

    n_workers = min(n_workers, multiprocessing.cpu_count())

    try:
        n_workers = min(n_workers, params['n_workers'])
    except:
        pass

    return n_workers


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


def plot_profiles(f, axdict, filestem, clip_threshold=0.75, res=10000, fac=0.05):

    with open('{}_shading.pickle'.format(filestem), 'rb') as f:
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

    img = imread('{}_shading.tif'.format(filestem))
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
        chstem = '{}_ch{:02d}'.format(filestem, ch)
        axdict = gen_subgrid(f, gs[idx], channel=ch)
        plot_images(f, axdict['I'], chstem, clip_threshold)
        plot_profiles(f, axdict, chstem, clip_threshold)

    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_shading-report.pdf'.format(outputstem))
    if ioff:
        plt.close(f)


if __name__ == "__main__":
    main(sys.argv[1:])
