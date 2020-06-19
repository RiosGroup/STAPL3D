#!/usr/bin/env python

"""Generate a mask that covers the tissue.

"""

import sys
import argparse

import os
import numpy as np
import pickle

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import distance_transform_edt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stapl3d import get_image, Image, MaskImage
#from wmem.prob2mask import prob2mask  # FIXME: get from wmem

from stapl3d.reporting import (
    gen_orthoplot,
    load_parameters,
    get_paths,
    get_centreslice,
    get_centreslices,
    )


def main(argv):
    """Generate a mask that covers the tissue."""

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
        '-r', '--resolution_level',
        type=int,
        default=4,
        help='the resolution level in the Imaris image pyramid',
        )
    parser.add_argument(
        '-s', '--sigma',
        type=float,
        default=48.0,
        help='std of the in-plane smoothing kernel',
        )
    parser.add_argument(
        '-m', '--use_median_thresholds',
        action='store_true',
        help='use the slicewise median as slicewise thresholds'
        )
    parser.add_argument(
        '-a', '--abs_threshold',
        type=int,
        default=1000,
        help='use this threshold (or as minimum if using median thresholds)',
        )
    parser.add_argument(
        '-o', '--outputstem',
        help='template for output',
        )
    parser.add_argument(
        '-S', '--save_steps',
        action='store_true',
        help='save intermediate results'
        )

    args = parser.parse_args()

    generate_dataset_mask(
        args.image_in,
        args.resolution_level,
        args.sigma,
        args.use_median_thresholds,
        args.abs_threshold,
        args.outputstem,
        args.save_steps,
        )


def generate_dataset_mask(
        image_in, resolution_level=4,
        sigma=48.0, use_median_thresholds=False, abs_threshold=1000,
        outputstem='', save_steps=False,
        ):
    """Generate a mask that covers the tissue."""

    step = 'mask'
    paths = get_paths(image_in, resolution_level, 0, outputstem, step, save_steps)
    report = {
        'parameters': locals(),
        'paths': paths,
        'medians': {},
        'centreslices': {}
        }

    im_data, report = extract_resolution_level(image_in, resolution_level, paths['steps'], report)

    im_mean, report, slc_thrs = extract_mean(im_data, 'c', True, paths['steps'], report)

    im_smooth, report = extract_smooth(im_mean, sigma, True, paths['steps'], report)

    im_thr, report = extract_masks(im_smooth, abs_threshold, paths['steps'], report)

    if use_median_thresholds:
        slc_thrs = [max(abs_threshold, m) for m in slc_thrs]
    else:
        n_planes = im_smooth.dims[im_smooth.axlab.index('z')]
        slc_thrs = [abs_threshold for _ in range(0, n_planes)]
    report['parameters']['slicewise_thresholds'] = slc_thrs

    im_mask, report = postproc_masks(im_smooth, slc_thrs, True, paths['main'], report)

    im_edt, report = calculate_distance_to_edge(im_mask, paths['steps'], report)

    with open(paths['params'], 'wb') as f:
        pickle.dump(report['parameters'], f)

    generate_report(image_in, report)

    im_data.close()
    im_mean.close()
    im_smooth.close()
    im_mask.close()
    im_edt.close()
    for im in im_thr:
        im.close()

    return im_mask


def extract_resolution_level(image_in, resolution_level, output='', report={}):
    """Extract data from a 5D Imaris image pyramid and create 4D image."""

    if resolution_level != -1:
        image_in = '{}/DataSet/ResolutionLevel {}'.format(image_in, resolution_level)

    mo = Image(image_in, permission='r')
    mo.load(load_data=False)

    report['parameters']['n_channels'] = mo.dims[mo.axlab.index('c')]

    return mo, report


def extract_mean(im, dim='c', keep_dtype=True, output='', report={}):
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
        nslsc = 8
    slc_thrs = []
    for zstart in range(0, im.dims[zdim], nslcs):
        zstop = min(im.dims[zdim], zstart + nslcs)
        im.slices[zdim] = mo.slices[zdim] = slice(zstart, zstop, None)
        data_mean = np.mean(im.slice_dataset(), axis=im.axlab.index(dim))
        if keep_dtype:
            data_mean = data_mean.astype(im.dtype)
        mo.write(data_mean)
        slc_thrs += list(np.median(np.reshape(data_mean, [data_mean.shape[0], -1]), axis=1))

    mo.slices = None
    mo.set_slices()

    im.close()

    c_slcs = {dim: get_centreslice(mo, '', dim) for dim in 'zyx'}
    report['centreslices'][ods] = c_slcs

    return mo, report, slc_thrs


def extract_smooth(im, sigma=48.0, keep_dtype=True, output='', report={}):
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

    c_slcs = {dim: get_centreslice(mo, '', dim) for dim in 'zyx'}
    report['centreslices'][ods] = c_slcs

    return mo, report


def extract_masks(im, abs_threshold=1000, output='', report={}):
    """Generate a series masks at different thresholds."""

    thresholds = [500, 1000, 2000, 3000, 4000, 5000]
    if abs_threshold not in thresholds:
        thresholds.append(abs_threshold)
        thresholds.sort()

    report['parameters']['simple_thresholds'] = thresholds

    mos = []
    c_slcs = {}
    for thr in thresholds:

        ods = 'mask_thr{:05d}'.format(thr)

        outputpath = ''
        if output:
            outputpath = output.format(ods)

        mo = prob2mask(im, lower_threshold=thr, upper_threshold=np.inf,
                       outputpath=outputpath)

        mo.load(load_data=False)
        c_slcs = {dim: get_centreslice(mo, '', dim) for dim in 'zyx'}
        report['centreslices'][ods] = c_slcs

        mos.append(mo)

    return mos, report


def postproc_masks(im, thrs=[], fill_holes=True, output='', report={}):
    """Apply slicewise thresholds to data and fill holes.

    NOTE: zyx assumed
    """

    ods = 'mask_thr{:05d}'.format(0)

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

    c_slcs = {dim: get_centreslice(mo, '', dim) for dim in 'zyx'}
    report['centreslices'][ods] = c_slcs

    return mo, report


def calculate_distance_to_edge(im, output='', report={}):
    """"Calculate the euclidian distance transform of the mask."""

    ods = 'mask_thr{:05d}_edt'.format(0)

    im.load(load_data=False)
    elsize = np.absolute(im.elsize)
    dt = np.zeros(im.ds.shape, dtype='float')
    for i, slc in enumerate(im.ds[:]):
        dt[i, :, :] = distance_transform_edt(slc, sampling=elsize[1:])

    props = im.get_props()
    mo = write_data(dt, props, output, ods)

    c_slcs = {dim: get_centreslice(mo, '', dim) for dim in 'zyx'}
    report['centreslices'][ods] = c_slcs

    return mo, report



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


def plot_images(axs, info_dict):
    """Show images in report."""

    centreslices = info_dict['centreslices']
    vmax = info_dict['plotinfo']['vmax']

    aspects = ['equal', 'auto', 'auto']
    for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

        axs[0][i].imshow(centreslices['mean'][dim], cmap='gray',
                         vmin=0, vmax=vmax, aspect=aspect)
        axs[1][i].imshow(centreslices['smooth'][dim], cmap='gray',
                         vmin=0, vmax=vmax, aspect=aspect)

        data = np.zeros_like(centreslices['mean'][dim], dtype='uint8')
        for ids in centreslices.keys():
            if ids.startswith('mask_thr') and 'edt' not in ids:
                data += centreslices[ids][dim]
        axs[2][i].imshow(data, cmap='inferno',
                         vmin=0, vmax=5, aspect=aspect)

        axs[3][i].imshow(centreslices['mean'][dim], cmap='gray',
                         vmin=0, vmax=vmax, aspect=aspect)
        axs[3][i].imshow(centreslices['mask_thr00000'][dim], cmap='inferno',
                         vmin=0, vmax=5, aspect=aspect, alpha=0.6)

        for a in axs:
            a[i].axis('off')


def add_titles(axs, info_dict):
    """Add plot titles to upper row of plot grid."""

    try:
        params = info_dict['parameters']
    except KeyError:
        params = {
            'n_channels': '???',
            'sigma': '???',
            'simple_thresholds': '???',
            'use_median_thresholds': '???',
            'abs_threshold': '???',
            }

    titles = []

    l1 = '{}-channel mean'.format(params['n_channels'])
    l2 = ''
    titles.append('{} \n {}'.format(l1, l2))

    l1 = 'smoothed (in-plane)'
    l2 = 'sigma = {}'.format(params['sigma'])
    titles.append('{} \n {}'.format(l1, l2))

    l1 = 'some masks'
    l2 = 'thrs = {}'.format(params['simple_thresholds'])
    titles.append('{} \n {}'.format(l1, l2))

    l1 = 'final mask'
    l0 = 'median' if params['use_median_thresholds'] else 'absolute'
    l2 = 'using {} threshold: thr = {}'.format(l0, params['abs_threshold'])
    titles.append('{} \n {}'.format(l1, l2))

    for j, title in enumerate(titles):
        axs[j][0].set_title(title)


def generate_report(image_in, info_dict={}, ioff=True):
    """Generate a QC report of the mask creation process."""

    report_type = 'mask'

    # Turn interactive plotting on/off.
    if ioff:
        plt.ioff()
    else:
        plt.ion()

    # Get paths from image if info_dict not provided.
    if not info_dict:
        im = Image(image_in)
        info_dict['paths'] = im.split_path()
        im.close()
        info_dict['parameters'] = get_parameters(info_dict, report_type)
        info_dict['centreslices'] = get_centreslices(info_dict)
        # info_dict['medians'] = get_medians(info_dict)

    # Create the axes.
    figsize = (18, 9)
    gridsize = (1, 4)
    f = plt.figure(figsize=figsize, constrained_layout=False)
    gs0 = gridspec.GridSpec(gridsize[0], gridsize[1], figure=f)
    axs = [gen_orthoplot(f, gs0[0, i]) for i in range(0, 4)]

    # Plot the images and graphs.
    info_dict['plotinfo'] = {'vmax': 10000}
    plot_images(axs, info_dict)

    # Add annotations and save as pdf.
    header = 'mLSR-3D Quality Control'
    figtitle = '{}: {} \n {}'.format(
        header,
        report_type,
        info_dict['paths']['fname']
        )
    figpath = '{}_{}-report.pdf'.format(
        info_dict['paths']['base'],
        report_type
        )
    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    add_titles(axs, info_dict)
    f.savefig(figpath)


if __name__ == "__main__":
    main(sys.argv[1:])
