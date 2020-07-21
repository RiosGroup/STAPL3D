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

import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import distance_transform_edt

from stapl3d import (
    get_outputdir,
    get_imageprops,
    get_params,
    get_paths,
    Image,
    MaskImage,
    )

from stapl3d.imarisfiles import (
    find_resolution_level,
    )

from stapl3d.reporting import (
    gen_orthoplot_with_colorbar,
    get_centreslices,
    )

logger = logging.getLogger(__name__)


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
    resolution_level=-1,
    sigma=48.0,
    use_median_thresholds=False,
    median_factor=1,
    abs_threshold=1000,
    thresholds=[500, 1000, 2000, 3000, 4000, 5000],
    postfix='',
    ):
    """Generate a mask that covers the tissue."""

    step_id = 'mask'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id)

    params = get_params(locals().copy(), parameter_file, step_id)

    generate_dataset_mask(
        image_in,
        params['resolution_level'],
        params['sigma'],
        params['use_median_thresholds'],
        params['median_factor'],
        params['abs_threshold'],
        params['thresholds'],
        params['distance_to_edge'],
        params['postfix'],
        outputdir,
        )


def generate_dataset_mask(
    image_in,
    resolution_level=-1,
    sigma=48.0,
    use_median_thresholds=False,
    median_factor=1,
    abs_threshold=1000,
    thresholds=[500, 1000, 2000, 3000, 4000, 5000],
    distance_to_edge=True,
    postfix='',
    outputdir='',
    ):
    """Generate a mask that covers the tissue."""

    # Prepare the output.
    step_id = 'mask'
    postfix = postfix or '_{}'.format(step_id)

    outputdir = get_outputdir(image_in, '', outputdir, step_id, step_id)

    paths = get_paths(image_in, resolution_level)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)

    filestem = '{}{}'.format(dataset, postfix)
    outputstem = os.path.join(outputdir, filestem)
    outputpat = '{}.h5/{}'.format(outputstem, '{}')

    logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)
    report = {'parameters': locals()}

    if '.ims' in image_in and resolution_level == -1:
        resolution_level = find_resolution_level(image_in)
    im_data = extract_resolution_level(image_in, resolution_level)

    im_mean, slc_thrs = extract_mean(im_data, 'c', True, outputpat)

    im_smooth = extract_smooth(im_mean, sigma, True, outputpat)

    if abs_threshold not in thresholds:
        thresholds.append(abs_threshold)
        thresholds.sort()

    im_thr = extract_masks(im_smooth, thresholds, outputpat)

    if use_median_thresholds:
        slc_thrs = [max(abs_threshold, m / median_factor) for m in slc_thrs]
    else:
        n_planes = im_smooth.dims[im_smooth.axlab.index('z')]
        slc_thrs = [abs_threshold for _ in range(0, n_planes)]

    report['parameters']['simple_thresholds'] = thresholds
    report['parameters']['slicewise_thresholds'] = slc_thrs

    im_mask = postproc_masks(im_smooth, slc_thrs, True, outputpat)

    if distance_to_edge:
        im_edt = calculate_distance_to_edge(im_mask, outputpat)


    # Save parameters.
    with open('{}.pickle'.format(outputstem), 'wb') as f:
        pickle.dump(report['parameters'], f, pickle.HIGHEST_PROTOCOL)

    # Print a report page to pdf.
    generate_report(outputdir, dataset, ioff=True)

    im_data.close()
    im_mean.close()
    im_smooth.close()
    im_mask.close()
    if distance_to_edge:
        im_edt.close()
    for im in im_thr:
        im.close()


def extract_resolution_level(image_in, resolution_level):
    """Extract data from a 5D Imaris image pyramid and create 4D image."""

    if resolution_level != -1:
        image_in = '{}/DataSet/ResolutionLevel {}'.format(image_in, resolution_level)

    mo = Image(image_in, permission='r')
    mo.load(load_data=False)

    return mo


def extract_mean(im, dim='c', keep_dtype=True, output=''):
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
        data = im.slice_dataset()
        if im.slices[zdim].stop - im.slices[zdim].start == 1:
            data = np.expand_dims(data, 0)
        data_mean = np.mean(data, axis=im.axlab.index(dim))
        if keep_dtype:
            data_mean = data_mean.astype(im.dtype)
        mo.write(data_mean)
        slc_thrs += list(np.median(np.reshape(data_mean, [data_mean.shape[0], -1]), axis=1))

    mo.slices = None
    mo.set_slices()

    im.close()

    return mo, slc_thrs


def extract_smooth(im, sigma=48.0, keep_dtype=True, output=''):
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

    return mo


def extract_masks(im, thresholds=[1000], output=''):
    """Generate a series masks at different thresholds."""

    data = im.slice_dataset()

    if np.issubdtype(data.dtype, np.integer):
        fstring = 'mask_thr{:05d}'
    else:
        fstring = 'mask_thr{:.5f}'

    mos = []
    c_slcs = {}
    for thr in thresholds:

        ods = fstring.format(thr)

        outputpath = ''
        if output:
            outputpath = output.format(ods)

        props = im.get_props(dtype='bool', squeeze=True)
        mo = MaskImage(outputpath, **props)
        mo.create()
        mask = data > thr
        mo.write(mask)

        mos.append(mo)

    return mos


def postproc_masks(im, thrs=[], fill_holes=True, output=''):
    """Apply slicewise thresholds to data and fill holes.

    NOTE: zyx assumed
    """

    ods = 'mask'.format(0)

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

    return mo


def calculate_distance_to_edge(im, output=''):
    """"Calculate the euclidian distance transform of the mask."""

    ods = 'dist2edge'

    im.load(load_data=False)
    elsize = np.absolute(im.elsize)
    dt = np.zeros(im.ds.shape, dtype='float')
    for i, slc in enumerate(im.ds[:]):
        dt[i, :, :] = distance_transform_edt(slc, sampling=elsize[1:])

    props = im.get_props()
    mo = write_data(dt, props, output, ods)

    return mo


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


def plot_params(f, axdict, info_dict={}):
    """Show images in report."""

    pars = {'resolution_level': 'Image pyramid resolution level',
            'sigma': 'Smoothing sigma',
            'use_median_thresholds': 'Use median threshold method',
            'abs_threshold': 'Absolute threshold',
            'thresholds': 'Thresholds visualized',
            'distance_to_edge': 'Distance to edge calculated',
            'postfix': 'Mask postfix'}

    cellText = []
    for par, name in pars.items():
        v = info_dict['parameters'][par]
        if not isinstance(v, list):
            v = [v]
        cellText.append([name, ', '.join(str(x) for x in v)])

    axdict['p'].table(cellText, loc='bottom')
    axdict['p'].axis('off')


def plot_images(f, axdict, info_dict={}, res=10000, add_profiles=True):
    """Show images in report."""

    def plot_colorbar(ax, im, cax=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        if cax is None:
            cax = divider.append_axes('top', size='5%', pad=0.05)
        cbar = f.colorbar(im, cax=cax, orientation='horizontal')
        return cbar

    def get_img(cslsc, ax_idx, dims):
        img = cslsc
        if ax_idx == 2:  # zy-image
            dims[0] = img.shape[0]
            img = img.transpose()
        if ax_idx == 0:  # yx-image
            dims[1:] = img.shape
        return img, dims

    centreslices = info_dict['centreslices']
    #meds = info_dict['medians']

    y_min, y_max = 0, 0
    dims = [0, 0, 0]

    bf_dict = {'mean': (0, 'r'), 'smooth': (1, 'g'), 'masks': (2, 'b'),
               'mask': (3, 'c'), 'dist2edge': (3, 'c')}

    aspects = ['auto', 'auto', 'auto']
    for dim, aspect, ax_idx in zip('xyz', aspects, [2, 1, 0]):
        for k, v in bf_dict.items():

            if k == 'masks':
                img, dims = get_img(centreslices['mean'][dim], ax_idx, dims)
                img = np.zeros_like(img, dtype='uint8')
                for ids in centreslices.keys():
                    if ids.startswith('mask'):
                        img += get_img(centreslices[ids][dim], ax_idx, dims)[0]
                axdict[k][ax_idx].imshow(img, cmap='gist_rainbow', aspect=aspect)
            elif k == 'mask':
                img, dims = get_img(centreslices['mean'][dim], ax_idx, dims)
                limg, dims = get_img(centreslices[k][dim], ax_idx, dims)
                from skimage.segmentation import find_boundaries
                b = find_boundaries(limg)
                bg = np.zeros_like(img)
                bg[~limg] = 8
                #bg[b] = 0
                axdict[k][ax_idx].imshow(bg, cmap='gist_rainbow', aspect=aspect)
                img = np.ma.masked_where(~limg, img)
                axdict[k][ax_idx].imshow(img, cmap='gray', aspect=aspect)
            elif k == 'dist2edge':
                img, dims = get_img(centreslices[k][dim], ax_idx, dims)
                im = axdict[k][ax_idx].imshow(img, cmap='rainbow', aspect=aspect)
            elif k == 'smooth':
                img, dims = get_img(centreslices[k][dim], ax_idx, dims)
                im = axdict[k][ax_idx].imshow(img, cmap='jet', aspect=aspect)
            else:
                img, dims = get_img(centreslices[k][dim], ax_idx, dims)
                im = axdict[k][ax_idx].imshow(img, cmap='gray', aspect=aspect)

            axdict[k][ax_idx].axis('off')
            y_min = min(y_min, np.floor(np.amin(img) / res) * res)
            y_max = max(y_max, np.ceil(np.amax(img) / res) * res)

    y_min, y_max = 0, 10000  # FIXME
    clim_dict = {'gray': [0, 10000], 'Greys': [0, 10000], 'jet': [0, 10000],
                 'rainbow': [0, 2000],
                 'gist_rainbow': [0, 10], 'inferno': [0, 5]}

    for i in [0, 1, 2]:
        for k in ['mean', 'smooth', 'masks', 'mask', 'dist2edge']:
            for im in axdict[k][i].get_images():
                cld = clim_dict[im.cmap.name]
                im.set_clim(cld[0], cld[1])
            if i == 0:
                plot_colorbar(axdict[k][i], im, axdict[k][3])


def plot_profiles(f, axdict, info_dict={}):
    """Show images in report."""

    axdict['z'].plot(info_dict['parameters']['slicewise_thresholds'], color='r', linewidth=1, linestyle='-')


def gen_subgrid(f, gs, fsize=7, channel=None, metric='median'):
    """3rows-2 columns: 3 image-triplet left, three plots right"""

    fdict = {'fontsize': fsize,
     'fontweight' : matplotlib.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])

    gs00 = gs0[0].subgridspec(10, 1)
    gs01 = gs0[1].subgridspec(3, 2)

    axdict = {k: gen_orthoplot_with_colorbar(f, gs01[i])
              for i, k in enumerate(['mean', 'smooth', 'masks', 'mask', 'dist2edge'])}

    axdict['p'] = f.add_subplot(gs00[0, 0])
    axdict['p'].set_title('parameters', fdict, fontweight='bold')
    axdict['p'].tick_params(axis='both', labelsize=fsize, direction='in')

    axdict['z'] = f.add_subplot(gs01[2, 1])
    axdict['z'].set_title('Z thresholds', fdict, fontweight='bold', loc='right')
    axdict['z'].tick_params(axis='both', labelsize=fsize, direction='in')

    return axdict


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

    return info_dict


def generate_report(outputdir, dataset, ioff=False):

    chsize = (11.69, 8.27)  # A4 portrait
    figtitle = 'STAPL-3D Z-stack mask report'
    filestem = os.path.join(outputdir, dataset)

    figtitle = '{} \n {}'.format(figtitle, dataset)
    subplots = [1, 1]

    figsize = (chsize[1]*subplots[1], chsize[0]*subplots[0])
    f = plt.figure(figsize=figsize, constrained_layout=False)
    gs = gridspec.GridSpec(subplots[0], subplots[1], figure=f)

    axdict = gen_subgrid(f, gs[0])
    image_in = '{}_mask.h5/mean'.format(filestem)
    info_dict = get_info_dict(image_in, channel=0)
    plot_params(f, axdict, info_dict)
    plot_images(f, axdict, info_dict)
    plot_profiles(f, axdict, info_dict)

    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_mask.pdf'.format(filestem))
    if ioff:
        plt.close(f)


if __name__ == "__main__":
    main(sys.argv[1:])
