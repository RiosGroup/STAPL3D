#!/usr/bin/env python

"""Functions for generating reports.

"""

import sys
import argparse

import os
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.color import label2rgb

from stapl3d import Image


def generate_report_page(outputdir, dataset, report_type, channel=None, block_id=None, cl=False):
    """Generate a QC report page."""

    fdict = {
        'fontsize': 10,
        'fontweight' : matplotlib.rcParams['axes.titleweight'],
        'verticalalignment': 'baseline',
        }

    figsize = (11.69, 8.27)  # A4 portrait
    figtitle = 'STAPL-3D QC report: {}'.format(report_type)
    filestem = os.path.join(outputdir, dataset)

    if channel is not None:
        figtitle = '{} \n {}: channel {:02d} \n'.format(figtitle, dataset, channel)
        filestem = '{}_ch{:02d}'.format(filestem, channel)

    outputstem = filestem

    f = plt.figure(figsize=figsize, constrained_layout=cl)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 10, 1], figure=f)

    plot_parameters(f, gs[0], filestem, fdict)


    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_{}-report.pdf'.format(outputstem, report_type))


def generate_subgrid(f, gs, fsize=7, nrows=3):

    fdict = {'fontsize': fsize,
     'fontweight' : matplotlib.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    gs0 = gs.subgridspec(3, 1, height_ratios=[1, 10, 1]) # pars, plots, stats


def load_parameters(filestem):
    """Load the parameters for the preprocessing step."""

    try:
        ppath = '{}.pickle'.format(filestem)
        if False:  # FIXME: pickle a dict (not list) in shading estimation
            with open(ppath, 'rb') as f:
                paramlist = pickle.load(f)
            params = {}
            params['metric'] = paramlist[0]
            params['noise_threshold'] = 1000
            params['quantile_threshold'] = paramlist[2]
            params['polynomial_order'] = paramlist[3]
        else:
            with open(ppath, 'rb') as f:
                params = pickle.load(f)
    except:
        params = {}

    return params


def select_parameters(report_type):
    """Return a dictionary of par-name pairs to include in report."""
    # TODO: set from yaml

    if report_type == 'shading':
        pars = {'noise_threshold': 'Noise threshold',
                'metric': 'Metric',
                'quantile_threshold': 'Quantile threshold',
                'polynomial_order': 'Polynomial order'}
        # stats = {}
    elif report_type == 'biasfield':
        pars = {'downsample_factors': 'Downsample factors',
                'n_iterations': 'N iterations',
                'n_fitlevels': 'N fitlevels',
                'n_bspline_cps': 'N b-spline components'}
        # stats = {}
    elif report_type == 'segmentation':
        pars = {}
        # stats = {}
    elif report_type == 'zipping':
        pars = {}
        # stats = {}

    return pars


def plot_parameters(f, gs, filestem, fdict={'fontsize': 10}):
    """Show parameter table in report."""

    params = load_parameters(filestem, 'shading')
    parsel = select_parameters(report_type)

    ax = f.add_subplot(gs)
    ax.set_title('Selected parameters', fdict, fontweight='bold')
    ax.tick_params(axis='both', direction='in', labelsize=fdict['fontsize'])

    cellText = []
    for par, name in parsel.items():
        print(par, name)
        v = params[par]
        if not isinstance(v, list):
            v = [v]
        cellText.append([name, ', '.join(str(x) for x in v)])

    ax.table(cellText, loc='bottom')
    ax.axis('off')


def get_paths(image_in, resolution_level=-1, channel=0, outputstem='', step='', save_steps=False):
    """Get the parameters for the preprocessing step."""

    # get paths from input file
    if resolution_level != -1:  # we should have an Imaris pyramid
        image_in = '{}/DataSet/ResolutionLevel {}'.format(image_in, resolution_level)
    im = Image(image_in, permission='r')
    im.load(load_data=False)
    paths = im.split_path()
    im.close()

    # define output basename
    paths['out_base'] = outputstem

    # define h5 output template
    paths['out_h5'] = '{}.h5/{}'.format(paths['out_base'], '{}')

    # define output for main results and intermediate steps
    if not outputstem:
        paths['main'] = paths['steps'] = ''
    else:
        ### FIXME ????????????? what is this?
        if save_steps:
            paths['main'] = paths['steps'] = paths['out_h5']
        else:
            paths['main'] = paths['out_h5']
            paths['steps'] = paths['out_h5']

    # define output for parameters
    paths['params'] = '{}.pickle'.format(paths['out_base'], step)

    return paths


def get_centreslices(info_dict, idss=[], ch=0):
    """Get the centreslices for all dims and steps."""

    h5_path = info_dict['paths']['file']
    image_in = h5_path + info_dict['paths']['int']
    im = Image(image_in)
    im.load(load_data=False)
    if not idss:
        idss = [k for k in im.file.keys()]
    im.close()

    centreslices = {ids: {dim: get_centreslice(h5_path, ids, dim, ch)
                          for dim in 'zyx'}
                    for ids in idss}

    return centreslices


def get_centreslice(image_in, ids, dim='z', ch=0):
    """Return an image's centreslice for a dimension."""

    if isinstance(image_in, Image):
        im = image_in
    else:
        im = Image('{}/{}'.format(image_in, ids))
        try:
            im.load(load_data=False)
        except KeyError:
            print('dataset {} not found'.format(ids))
            return None

    if len(im.dims) > 3:
        ch_idx = im.axlab.index('c')
        im.slices[ch_idx] = slice(ch, ch + 1, 1)

    dim_idx = im.axlab.index(dim)
    cslc = int(im.dims[dim_idx] / 2)

    slcs = [slc for slc in im.slices]
    im.slices[dim_idx] = slice(cslc, cslc+1, 1)
    data = im.slice_dataset()
    im.slices = slcs

    return data


def get_cslc(data, axis=0):
    """Get centreslices from a numpy array."""

    slcs = [slice(0, s, 1) for s in data.shape]
    cslc = int(data.shape[axis] / 2)
    slcs[axis] = slice(cslc, cslc+1, 1)

    return np.copy(np.squeeze(data[tuple(slcs)]))


def get_zyx_medians(data, mask=None, thr=0, metric='median'):
    """Compute the dataset (masked) median profiles over z, y, and x."""

    if mask is None:
        mask = data < thr
    d = np.ma.masked_array(data, mask)

    def get_dim_median(d, tp, metric='median'):
        if metric == 'median':
            return np.ma.median(np.reshape(np.transpose(d, tp), [d.shape[tp[0]], -1]), axis=1)
        elif metric == 'mean':
            return np.ma.mean(np.reshape(np.transpose(d, tp), [d.shape[tp[0]], -1]), axis=1)
        elif metric == 'std':
            return np.ma.std(np.reshape(np.transpose(d, tp), [d.shape[tp[0]], -1]), axis=1)

    tps = {'z': [0, 1, 2], 'y': [1, 2, 0], 'x': [2, 0, 1]}
    meds = {dim: get_dim_median(d, tp, metric) for dim, tp in tps.items()}

    return meds


def gen_orthoplot(f, gs):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_xy = 5
    size_z = 1
    size_t = size_xy + size_z

    gs_sub = gs.subgridspec(size_t, size_t)

    # central: yx-image
    axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy]))
    # middle-bottom: zx-image
    axs.append(f.add_subplot(gs_sub[size_xy:, :size_xy], sharex=axs[0]))
    # right-middle: zy-image
    axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:], sharey=axs[0]))

    return axs


def gen_orthoplot_with_colorbar(f, gs):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_xy = 10
    size_z = 2
    size_c = 1
    size_t = size_xy + size_z + size_c

    gs_sub = gs.subgridspec(size_xy + size_z + size_c, size_xy + size_z)

    # central: yx-image
    axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy]))
    # middle-bottom: zx-image
    axs.append(f.add_subplot(gs_sub[size_xy:, :size_xy], sharex=axs[0]))
    # right-middle: zy-image
    axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:], sharey=axs[0]))
    # right colorbar
    #axs.append(f.add_subplot(gs_sub[:, -size_c:]))
    # bottom colorbar
    axs.append(f.add_subplot(gs_sub[-size_c:, :]))

    return axs


def gen_orthoplot_with_profiles(f, gs):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_p = 2
    size_xy = 5
    size_z = 2
    size_t = size_p + size_xy + size_z

    gs_sub = gs.subgridspec(size_t, size_t)

    # central: yx-image
    axs.append(f.add_subplot(gs_sub[size_p:size_p+size_xy, size_p:size_p+size_xy]))
    # middle-bottom: zx-image
    axs.append(f.add_subplot(gs_sub[size_p+size_xy:, size_p:size_p+size_xy], sharex=axs[0]))
    # right-middle: zy-image
    axs.append(f.add_subplot(gs_sub[size_p:size_p+size_xy, size_p+size_xy:], sharey=axs[0]))
    # right-top: z-profiles
    axs.append(f.add_subplot(gs_sub[:size_p, size_p+size_xy:], sharex=axs[2]))
    # left-middle: y-profiles
    axs.append(f.add_subplot(gs_sub[size_p:size_p+size_xy, :size_p], sharey=axs[0]))
    # middle-top: x-profiles
    axs.append(f.add_subplot(gs_sub[:size_p, size_p:size_p+size_xy], sharex=axs[0]))

    return axs


def merge_reports(pdfs, outputpath):
    """Merge pages of a report."""

    if not pdfs:
        return

    from PyPDF2 import PdfFileMerger

    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(outputpath)
    merger.close()


def add_titles(axs, info_dict):
    """Add plot titles to upper row of plot grid."""

    return


def zip_parameters(filestem, report_type='bfc'):
    """Zip a set of parameterfiles for archiving."""

    import zipfile
    from glob import glob
    zips = glob('{}_*_{}-params.pickle'.format(filestem, report_type))
    zips.sort()
    outputpath = '{}_{}-params.zip'.format(filestem, report_type)
    zf = zipfile.ZipFile(outputpath, mode='w')
    for pfile in zips:
        zf.write(pfile, basename(pfile))
    zf.close()
