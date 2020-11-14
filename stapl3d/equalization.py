#!/usr/bin/env python

"""Calculate contrast to noise.

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
import pandas as pd

from glob import glob

from scipy import stats
from scipy.ndimage.filters import gaussian_filter

from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

from sklearn.mixture import GaussianMixture

from stapl3d.preprocessing import shading
from stapl3d.reporting import (
    merge_reports,
    zip_parameters,
    )
from stapl3d import (
    parse_args_common,
    get_n_workers,
    get_outputdir,
    get_params,
    Image,
    )


logger = logging.getLogger(__name__)


def main(argv):
    """Calculate contrast to noise.

    """

    step_ids = ['equalization'] * 4
    fun_selector = {
        'smooth': export_and_smooth,
        'segment': tissue_mask,
        'metrics': calculate_cnr,
        'postprocess': postprocess,
        }

    args, mapper = parse_args_common(step_ids, fun_selector, *argv)

    for step, step_id in mapper.items():
        fun_selector[step](
            args.image_in,
            args.parameter_file,
            step_id,
            args.outputdir,
            args.n_workers,
            )


def export_and_smooth(
    image_in,
    parameter_file,
    step_id='smooth',
    outputdir='',
    n_workers=0,
    sigma=60,
    ):

    outputdir = get_outputdir(image_in, parameter_file, outputdir,
                              'equalization', 'equalization')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    datadir = os.path.dirname(image_in)
    filepaths = glob(os.path.join(datadir, '*.czi'))

    arglist = [
        (
            filepath,
            params['sigma'],
            step_id,
            outputdir,
        )
        for filepath in filepaths]

    n_workers = get_n_workers(len(filepaths), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(run_export_and_smooth, arglist)


def run_export_and_smooth(
    filepath,
    sigma=60,
    step_id='smooth',
    outputdir='',
    ):

    filename = os.path.basename(filepath)
    filestem = filename.split('.czi')[0]

    # Load image => czi (/ lif?)
    iminfo = shading.get_image_info(filepath)
    data = np.transpose(shading.read_tiled_plane(filepath, 0, 0)[0])
    data_smooth = gaussian_filter(data.astype('float'), sigma)

    # Export as nifti file
    props = {}
    props['axlab'] = 'xyz'
    props['shape'] = iminfo['dims_zyxc'][:3][::-1]
    props['elsize'] = iminfo['elsize_zyxc'][:3][::-1]
    props['elsize'][2] = 1.0  # FIXME

    outputpath = os.path.join(outputdir, '{}.nii.gz'.format(filestem))
    data = np.reshape(data, props['shape'])
    output_nifti(outputpath, data, props)

    outputpath = os.path.join(outputdir, '{}_smooth.nii.gz'.format(filestem))
    data_smooth = np.reshape(data_smooth, props['shape'])
    output_nifti(outputpath, data_smooth, props)

    # Dump parameters.
    params = {'sigma': sigma}
    outputpath = os.path.join(outputdir, '{}.pickle'.format(filestem))
    with open(outputpath,'wb') as f:
        pickle.dump(params, f)


def tissue_mask(
    image_in,
    parameter_file,
    step_id='smooth',
    outputdir='',
    n_workers=0,
    otsu_lower=0.5,
    otsu_upper=0.1,
    threshold_lower=0,
    threshold_upper=0,
    ):

    outputdir = get_outputdir(image_in, parameter_file, outputdir,
                              'equalization', 'equalization')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    datadir = os.path.dirname(image_in)
    filepaths = glob(os.path.join(datadir, '*.czi'))

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    thresholds = []
    for filepath in filepaths:
        try:
            filestem = os.path.basename(filepath).split('.czi')[0]
            thrs = params['thresholds'][filestem]
        except KeyError:
            thrs = [0, 0]
        thresholds.append(thrs)

    arglist = [
        (
            filepath,
            params['otsu_lower'],
            params['otsu_upper'],
            thrs[0],
            thrs[1],
            step_id,
            outputdir,
        )
        for filepath, thrs in zip(filepaths, thresholds)]

    n_workers = get_n_workers(len(filepaths), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(run_tissue_mask, arglist)


def run_tissue_mask(
    filepath,
    otsu_lower=0.1,
    otsu_upper=1.1,
    threshold_lower=0,
    threshold_upper=0,
    step_id='tissue_mask',
    outputdir='',
    ):

    filename = os.path.basename(filepath)
    filestem = filename.split('.czi')[0]

    data, _ = load_nifti(outputdir, filestem, '')
    smooth, props = load_nifti(outputdir, filestem, '_smooth')

    if not threshold_lower:
        otsu = threshold_otsu(smooth)
        threshold_lower = otsu * otsu_lower
        threshold_upper = otsu * otsu_upper

    noise_mask = smooth < threshold_lower
    tissue_mask = smooth > threshold_upper
    tissue_mask = np.logical_and(tissue_mask, ~noise_mask)

    outputpath = os.path.join(outputdir, '{}_noise_mask.nii.gz'.format(filestem))
    output_nifti(outputpath, noise_mask.astype('uint8'), props)

    outputpath = os.path.join(outputdir, '{}_tissue_mask.nii.gz'.format(filestem))
    output_nifti(outputpath, tissue_mask.astype('uint8'), props)

    # Dump parameters.
    outputpath = os.path.join(outputdir, '{}.pickle'.format(filestem))
    with open(outputpath,'rb') as f:
        params = pickle.load(f)
    pars = {
        'threshold_lower': threshold_lower,
        'threshold_upper': threshold_upper,
        'otsu_lower': otsu_lower,
        'otsu_upper': otsu_upper,
        'otsu': otsu,
    }
    params.update(pars)
    with open(outputpath,'wb') as f:
        pickle.dump(params, f)


def calculate_cnr(
    image_in,
    parameter_file,
    step_id='smooth',
    outputdir='',
    n_workers=0,
    methods=['seg'],
    quantiles=[0.5, 0.99],
    ):

    outputdir = get_outputdir(image_in, parameter_file, outputdir,
                              'equalization', 'equalization')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    datadir = os.path.dirname(image_in)
    filepaths = glob(os.path.join(datadir, '*.czi'))

    arglist = [
        (
            filepath,
            params['methods'],
            params['quantiles'],
            step_id,
            outputdir,
        )
        for filepath in filepaths]

    n_workers = get_n_workers(len(filepaths), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(run_calculate_cnr, arglist)


def run_calculate_cnr(
    filepath,
    methods=['seg'],
    quantiles=[0.5, 0.99],
    step_id='tissue_mask',
    outputdir='',
    ):

    filename = os.path.basename(filepath)
    filestem = filename.split('.czi')[0]

    image, props = load_nifti(outputdir, filestem, '')
    smooth, props = load_nifti(outputdir, filestem, '_smooth')

    # FIXME
    # if isinstance(image.dtype, int):
    #     infun = np.iinfo
    # elif isinstance(image.dtype, float):
    #     infun = np.finfo
    infun = np.iinfo

    def q_base(image, quantiles, outputdir, dataset, infun=np.iinfo):
        """First approximation: simple quantiles."""
        data = np.ravel(image)
        vals, score = get_measures(data, quantiles)
        cnr = None
        return pd.DataFrame([vals[0], vals[1], score, cnr]).T

    def q_clip(image, quantiles, outputdir, dataset, infun=np.iinfo):
        """Second approximation: simple quantiles of non-clipping."""
        mask = np.logical_and(image > infun(image.dtype).min,
                              image < infun(image.dtype).max)
        data = np.ravel(image[mask])
        vals, score = get_measures(data, quantiles)
        cnr = None
        return pd.DataFrame([vals[0], vals[1], score, cnr]).T

    def q_mask(image, quantiles, outputdir, dataset, infun=np.iinfo):
        """Third approximation: simple quantiles of image[tissue_mask]."""
        tissue_mask, _ = load_nifti(outputdir, dataset, '_tissue_mask')
        tissue_mask = tissue_mask.astype('bool')
        mask = np.logical_and(tissue_mask, image < infun(image.dtype).max)
        data = np.ravel(image[mask])
        mode = stats.mode(data)[0][0]
        vals, score = get_measures(data, quantiles)
        # noise_mask = ~tissue_mask
        noise_mask, _ = load_nifti(outputdir, dataset, '_noise_mask')
        noise_mask = noise_mask.astype('bool')
        cnr = get_cnr(image, noise_mask, vals[1], vals[0])
        return pd.DataFrame([vals[0], vals[1], score, cnr]).T

    def gmm(image, quantiles, outputdir, dataset, infun=np.iinfo):
        """Fourth approximation: 2-comp GMM of segmentation image[tissue_mask]."""
        tissue_mask, _ = load_nifti(outputdir, dataset, '_tissue_mask')
        tissue_mask = tissue_mask.astype('bool')
        X_train = np.ravel(image[tissue_mask])
        X_train = X_train.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, verbose=0, covariance_type="full")
        gmm.fit(X_train)
        means = [gmm.means_[0][0], gmm.means_[1][0]]
        vals = np.sort(means)
        score = vals[1] / vals[0]
        cnr = get_cnr(image, noise_mask, vals[1], vals[0])
        return pd.DataFrame([vals[0], vals[1], score, cnr]).T

    def seg(image, quantiles, outputdir, dataset, infun=np.iinfo):
        """Fifth approximation: 3-comp segmentation image[tissue_mask]."""

        tissue_mask, _ = load_nifti(outputdir, dataset, '_tissue_mask')
        tissue_mask = tissue_mask.astype('bool')
        noise_mask, _ = load_nifti(outputdir, dataset, '_noise_mask')
        noise_mask = noise_mask.astype('bool')

        df = q_mask(image, quantiles, outputdir, dataset, infun=np.iinfo)
        signal_mask = image > df.iloc[0, 1]
        remove_small_objects(signal_mask, min_size=3, connectivity=1, in_place=True)

        segmentation = np.zeros_like(tissue_mask, dtype='uint8')
        segmentation[tissue_mask] = 1
        segmentation[signal_mask] = 2

        outputpath = os.path.join(outputdir, '{}_segmentation.nii.gz'.format(filestem))
        output_nifti(outputpath, segmentation, props)

        vals = [np.median(np.ravel(image[segmentation==1])),
                np.median(np.ravel(image[segmentation==2]))]
        score = vals[1] / vals[0]
        cnr = get_cnr(image, noise_mask, vals[1], vals[0])
        return pd.DataFrame([vals[0], vals[1], score, cnr]).T

    df = pd.DataFrame()
    metrics = ['q1', 'q2', 'contrast', 'cnr']
    meths = {'q_base': q_base, 'q_clip': q_base, 'q_mask': q_base, 'gmm': gmm, 'seg': seg}
    for method, fun in meths.items():
        if method in methods:
            df0 = fun(image, quantiles, outputdir, filestem, infun=infun)
            df0.columns=['{}-{}'.format(method, metric) for metric in metrics]
            df = pd.concat([df, df0], axis=1)

    df.index = [filestem]
    outputpath = os.path.join(outputdir, '{}.csv'.format(filestem))
    df.to_csv(outputpath, index_label='sample_id')

    # Dump parameters.
    outputpath = os.path.join(outputdir, '{}.pickle'.format(filestem))
    with open(outputpath,'rb') as f:
        params = pickle.load(f)
    pars = {
        'quantiles': quantiles,
        }
    params.update(pars)
    with open(outputpath,'wb') as f:
        pickle.dump(params, f)

    generate_report(outputdir, filestem)


def postprocess(
    image_in,
    parameter_file,
    step_id='postprocess',
    outputdir='',
    n_workers=0,
    ):

    outputdir = get_outputdir(image_in, parameter_file, outputdir, 'equalization', 'equalization')
    inputpat = '{}/*'.format(outputdir)
    outputstem = os.path.join(outputdir, 'equalization_assay')

    pdfs = glob('{}_equalization_assay.pdf'.format(inputpat))
    pdfs.sort()
    pdf_out = '{}.pdf'.format(outputstem)
    merge_reports(pdfs, pdf_out)

    pickles = glob('{}.pickle'.format(inputpat))
    pickles.sort()
    zip_out = '{}.zip'.format(outputstem)
    zip_parameters(pickles, zip_out)

    csvs = glob('{}.csv'.format(inputpat))
    csvs.sort()
    df = pd.DataFrame()
    for csvfile in csvs:
        df0 = pd.read_csv(csvfile)
        df = pd.concat([df, df0], axis=0)
    df = df.set_index('sample_id')
    outputpath = os.path.join(outputdir, 'equalization_assay.csv')
    df.to_csv(outputpath, index_label='sample_id')

    # Plot results.
    step_id = 'equalization_assay'
    dataset = 'summary'

    chsize = (11.69, 8.27)  # A4 portrait
    figtitle = 'STAPL-3D {} report \n {}'.format(step_id, dataset)
    filestem = os.path.join(outputdir, dataset)
    subplots = [1, 1]
    figsize = (chsize[1]*subplots[1], chsize[0]*subplots[0])
    f, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    df = df.sort_values('seg-cnr')
    df['seg-cnr'].plot(ax=ax, kind='barh')
    ax.set_title("contrast-to-noise", fontsize=12)

    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_summary.pdf'.format(outputstem))


def output_nifti(outputpath, data, props):

    if len(data.shape) == 4:
        props['axlab'] += 'c'
        props['elsize'] += [1]
    mo = Image(outputpath, **props)
    mo.dtype = data.dtype
    mo.dims = data.shape
    mo.slices = None
    mo.create()
    mo.write(data)
    mat = mo.get_transmat()
    mo.nii_write_mat(data, mo.slices, mat)
    mo.close()


def load_nifti(datadir, filestem, postfix):

    inputpath = os.path.join(datadir, '{}{}.nii.gz'.format(filestem, postfix))
    im = Image(inputpath)
    im.load()
    data = im.ds[:]
    props = im.get_props()
    im.close()

    return data, props


def get_measures(data, quantiles=[0.5, 0.9]):

    quantiles = np.quantile(data, quantiles)

    score = quantiles[1] / quantiles[0]

    return quantiles, score


def get_cnr(image, noise_mask, sigval, bgval):

    noise = np.ravel(image[noise_mask])
    if noise.size > 0:
        cnr = (sigval - bgval) / np.std(noise)
    else:
        cnr = 0

    return cnr


def plot_params(f, axdict, info_dict={}):
    """Show images in report."""

    pars = {'sigma': 'smoothing sigma',
            'otsu_upper': 'otsu factor: noise',
            'otsu_lower': 'otsu factor: tissue',
            'otsu': 'otsu threshold',
            'threshold_lower': 'threshold_lower',
            'threshold_upper': 'threshold_lower',
            'quantiles': 'quantiles'}

    cellText = []
    for par, name in pars.items():
        v = info_dict['parameters'][par]
        if not isinstance(v, list):
            v = [v]
        try:
            fs = '{:.2f}' if np.issubdtype(v[0].dtype, np.float) else '{}'
        except AttributeError:
            fs = '{:.2f}' if isinstance(v[0], float) else '{}'
        cellText.append([name, ', '.join(fs.format(x) for x in v)])

    axdict['p'].table(cellText, loc='bottom')
    axdict['p'].axis('off')


def plot_images(f, axdict, info_dict):

    image = info_dict['centreslices']['image']['z']
    image_smooth = info_dict['centreslices']['smooth']['z']
    noise_mask = info_dict['centreslices']['noise_mask']['z']

    segmentation = info_dict['centreslices']['seg']['z']
    tissue_mask = segmentation > 0
    background_mask = segmentation == 1
    signal_mask = segmentation == 2

    w = info_dict['elsize']['x'] * image.shape[0]  # note xyz nifti
    h = info_dict['elsize']['y'] * image.shape[1]
    extent = [0, w, 0, h]
    ax = axdict['i0']
    ax.imshow(image, cmap="gray", extent=extent)
    ax.set_axis_off()

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=12)
    scalebar = AnchoredSizeBar(ax.transData,
                               int(w/5), r'{} $\mu$m'.format(int(w/5)), 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)

    ax = axdict['i1']
    ax.imshow(image_smooth, cmap="gray")
    if noise_mask is not None:
        ax.contour(noise_mask, [0.5], colors='r', linestyles='dashed')
    if tissue_mask is not None:
        ax.contour(segmentation > 0, [0.5], colors='b', linestyles='dashed')
    ax.set_axis_off()

    ax = axdict['i2']
    clabels = label2rgb(segmentation, image=image, alpha=0.5, bg_label=0, colors=[[1, 0, 0], [0, 1, 0]])
    ax.imshow(clabels)
    ax.set_axis_off()

    # Plot histograms
    def draw_thresholds(ax, thresholds, colors, linestyles):
        for t, c, l in zip(thresholds, colors, linestyles):
            if t is not None:
                ax.axvline(t, color=c, linestyle=l)

    thresholds = (
        info_dict['parameters']['threshold_lower'],
        info_dict['parameters']['otsu'],
        info_dict['parameters']['threshold_upper'],
        )

    ax = axdict['h0']
    ax.hist(np.ravel(image), bins=256)

    ax = axdict['h1']
    ax.hist(np.ravel(image_smooth), bins=256)
    draw_thresholds(ax, thresholds, 'rkb', '---')

    ax = axdict['h2']
    data = [np.ravel(image[background_mask]), np.ravel(image[signal_mask])]
    ax.hist(data, bins=256, histtype='bar', stacked=True)


def gen_subgrid(f, gs, fsize=7):
    """4rows-2 columns: 4 images left, 4 plots right"""

    fdict = {'fontsize': fsize,
     'fontweight' : matplotlib.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])

    gs00 = gs0[0].subgridspec(10, 1)
    gs01 = gs0[1].subgridspec(3, 2)

    axdict = {}

    # parameter table
    axdict['p'] = f.add_subplot(gs00[0, 0])
    axdict['p'].set_title('parameters', fdict, fontweight='bold')
    axdict['p'].tick_params(axis='both', labelsize=fsize, direction='in')

    # images
    titles = {'i0': 'image', 'i1': 'tissue / noise regions', 'i2': 'segmentation'}
    for i, (a, t) in enumerate(titles.items()):
        axdict[a] = f.add_subplot(gs01[i, 0])
        axdict[a].set_title(t, fdict, fontweight='bold', loc='left')
        axdict[a].tick_params(axis='both', labelsize=fsize, direction='in')

    # histograms
    titles = {'h0': 'histogram', 'h1': 'histogram smoothed image', 'h2': 'histogram tissue'}
    for i, (a, t) in enumerate(titles.items()):
        axdict[a] = f.add_subplot(gs01[i, 1])
        axdict[a].set_title(t, fdict, fontweight='bold', loc='right')
        axdict[a].tick_params(axis='both', labelsize=fsize, direction='in')

    return axdict


def get_info_dict(image_in, info_dict={}):

    im = Image(image_in)
    im.load(load_data=False)
    info_dict['elsize'] = {dim: im.elsize[i] for i, dim in enumerate(im.axlab)}
    info_dict['paths'] = im.split_path()  # FIXME: out_base
    info_dict['paths']['out_base'] = info_dict['paths']['base']
    im.close()

    ppath = '{}.pickle'.format(info_dict['paths']['base'])
    with open(ppath, 'rb') as f:
        info_dict['parameters'] = pickle.load(f)

    # TODO: generalize 3D?
    # TODO: integrate with get_centreslices() => nii or make it h5
    image, _ = load_nifti(info_dict['paths']['dir'], info_dict['paths']['fname'], '')
    smooth, _ = load_nifti(info_dict['paths']['dir'], info_dict['paths']['fname'], '_smooth')
    seg, _ = load_nifti(info_dict['paths']['dir'], info_dict['paths']['fname'], '_segmentation')
    noise_mask, _ = load_nifti(info_dict['paths']['dir'], info_dict['paths']['fname'], '_noise_mask')

    info_dict['centreslices'] = {
        'image': {'z': np.squeeze(image)},
        'smooth': {'z': np.squeeze(smooth)},
        'seg': {'z': np.squeeze(seg)},
        'noise_mask': {'z': np.squeeze(noise_mask)},
        }

    return info_dict


def generate_report(outputdir, dataset, channel=None, ioff=False):
    """Generate a QC report of the equalization."""

    step_id = 'equalization_assay'

    chsize = (11.69, 8.27)  # A4 portrait
    figtitle = 'STAPL-3D {} report \n {}'.format(step_id, dataset)
    filestem = os.path.join(outputdir, dataset)
    subplots = [1, 1]
    figsize = (chsize[1]*subplots[1], chsize[0]*subplots[0])
    f = plt.figure(figsize=figsize, constrained_layout=False)
    gs = gridspec.GridSpec(subplots[0], subplots[1], figure=f)

    outputstem = filestem

    axdict = gen_subgrid(f, gs[0], fsize=10)

    image_in = os.path.join(outputdir, '{}.nii.gz'.format(dataset))
    info_dict = get_info_dict(image_in)

    plot_params(f, axdict, info_dict)
    plot_images(f, axdict, info_dict)
    # plot_profiles(f, axdict, info_dict)
    info_dict.clear()

    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    f.savefig('{}_{}.pdf'.format(outputstem, step_id))
    if ioff:
        plt.close(f)


if __name__ == "__main__":
    main(sys.argv[1:])
