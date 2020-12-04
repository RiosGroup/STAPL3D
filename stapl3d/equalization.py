#!/usr/bin/env python

"""Calculate metrics for mLSR-3D equalization assay.

# TODO: implement other than czi (e.g. make tif a possibility here)

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

from stapl3d import parse_args, Stapl3r, Image, transpose_props, get_imageprops, get_paths
from stapl3d.preprocessing import shading
from stapl3d.reporting import merge_reports, get_centreslices

# from stapl3d import mplog
logger = logging.getLogger(__name__)


def main(argv):
    """Calculate metrics for mLSR-3D equalization assay."""

    steps = ['smooth', 'segment', 'metrics', 'postprocess']
    args = parse_args('equalization', steps, *argv)

    equalizer = Equalizer(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        dataset=args.dataset,
        suffix=args.suffix,
        n_workers=args.n_workers,
    )

    for step in args.steps:
        equalizer._fun_selector[step]()


class Equalizer(Stapl3r):
    """Calculate metrics for mLSR-3D equalization assay."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Equalizer, self).__init__(
            image_in, parameter_file,
            module_id='equalization',
            **kwargs,
            )

        self._fun_selector = {
            'smooth': self.smooth,
            'segment': self.segment_regions,
            'metrics': self.calculate_metrics,
            'postprocess': self.postprocess,
            }

        default_attr = {
            'step_id': 'equalization',
            'sigma': 60,
            'filepat': '*.czi',  # TODO
            'filepaths': [],
            'otsu_factor_noise': 0.1,
            'otsu_factor_tissue': 1.1,
            'thresholds': {},
            'threshold_noise': 0,
            'threshold_tissue': 0,
            '_otsus': {},
            'segment_quantile': 0.99,
            'segment_min_size': 3,
            'methods': ['seg'],
            'quantiles': [0.50, 0.99],
            '_metrics': {},
            'outputformat': '.h5',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self.set_filepaths()

        self._parsets = {
            'smooth': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('sigma',),
                'spar': ('n_workers', 'filepaths'),
                },
            'segment': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('otsu_factor_noise', 'otsu_factor_tissue',
                         'threshold_noise', 'threshold_tissue',
                         'thresholds', '_otsus',
                         'segment_quantile', 'segment_min_size'),
                'spar': ('filepaths',),
                },
            'metrics': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('methods', 'quantiles', '_metrics'),
                'spar': ('filepaths',),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': (),
                },
            }

        # TODO: merge with parsets
        self._partable = {
            'sigma': 'Smoothing sigma',
            'otsu_factor_noise': 'Multiplication factor for noise threshold',
            'otsu_factor_tissue': 'Multiplication factor for tissue threshold',
            # 'threshold_noise': 'Noise threshold',
            # 'threshold_tissue': 'Tissue threshold',
            'segment_quantile': 'Quantile used for segmentation',
            'segment_min_size': 'Minimal connected component size',
            'quantiles': 'Quantiles separating signal and tissue background',
            }

    def smooth(self, **kwargs):
        """Smooth images with a gaussian kernel."""

        self.set_parameters('smooth', kwargs)
        arglist = self._get_arglist(['filepaths'])
        self.set_n_workers(len(arglist))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._smooth_image, arglist)

    def _smooth_image(self, filepath):
        """Smooth an image with a gaussian kernel."""

        filestem, ext, outputpat = self._get_outputpat(filepath)

        # self._logmp(filestem)

        # Load image => czi (/ lif?)
        data = np.transpose(shading.read_tiled_plane(filepath, 0, 0)[0])
        data_smooth = gaussian_filter(data.astype('float'), self.sigma)

        iminfo = shading.get_image_info(filepath)
        # TODO: integrate iminfo and Image attributes
        props = {}
        props['path'] = ''
        props['permission'] = 'r+'
        props['axlab'] = 'zyx'
        props['shape'] = iminfo['dims_zyxc'][:3]
        props['elsize'] = iminfo['elsize_zyxc'][:3]
        props['elsize'] = [es if es else 1.0 for es in props['elsize']]
        # FIXME: for 2D the z-axis is returned 0.0
        if 'nii' in self.outputformat:
            props = transpose_props(props)

        for ids, out in {'data': data, 'smooth': data_smooth}.items():
            props['path'] = outputpat.format(ids)
            write_image(out, props)

    def segment_regions(self, **kwargs):
        """Segment the noise and tissue region in the image."""

        self.set_parameters('segment', kwargs)
        arglist = self._get_arglist(['filepaths'])
        self.set_n_workers(len(self.filepaths))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._segment_regions_image, arglist)

    def _segment_regions_image(self, filepath):

        filestem, ext, outputpat = self._get_outputpat(filepath)

        data, props = load_image(outputpat.format('data'))
        smooth, props = load_image(outputpat.format('smooth'))

        masks, otsu, thrs = self._region_masks(smooth, filestem)
        masks = self._tissue_segmentation(data, masks)

        for ids, out in masks.items():
            props['path'] = outputpat.format(ids)
            write_image(out, props)

        self.thresholds = {filestem: thrs}
        self._otsus = {filestem: otsu}
        basename = self.format_([filestem, self._module_id, self.step])
        outstem = os.path.join(self.directory, basename)
        self.dump_parameters(step=self.step, filestem=outstem)

    def _region_masks(self, data, filestem):

        if filestem in self.thresholds.keys():
            thresholds = self.thresholds[filestem]
        else:
            thresholds = [self.threshold_noise, self.threshold_tissue]

        otsu = None
        if not any(thresholds):
            otsu = float(threshold_otsu(data))
            mi, ma = np.amin(data), np.amax(data)
            thresholds[0] = float(thresholds[0] or mi + otsu * self.otsu_factor_noise)
            thresholds[1] = float(thresholds[1] or otsu * self.otsu_factor_tissue)

        noise_mask = data < thresholds[0]
        tissue_mask = data > thresholds[1]
        tissue_mask = np.logical_and(tissue_mask, ~noise_mask)

        masks = {'noise_mask': noise_mask, 'tissue_mask': tissue_mask}

        return masks, otsu, thresholds

    def _tissue_segmentation(self, data, masks):

        noise_mask = masks['noise_mask']
        tissue_mask = masks['tissue_mask']

        # clipper
        infun = np.iinfo if data.dtype.kind in 'ui' else np.finfo
        clipping_mask = data == infun(data.dtype).max
        tissue_mask &= ~clipping_mask

        signal_mask = data > np.quantile(data[tissue_mask], self.segment_quantile)
        signal_mask &= ~clipping_mask
        remove_small_objects(signal_mask, min_size=self.segment_min_size,
                             in_place=True)

        segmentation = np.zeros_like(tissue_mask, dtype='uint8')
        segmentation[noise_mask]  = 1
        segmentation[tissue_mask] = 2
        segmentation[signal_mask] = 3
        segmentation[clipping_mask] = 0

        masks['segmentation'] = segmentation

        return masks

    def calculate_metrics(self, **kwargs):

        self.set_parameters('metrics', kwargs)
        arglist = self._get_arglist(['filepaths'])
        self.set_n_workers(len(self.filepaths))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._calculate_metrics_image, arglist)

    def _calculate_metrics_image(self, filepath):

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

        def q_base(image, quantiles, outputdir, dataset):
            """Simple quantiles."""
            data = np.ravel(image)
            vals, score = get_measures(data, quantiles)
            cnr = None
            return pd.DataFrame([vals[0], vals[1], score, cnr]).T

        def q_clip(image, quantiles, outputdir, dataset):
            """Simple quantiles of non-clipping."""
            infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
            mask = np.logical_and(image > infun(image.dtype).min,
                                  image < infun(image.dtype).max)
            data = np.ravel(image[mask])
            vals, score = get_measures(data, quantiles)
            cnr = None
            return pd.DataFrame([vals[0], vals[1], score, cnr]).T

        def q_mask(image, quantiles, outputdir, dataset):
            """Simple quantiles of image[tissue_mask]."""
            tissue_mask, _ = load_image(outputpat.format('tissue_mask'))
            tissue_mask = tissue_mask.astype('bool')
            infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
            mask = np.logical_and(tissue_mask, image < infun(image.dtype).max)
            data = np.ravel(image[mask])
            mode = stats.mode(data)[0][0]
            vals, score = get_measures(data, quantiles)
            # noise_mask = ~tissue_mask
            noise_mask, _ = load_image(outputpat.format('noise_mask'))
            noise_mask = noise_mask.astype('bool')
            cnr = get_cnr(image, noise_mask, vals[1], vals[0])
            return pd.DataFrame([vals[0], vals[1], score, cnr]).T

        def gmm(image, quantiles, outputdir, dataset):
            """Two-comp GMM of segmentation image[tissue_mask]."""
            tissue_mask, _ = load_image(outputpat.format('tissue_mask'))
            tissue_mask = tissue_mask.astype('bool')
            X_train = np.ravel(image[tissue_mask])
            X_train = X_train.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=100, verbose=0,
                                  covariance_type="full")
            gmm.fit(X_train)
            means = [gmm.means_[0][0], gmm.means_[1][0]]
            vals = np.sort(means)
            score = vals[1] / vals[0]
            cnr = get_cnr(image, noise_mask, vals[1], vals[0])
            return pd.DataFrame([vals[0], vals[1], score, cnr]).T

        def seg(image, quantiles, outputdir, dataset):
            """Three-comp segmentation."""
            segmentation, _ = load_image(outputpat.format('segmentation'))
            vals = [np.median(np.ravel(image[segmentation==2])),
                    np.median(np.ravel(image[segmentation==3]))]
            score = vals[1] / vals[0]
            cnr = get_cnr(image, segmentation==3, vals[1], vals[0])
            return pd.DataFrame([vals[0], vals[1], score, cnr]).T

        filestem, ext, outputpat = self._get_outputpat(filepath)

        image, _ = load_image(outputpat.format('data'))
        smooth, props = load_image(outputpat.format('smooth'))

        df = pd.DataFrame()
        metrics = ['q1', 'q2', 'contrast', 'cnr']
        meths = {'q_base': q_base, 'q_clip': q_base, 'q_mask': q_base,
                 'gmm': gmm, 'seg': seg}
        for method, fun in meths.items():
            if method in self.methods:
                df0 = fun(image, self.quantiles, self.directory, filestem)
                df0.columns=['{}-{}'.format(method, metric) for metric in metrics]
                df = pd.concat([df, df0], axis=1)

        df.index = [filestem]

        # NB: need single-file output for HCP distributed system
        outputpath = os.path.join(self.directory, '{}.csv'.format(filestem))
        df.to_csv(outputpath, index_label='sample_id')

        self._metrics = {filestem: df.to_dict(orient='list')}
        basename = self.format_([filestem, self._module_id, self.step])
        outstem = os.path.join(self.directory, basename)
        self.dump_parameters(step=self.step, filestem=outstem)

    def postprocess(self, **kwargs):

        self.set_parameters('postprocess', kwargs)
        arglist = self._get_arglist(['filepaths'])
        self.set_n_workers(len(self.filepaths))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        self._postprocess()

    def _postprocess(self, basename='equalization_assay'):

        self._postprocess_merge(basename)

        if self._compute_env == 'SLURM' or self._compute_env == 'SGE':
            self.n_workers = 1
        else:
            self.set_n_workers(len(self.filepaths))

        arglist = [(filepath,) for filepath in self.filepaths]
        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._postprocess_report, arglist)

        outputstem = os.path.join(self.directory, basename)
        ext = os.path.splitext(self.filepaths[0])[1]

        # Merge csv's.
        pdfs = []
        for filepath in self.filepaths:
            basename = os.path.basename(os.path.splitext(filepath)[0])
            filename = '{}.pdf'.format(basename)
            pdfs.append(os.path.join(self.directory, filename))
        pdfs.sort()
        merge_reports(pdfs, '{}.pdf'.format(outputstem))

        self.summary_report(basename=basename)

    def _postprocess_merge(self, basename='equalization_assay'):

        outputstem = os.path.join(self.directory, basename)
        ext = os.path.splitext(self.filepaths[0])[1]

        # Merge csv's.
        # csvs = [filepath.replace(ext, '.csv') for filepath in self.filepaths]
        csvs = []
        for filepath in self.filepaths:
            basename = os.path.basename(os.path.splitext(filepath)[0])
            filename = '{}.csv'.format(basename)
            csvs.append(os.path.join(self.directory, filename))
        csvs.sort()
        merge_csvs(csvs, '{}.csv'.format(outputstem))
        for csv in csvs:
            os.remove(csv)

        # Aggregate thresholds/metrics from ymls.
        steps = {
            'segment': [
                ['equalization', 'segment', 'params', 'thresholds'],
                ['equalization', 'segment', 'params', '_otsus'],
                ],
            'metrics': [
                ['equalization', 'metrics', 'params', '_metrics'],
                ],
        }
        for step, trees in steps.items():
            ymls = []
            for filepath in self.filepaths:
                filestem = os.path.splitext(os.path.basename(filepath))[0]
                basename = self.format_([filestem, self._module_id, step])
                ymlpath = os.path.join(self.directory, '{}.yml'.format(basename))
                ymls.append(ymlpath)
            ymls.sort()
            bname = self.format_([self.dataset, self._module_id, step])
            ymlstem = os.path.join(self.directory, bname)
            self._merge_parameters(ymls, trees, ymlstem)
            for yml in ymls:
                os.remove(yml)

    def _postprocess_report(self, filepath):
        """Generate reports."""

        filestem, ext = os.path.splitext(os.path.basename(filepath))
        self.report(basename=filestem)

    def set_filepaths(self):
        """Set the filepaths by globbing the directory."""

        # directory = os.path.abspath(self.directory)
        directory = os.path.abspath(os.path.dirname(self.image_in))
        self.filepaths = sorted(glob(os.path.join(directory, self.filepat)))

    def _get_outputpat(self, filepath, ext='.czi'):
        """Get io-info from filepath."""

        filestem = os.path.basename(filepath).split(ext)[0]

        if 'h5' in self.outputformat:
            filepat = '{}.h5/{}'.format(filestem, '{}')
        elif 'nii' in self.outputformat:
            filepat = '{}_{}.nii.gz'.format(filestem, '{}')

        outputpat = os.path.join(self.directory, filepat)

        return filestem, ext, outputpat

    def _get_info_dict(self, filestem, info_dict={}, channel=None):

        basename = os.path.basename(filestem)
        info_dict['filestem'] = basename

        p = self.load_dumped_pars()
        p['threshold_noise']  = p['thresholds'][basename][0]
        p['threshold_tissue'] = p['thresholds'][basename][1]
        p['threshold_otsu']   = p['_otsus'][basename]
        # NOTE: doing this for 'seg' method first => TODO
        p['cnr'] = p['_metrics'][basename]['seg-cnr'][0]
        p['contrast'] = p['_metrics'][basename]['seg-contrast'][0]
        p['median_bg'] = p['_metrics'][basename]['seg-q1'][0]
        p['median_fg'] = p['_metrics'][basename]['seg-q2'][0]
        info_dict['parameters'] = p

        filepath = '{}.h5/data'.format(filestem)
        info_dict['props'] = get_imageprops(filepath)
        info_dict['paths'] = get_paths(filepath)
        info_dict['centreslices'] = get_centreslices(info_dict)

        return info_dict

    def _gen_subgrid(self, f, gs, channel=None):
        """4rows-2 columns: 4 images left, 4 plots right"""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(3, 2)

        tk = {0: 'lc', 1: 'rc'}
        d = {0: ['image', 'tissue / noise regions', 'segmentation'],
             1: ['histogram', 'histogram smoothed image', 'histogram tissue']}

        for j, val in d.items():
            for i, k in enumerate(val):
                axdict[k] = f.add_subplot(gs01[i, j])
                axdict[k].tick_params(axis='both', direction='in')
                titles[k] = (k, tk[j], 0)
                if j:
                    for l in ['top', 'right']:
                        axdict[k].spines[l].set_visible(False)

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict, logscale=True):

        # images
        image = info_dict['centreslices']['data']['z']
        image_smooth = info_dict['centreslices']['smooth']['z']
        segmentation = info_dict['centreslices']['segmentation']['z']

        thrcolors = ['r', 'b', 'g']
        segcolors = [[0, 1, 0], [1, 0, 1], [0, 1, 1]]
        infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
        dmax = infun(image.dtype).max

        # thresholds
        p = info_dict['parameters']
        t = ['noise', 'tissue', 'otsu']
        thresholds = [p['threshold_{}'.format(k)] for k in t]

        # image with scalebar
        x_idx = 2; y_idx = 1;  # FIXME
        w = info_dict['props']['elsize'][x_idx] * image.shape[1]  # note xyz nifti
        h = info_dict['props']['elsize'][y_idx] * image.shape[0]
        extent = [0, w, 0, h]

        ax = axdict['image']
        ax.imshow(image, cmap="gray", extent=extent)
        ax.set_axis_off()
        self._add_scalebar(ax, w)

        # smoothed image with contours at thresholds
        ax = axdict['tissue / noise regions']
        ax.imshow(image_smooth, cmap="gray")
        thrs = [thresholds[0], thresholds[2]]
        cs = ax.contour(image_smooth, thrs, colors=thrcolors[:2], linestyles='dashed')
        ax.clabel(cs, inline=1, fontsize=5)
        labels = ['thr={:.5}'.format(thr) for thr in thrs]
        for i in range(len(labels)):
            cs.collections[i].set_label(labels[i])
        ax.set_axis_off()

        # segmentation image
        ax = axdict['segmentation']
        from skimage import img_as_float
        img = img_as_float(image)
        clabels = label2rgb(segmentation, image=img, alpha=1.0, bg_label=0, colors=segcolors)
        ax.imshow(clabels)
        ax.set_axis_off()

        # full histogram
        ax = axdict['histogram']
        ax.hist(np.ravel(image), bins=256, log=logscale, color=[0, 0, 0])
        ax.set_xlim([0, dmax])

        # smoothed image histogram with thresholds
        ax = axdict['histogram smoothed image']
        ax.hist(np.ravel(image_smooth), bins=256, log=logscale, color=[0, 0, 0])
        linestyles = '--:'
        labels = ['{:.5}'.format(thr) for thr in thresholds]
        self._draw_thresholds(ax, thresholds, thrcolors, linestyles, labels)

        # image histogram from background and signal-of-interest regions
        ax = axdict['histogram tissue']
        data = [np.ravel(image[segmentation == 2]), np.ravel(image[segmentation == 3])]
        ax.hist(data, bins=256, log=logscale, histtype='bar', stacked=True, color=segcolors[1:])
        ax.set_xlim([0, dmax])

        mets = ['cnr', 'contrast', 'median_fg', 'median_bg']
        labs = ['{} = {}'.format(k, '{0: >8.2f}'.format(p[k])) for k in mets]
        lab = '\n'.join(labs)
        ax.annotate(
            text=lab, xy=(1.01, 1.01), c='k',
            xycoords='axes fraction', va='top', ha='right',
            rotation=0, fontsize=7, fontfamily='monospace',
            )

    def _summary_report(self, f, axdict, info_dict):
        """Plot summary report."""

        ax = axdict['graph']
        df = info_dict['df']
        df = df.sort_values('seg-cnr')
        c = plt.cm.rainbow(np.linspace(0, 1, df.shape[0]))
        df['seg-cnr'].plot(ax=ax, kind='barh', color=c)
        ax.set_title("contrast-to-noise", fontsize=12)

    def _get_info_dict_summary(self, filestem, info_dict={}, channel=None):

        info_dict['df'] = pd.read_csv('{}.csv'.format(filestem), index_col='sample_id')

        return info_dict


def load_image(inputpath):
    im = Image(inputpath)
    im.load()
    data = im.ds[:]
    props = im.get_props()
    im.close()
    return data, props


def write_image(data, props):
    props['dtype'] = data.dtype
    mo = Image(**props)
    mo.create()
    mo.write(data)
    mo.close()


def merge_csvs(csvs, outputpath):
    df = pd.DataFrame()
    for csvfile in csvs:
        df0 = pd.read_csv(csvfile)
        df = pd.concat([df, df0], axis=0)
    df = df.set_index('sample_id')
    df.to_csv(outputpath, index_label='sample_id')


if __name__ == "__main__":
    main(sys.argv[1:])
