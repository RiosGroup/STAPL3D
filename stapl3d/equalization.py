#!/usr/bin/env python

"""Calculate metrics for mLSR-3D equalization assay.

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

from stapl3d import parse_args, Stapl3r, Image, transpose_props, get_imageprops, get_paths
from stapl3d.preprocessing import shading
from stapl3d.reporting import get_centreslices

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(argv):
    """Calculate metrics for mLSR-3D equalization assay."""

    steps = ['smooth', 'segment', 'metrics', 'postprocess']
    args = parse_args('equalization', steps, *argv)

    equaliz3r = Equaliz3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        print(f'Running {equaliz3r._module_id}:{step}')
        equaliz3r._fun_selector[step]()


class Equaliz3r(Stapl3r):
    """Calculate metrics for mLSR-3D equalization assay.

    Methods
    ----------
    run
        Run all steps in the equalization module.
    smooth
        Smooth images with a gaussian kernel.
    segment
        Segment the noise and tissue region in the images.
    metrics
        Calculate metrics of equalization images.
    postprocess
        Merge outputs of individual equalization images.
    view
        View equalization image and segmentations with napari.

    Parameters
    ----------
    image_in : string
        Path to dataset.
    parameter_file : string
        Path to yaml parameter file.
    module_id : string
        Name of the STAPL3D module.
    step_id: string
        Identifier of the yaml parameterfile entry.
    directory : string
        Name of output subdirectory.
    prefix : string
        Output prefix.
    max_workers : int
        Maximal number of cores to use for processing.

    Attributes
    ----------
    filepat : string
        Regex to use on the data directory. [needs set_filepaths()]
    filepaths : list
        Paths to files to process.
    sigma : float
        Kernel width for Gaussian smoothing, default 60.
    thresholds : dict
        Specified thresholds for each sample. k: v -> <stem>: [noise, tissue]
    threshold_noise : float
        Specified threshold (global) for noise regions,
        (unused if 'thresholds' is specified).
    threshold_tissue : float
        Specified threshold (global) for tissue regions,
        (unused if 'thresholds' is specified).
    otsu_factor_noise: float
        Otsu multiplication factor for noise threshold, default 0.1,
        (unused if 'thresholds' or 'threshold_noise' is specified).
    otsu_factor_tissue: float
        Otsu multiplication factor for tissue threshold, default 1.1,
        (unused if 'thresholds' or 'threshold_tissue' is specified).
    segment_quantile : float
        Quantile of tissue intensities to split foreground and background,
        default 0.99.
    segment_min_size : int
        Minimum size of connected components of the foreground clusters,
        default 3.
    methods : list
        Quantification methods, default ['seg'].
    quantiles : list
        Quantiles of image intensities for metric calculation,
        not used for the default 'seg' method,
        default [0.50, 0.99].
    use_dirtree : bool
        Switch to derive groupings from directory trees
        or parameterfile, default 'True'. Expected tree structure:
        <root>/<species>/<antibody> with 'primaries' as a separate species.
    metric : str
        Metric to use in the output plots.
    df : pandas dataframe
        Dataframe with quantification of the equalization assay.

    Examples
    --------
    >>> %gui qt

    >>> # fetch and write some testdata
    >>> import os
    >>> datadir = os.path.join('.', 'eqtest')
    >>> os.makedirs(datadir)
    >>> from skimage import data, io
    >>> cells3d = data.cells3d()
    >>> ch = 1
    >>> for slc in range(25, 35):
    >>>     filepath = os.path.join(datadir, f'cells3d_slice{slc}.tif')
    >>>     io.imsave(filepath, cells3d[slc, ch, ...])

    >>> # run equalization
    >>> from stapl3d import equalization
    >>> equaliz3r = equalization.Equaliz3r(filepath)
    >>> equaliz3r.sigma = 10
    >>> equaliz3r.run()

    """

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Equaliz3r, self).__init__(
            image_in, parameter_file,
            module_id='equalization',
            **kwargs,
            )

        self._fun_selector = {
            'smooth': self.smooth,
            'segment': self.segment,
            'metrics': self.metrics,
            'postprocess': self.postprocess,
            }

        self._parallelization = {
            'smooth': ['filepaths'],
            'segment': ['filepaths'],
            'metrics': ['filepaths'],
            'postprocess': [],
            }

        self._parameter_sets = {
            'smooth': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('sigma', '_sigmas'),
                'spar': ('_n_workers', 'filepaths'),
                },
            'segment': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('otsu_factor_noise', 'otsu_factor_tissue',
                         'threshold_noise', 'threshold_tissue',
                         'thresholds', '_otsus',
                         'segment_quantile', 'segment_min_size'),
                'spar': ('_n_workers', 'filepaths',),
                },
            'metrics': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('methods', 'quantiles', '_metrics'),
                'spar': ('_n_workers', 'filepaths',),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': (),
                },
            }

        self._parameter_table = {
            'sigma': 'Smoothing sigma',
            'otsu_factor_noise': 'Multiplication factor for otsu noise threshold',
            'otsu_factor_tissue': 'Multiplication factor for otsu tissue threshold',
            'threshold_noise': 'Global noise threshold',
            'threshold_tissue': 'Global tissue threshold',
            'segment_quantile': 'Quantile used for segmentation',
            'segment_min_size': 'Minimal connected component size',
            'quantiles': 'Quantiles separating signal and tissue background',
            }

        default_attr = {
            'filepat': '*.*',
            'filepaths': [],
            'sigma': 10,
            '_sigmas': {},
            'otsu_factor_noise': 0.1,
            'otsu_factor_tissue': 1.1,
            'thresholds': {},
            'threshold_noise': 0,
            'threshold_tissue': 0,
            '_otsus': {},
            'segment_quantile': 0.95,
            'segment_min_size': 3,
            'methods': ['seg'],
            'quantiles': [0.50, 0.99],
            '_metrics': {},
            'use_dirtree': True,
            'metric': 'foreground',
            'df': pd.DataFrame(),
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._images = ['data', 'smooth']
        self._labels = ['noise_mask', 'tissue_mask', 'segmentation']

    def _init_paths(self):

        vols_d = ['data', 'smooth']
        vols_m = ['noise_mask', 'tissue_mask', 'segmentation']

        fstring = '{}.h5/{}'
        stem = self._build_path()
        fpat = os.path.join('{d}', '{f}')

        self._paths = {
            'smooth': {
                'inputs': {
                    'data': fstring.format(fpat, 'data'),
                    },
                'outputs': {
                    **{ods: fstring.format(fpat, ods) for ods in vols_d},
                    **{'yml': f'{fpat}_smooth'},
                    },
                },
            'segment': {
                'inputs': {
                    ods: fstring.format(fpat, ods) for ods in vols_d
                    },
                'outputs': {
                    **{ods: fstring.format(fpat, ods) for ods in vols_m},
                    **{'yml': f'{fpat}_segment'},
                    **{'stem': f'{fpat}'},
                    },
                },
            'metrics': {
                'inputs': {
                    **{ods: fstring.format(fpat, ods) for ods in vols_d + vols_m},
                    **{'yml': f'{fpat}_segment.yml'},
                    },
                'outputs': {
                    'csv': f'{fpat}.csv',
                    'yml': f'{fpat}_metrics',
                    'report': f'{fpat}.pdf',
                    },
                },
            'postprocess': {
                'inputs': {
                    'data': fstring.format(fpat, 'data'),
                    'csv': f'{fpat}.csv',
                    'yml_smooth': f'{fpat}_smooth.yml',
                    'yml_segment': f'{fpat}_segment.yml',
                    'yml_metrics': f'{fpat}_metrics.yml',
                    'report': f'{fpat}.pdf',
                    },
                'outputs': {
                    'csv': f'{stem}.csv',
                    'yml_smooth': f'{stem}_smooth.yml',
                    'yml_segment': f'{stem}_segment.yml',
                    'yml_metrics': f'{stem}_metrics.yml',
                    'report': f'{stem}.pdf',
                    'summary': f'{stem}_summary.pdf',
                    },
                },
            }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def _get_filepaths_inout(self, filepath):
        """Smooth images with a gaussian kernel."""

        dir, base = os.path.split(filepath)
        filestem = os.path.splitext(base)[0]
        if not self.use_dirtree:
            dir = self.directory
        reps = {'d': dir, 'f': filestem}
        inputs = self._prep_paths(self.inputs, reps=reps)
        outputs = self._prep_paths(self.outputs, reps=reps)

        return filestem, inputs, outputs

    def smooth(self, **kwargs):
        """Smooth images with a gaussian kernel."""

        if not self.filepaths:
            self.set_filepaths()

        arglist = self._prep_step('smooth', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._smooth_image, arglist)

    def _smooth_image(self, filepath):
        """Smooth an image with a gaussian kernel."""

        filestem, inputs, outputs = self._get_filepaths_inout(filepath)

        data, props = load_image(filepath)

        data_smooth = gaussian_filter(data.astype('float'), self.sigma)

        vols = {'data': data, 'smooth': data_smooth}
        for ids, out in vols.items():
            props['path'] = outputs[ids]
            write_image(out, props, attrs={'sigma': self.sigma})

        self._sigmas = {filestem: self.sigma}

        self.dump_parameters(self.step, outputs['yml'])

    def segment(self, **kwargs):
        """Segment the noise and tissue region in the image."""

        if not self.filepaths:
            self.set_filepaths()

        arglist = self._prep_step('segment', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._segment_regions_image, arglist)

    def _segment_regions_image(self, filepath):

        filestem, inputs, outputs = self._get_filepaths_inout(filepath)

        data, props = load_image(inputs['data'])
        smooth, props = load_image(inputs['smooth'])

        segmentation = np.zeros_like(data, dtype='uint8')

        # clipper
        infun = np.iinfo if data.dtype.kind in 'ui' else np.finfo
        clipping_mask = data == infun(data.dtype).max

        # noise / tissue separation
        thresholds = self._pick_thresholds(smooth, filestem)
        noise_mask = smooth < thresholds[0]
        tissue_mask = smooth > thresholds[1]
        tissue_mask = np.logical_and(tissue_mask, ~noise_mask)
        tissue_mask &= ~clipping_mask

        # foreground / background separation
        signal_mask = data > np.quantile(data[tissue_mask], self.segment_quantile)
        signal_mask[~tissue_mask] = False
        signal_mask &= ~clipping_mask
        remove_small_objects(signal_mask, min_size=self.segment_min_size, in_place=True)

        segmentation[noise_mask]  = 1
        segmentation[tissue_mask] = 2
        segmentation[signal_mask] = 3
        segmentation[clipping_mask] = 0

        masks = {
            'noise_mask': [noise_mask, {'threshold_noise': thresholds[0]}],
            'tissue_mask': [tissue_mask, {'threshold_noise': thresholds[0]}],
            'segmentation': [segmentation,
                             {'segment_quantile': self.segment_quantile,
                             'segment_min_size': self.segment_min_size,
                             }],
            }
        for ids, out in masks.items():
            props['path'] = outputs[ids]
            write_image(out[0], props, attrs=out[1])

        self.dump_parameters(self.step, outputs['yml'])

    def _pick_thresholds(self, data, filestem):
        """Switch between threshold options."""

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

        self.thresholds = {filestem: thresholds}
        self._otsus = {filestem: otsu}

        return thresholds

    def metrics(self, **kwargs):
        """Calculate metrics of equalization images."""

        if not self.filepaths:
            self.set_filepaths()

        arglist = self._prep_step('metrics', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._calculate_metrics_image, arglist)

    def _calculate_metrics_image(self, filepath):
        """Calculate metrics of an equalization image."""

        def get_measures(data, quantiles=[0.5, 0.9]):
            quantiles = np.quantile(data, quantiles)
            score = quantiles[1] / quantiles[0]
            return quantiles, score

        def get_snr(image, noise_mask, sigval, bgval=None):
            noise = np.ravel(image[noise_mask])
            if noise.size > 0:
                noise_sd = np.std(noise)
                snr = sigval / noise_sd
                cnr = (sigval - bgval) / noise_sd
            else:
                cnr, noise_sd = 0, 0
            return cnr, noise_sd, snr

        def get_cnr(image, noise_mask, sigval, bgval):
            noise = np.ravel(image[noise_mask])
            if noise.size > 0:
                noise_sd = np.std(noise)
                cnr = (sigval - bgval) / noise_sd
            else:
                cnr, noise_sd = 0, 0
            return cnr, noise_sd

        def q_base(image, quantiles, segpath):
            """Simple quantiles."""
            data = np.ravel(image)
            vals, score = get_measures(data, quantiles)
            return pd.DataFrame([vals[0], vals[1], None, None, None, None]).T

        def q_clip(image, quantiles, segpath):
            """Simple quantiles of non-clipping."""
            infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
            mask = np.logical_and(image > infun(image.dtype).min,
                                  image < infun(image.dtype).max)
            data = np.ravel(image[mask])
            vals, score = get_measures(data, quantiles)
            return pd.DataFrame([vals[0], vals[1], None, None, None, None]).T

        def q_mask(image, quantiles, segpath):
            """Simple quantiles of image[tissue_mask]."""
            segmentation, _ = load_image(segpath)
            tissue_mask = segmentation >= 2
            infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
            mask = np.logical_and(tissue_mask, image < infun(image.dtype).max)
            data = np.ravel(image[mask])
            #mode = stats.mode(data)[0][0]
            vals, score = get_measures(data, quantiles)
            tissue, signal = vals[1], vals[0]

            contrast = tissue / signal
            cnr, noise_sd, snr = get_snr(image, segmentation==1, vals[1], vals[0])

            return pd.DataFrame([vals[0], vals[1], noise_sd, snr, contrast, cnr]).T

        def seg(image, quantiles, segpath):
            """Three-comp segmentation."""

            segmentation, _ = load_image(segpath)

            #fun = self._metric_fun
            fun = np.median
            fun = np.mean

            signal = fun(np.ravel(image[segmentation==2]))
            tissue = fun(np.ravel(image[segmentation==3]))

            contrast = tissue / signal
            cnr, noise_sd, snr = get_snr(image, segmentation==1, tissue, signal)

            return pd.DataFrame([signal, tissue, noise_sd, snr, contrast, cnr]).T

        filestem, inputs, outputs = self._get_filepaths_inout(filepath)

        data, props = load_image(inputs['data'])
        smooth, props = load_image(inputs['smooth'])

        df = pd.DataFrame()
        metrics = ['background', 'foreground', 'noise_sd', 'snr', 'contrast', 'cnr']
        meths = {'q_base': q_base, 'q_clip': q_clip, 'q_mask': q_mask, 'seg': seg}
        for method, fun in meths.items():
            if method in self.methods:
                df0 = fun(data, self.quantiles, inputs['segmentation'])
                df0.columns=['{}-{}'.format(method, metric) for metric in metrics]
                df = pd.concat([df, df0], axis=1)

        df.index = [filestem]

        def stratify_from_dirtree(df, filepath):
            comps = filepath.split(os.sep)
            newcols = {
                'secondaries': comps[-3] != 'primaries',
                'primaries': comps[-3] == 'primaries',
                'antibody': comps[-2],
                'species': comps[-3],
                }
            for k, v in newcols.items():
                df.insert(0, k, v)
            return df

        def stratify_from_parfile(df, filestem):

            def find_primaries(mapping, name):
                for v, prefix in mapping.items():
                    if name.lower().startswith(prefix.lower()):
                        return True, v
                return False, 'None'

            def find_secondaries(mapping, name):
                for species, specdict in mapping.items():
                    for ab, prefix in specdict.items():
                        if name.lower().startswith(prefix.lower()):
                            return True, species, ab
                return False, 'None', 'None'

            if 'primaries' in self._cfg[self.step_id]:
                mapping = self._cfg[self.step_id]['primaries']
                in_primaries, antibody = find_primaries(mapping, filestem)
                newcols = {
                    'secondaries': False,
                    'primaries': in_primaries,
                    'antibody': antibody,
                    'species': 'primaries',
                    }
                for k, v in newcols.items():
                    df.insert(0, k, v)

            if 'secondaries' in self._cfg[self.step_id]:
                mapping = self._cfg[self.step_id]['secondaries']
                in_secondaries, species, antibody = find_secondaries(mapping, filestem)
                newcols = {'secondaries': in_secondaries}
                if in_secondaries:
                    newcols['antibody'] = antibody
                    newcols['species'] = species
                for k, v in newcols.items():
                    df.insert(0, k, v)

            return df

        if self.use_dirtree:
            df = stratify_from_dirtree(df, filepath)
        elif self.step_id in self._cfg.keys():
            df = stratify_from_parfile(df, filestem)

        # NB: need single-file output for HCP distributed system
        cols = list(df.columns)

        df = df[cols]
        df.to_csv(outputs['csv'], index_label='sample_id')

        self._metrics = {filestem: df.to_dict(orient='list')}
        self.dump_parameters(self.step, outputs['yml'])

        # NOTE: doing this for 'seg' method only => TODO
        inputstem = inputs['data'].replace('.h5/data', '')
        pars = self._collect_parameters(inputstem)
        self.report(
            outputpath=outputs['report'],
            name=filestem, filestem=filestem,
            inputs=inputs, outputs=outputs,
            parameters=pars,
            )

    def postprocess(self, **kwargs):
        """Merge outputs of individual equalization images."""

        if not self.filepaths:
            self.set_filepaths()

        arglist = self._prep_step('postprocess', kwargs)
        self._postprocess()

    def _postprocess(self, basename='equalization_assay'):
        """Merge outputs of individual equalization images."""

        def get_filelist(filetype):
            filelist = [self._get_filepaths_inout(filepath)[1][filetype]
                        for filepath in self.filepaths]
            filelist.sort()
            return filelist

        outputs = self._prep_paths(self.outputs)

        steps = {
            'smooth': [
                ['equalization', 'smooth', 'params', '_sigmas'],
                ],
            'segment': [
                ['equalization', 'segment', 'params', 'thresholds'],
                ['equalization', 'segment', 'params', '_otsus'],
                ],
            'metrics': [
                ['equalization', 'metrics', 'params', '_metrics'],
                ],
        }
        for step, trees in steps.items():
            self._merge(
                get_filelist(f'yml_{step}'),
                outputs[f'yml_{step}'],
                merge_parameters,
                trees=trees,
                )
        pars = self.load_dumped_pars()
        self._sigmas = pars['_sigmas']
        self._otsus = pars['_otsus']
        self.thresholds = pars['thresholds']
        self._metrics = pars['_metrics']

        self._merge(get_filelist('csv'), outputs['csv'], merge_csvs)

        self._merge(get_filelist('report'), outputs['report'], self._merge_reports)

        self.df = pd.read_csv(outputs['csv'], index_col='sample_id')

        self.summary_report(outputpath=outputs['summary'])

    def set_filepaths(self):
        """Set the filepaths by globbing the directory."""

        if not self.use_dirtree:
            filepat = self.filepat
            self.filepaths = sorted(glob(os.path.join(self.datadir, filepat)))
        else:
            filepat = os.path.join('*', '*', self.filepat)
            self.filepaths = sorted(glob(os.path.join(self.datadir, filepat)))
            if not self.filepaths:
                filepat = self.filepat
                self.filepaths = sorted(glob(os.path.join(self.datadir, filepat)))
                if self.filepaths:
                    self.use_dirtree = False

    def _collect_parameters(self, inputstem, steps=['smooth', 'segment', 'metrics']):
        """Collect parameters of steps for file-specific report generation."""

        pars = {}
        for step in steps:
            ymlpath = f'{inputstem}_{step}.yml'
            with open(ymlpath, 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            pars = {**pars, **cfg[self._module_id][step]['params']}

        return pars

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        filestem = kwargs['filestem']

        kwargs['sigma']            = p['_sigmas'][filestem]

        kwargs['threshold_noise']  = p['thresholds'][filestem][0]
        kwargs['threshold_tissue'] = p['thresholds'][filestem][1]
        kwargs['threshold_otsu']   = p['_otsus'][filestem]

        method = self.methods[0]
        kwargs['method']           = method
        kwargs['metric']           = self.metric
        kwargs['cnr']              = p['_metrics'][filestem][f'{method}-cnr'][0]
        kwargs['contrast']         = p['_metrics'][filestem][f'{method}-contrast'][0]
        kwargs['background']       = p['_metrics'][filestem][f'{method}-background'][0]
        kwargs['foreground']       = p['_metrics'][filestem][f'{method}-foreground'][0]

        filepath = kwargs['inputs']['data']
        kwargs['props'] = get_imageprops(filepath)
        kwargs['paths'] = get_paths(filepath)
        kwargs['centreslices'] = get_centreslices(kwargs)

        return kwargs

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
        # p = info_dict['parameters']
        t = ['noise', 'tissue', 'otsu']
        thresholds = [info_dict['threshold_{}'.format(k)] for k in t]

        # image with scalebar
        x_idx = 2; y_idx = 1;  # FIXME
        w = info_dict['props']['elsize'][x_idx] * image.shape[1]  # note xyz nifti
        h = info_dict['props']['elsize'][y_idx] * image.shape[0]
        extent = [0, w, 0, h]

        ax = axdict['image']
        ax.imshow(image, cmap="gray", extent=extent, vmin=0, vmax=info_dict['foreground'])
        ax.set_axis_off()
        self._add_scalebar(ax, w)

        # smoothed image with contours at thresholds
        ax = axdict['tissue / noise regions']
        ax.imshow(image_smooth, cmap="gray")
        cs = ax.contour(image_smooth, thresholds[:2], colors=thrcolors[:2], linestyles='dashed')
        ax.clabel(cs, inline=1, fontsize=5)
        labels = ['thr={:.5}'.format(float(thr)) for thr in thresholds[:2]]
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
        if info_dict['threshold_otsu'] is None:
            thresholds = thresholds[:2]
            linestyles = linestyles[:2]
            thrcolors = thrcolors[:2]
        labels = ['{:.5}'.format(float(thr)) for thr in thresholds]
        self._draw_thresholds(ax, thresholds, thrcolors, linestyles, labels)

        # image histogram from background and signal-of-interest regions
        ax = axdict['histogram tissue']
        data = [np.ravel(image[segmentation == 2]), np.ravel(image[segmentation == 3])]
        ax.hist(data, bins=256, log=logscale, histtype='bar', stacked=True, color=segcolors[1:])
        ax.set_xlim([0, dmax])

        mets =  ['foreground', 'background', 'contrast', 'cnr']
        vals = {
            **{'method': '{0: >8s}'.format(info_dict['method'])},
            **{k:        '{0: >8.2f}'.format(info_dict[k]) for k in mets},
            }
        labs = ['{} = {}'.format(k, v) for k, v in vals.items()]
        lab = '\n'.join(labs)
        ax.annotate(
            text=lab, xy=(1.01, 1.01), c='k',
            xycoords='axes fraction', va='top', ha='right',
            rotation=0, fontsize=7, fontfamily='monospace',
            )

    def _summary_report(self, f, axdict, info_dict):
        """Plot summary report."""

        try:
            self._plot_stratified(f, axdict, info_dict)
        except (KeyError, IndexError):
            plt.clf()
            self._plot_basic(f, axdict, info_dict)

    def _get_info_dict_summary(self, filestem, info_dict={}, channel=None):

        info_dict['method']           = self.methods[0]
        info_dict['metric']           = self.metric

        filepath = self._abs(self.outputs['csv'].format(filestem))
        info_dict['df'] = pd.read_csv(filepath, index_col='sample_id')

        return info_dict

    def _plot_basic(self, f, axdict, info_dict):

        sortcol = '{}-{}'.format(info_dict['method'], info_dict['metric'])

        gs = axdict['gs']
        ax = f.add_subplot(gs[0].subgridspec(1, 3)[1:])

        df = info_dict['df']
        df = df.sort_values(sortcol)
        c = plt.cm.rainbow(np.linspace(0, 1, df.shape[0]))
        df[sortcol].plot(ax=ax, kind='barh', color=c)
        ax.set_title(info_dict['metric'], fontsize=12)

    def _plot_stratified(self, f, axdict, info_dict):

        sortcol = '{}-{}'.format(info_dict['method'], info_dict['metric'])

        df_p = self.df[self.df['primaries']]
        df_s = self.df[self.df['secondaries']]

        dfa = df_s.groupby(['species', 'antibody']).agg([np.mean, np.std])
        n_species = dfa.index.get_level_values(0).nunique()

        gs = axdict['gs']
        gs01 = gs[0].subgridspec(3, n_species, height_ratios=[5, 1, 5])

        axdict['graph1'] = f.add_subplot(gs01[0, :])
        self._plot_primaries(metric=sortcol, ax=axdict['graph1'])

        axs = []
        for ab in range(n_species):
            if len(axs) == 0:
                axs.append(f.add_subplot(gs01[2, ab]))
            else:
                axs.append(f.add_subplot(gs01[2, ab], sharey=axs[-1]))
        axdict['graph2'] = axs
        self._plot_secondaries(metric=sortcol, axs=axdict['graph2'])

    def _plot_primaries(self, metric='seg-foreground', ax=None):

        df = self.df[self.df['primaries']]
        dfa = df.groupby('antibody').agg([np.mean, np.std])
        dfb = dfa.xs(metric, axis=1).sort_values('mean', ascending=False)

        if ax is not None:
            dfb.plot(kind="bar", y="mean", yerr='std',
                     legend=False, title="Primaries", ax=ax)
            ax.set_ylabel(self.metric)
        else:
            try:
                import seaborn as sns
            except ImportError:
                dfb.plot(kind="bar", y="mean", yerr='std',
                         legend=False, figsize=(16, 10), title="Primaries",
                         ax=ax)
                ax.set_ylabel(self.metric)
            else:
                g = sns.catplot(
                    kind='bar',
                    x='antibody',
                    y=metric,
                    data=df,
                    estimator=np.mean,
                    ci='sd',
                    order=dfb.index,
                    height=8,
                    aspect=2,
                )

                g.set_xticklabels(rotation=60, fontsize=15)
                g.set_yticklabels(fontsize=15)
                g.set_axis_labels('', 'intensity', fontsize=20)

    def _plot_secondaries(self, metric='seg-foreground', axs=None):

        df = self.df[self.df['secondaries']]
        dfa = df.groupby(['species', 'antibody']).agg([np.mean, np.std])
        dfb = dfa.xs(metric, axis=1).dropna().sort_values('mean', ascending=False)

        dfb.unstack(level=0).plot(
            kind='bar',
            y="mean",
            yerr='std',
            subplots=True,
            legend=False,
            ax=axs,
        )
        axs[0].set_ylabel(self.metric)

        #sns.catplot(kind='bar', data=df, col='species', estimator=np.mean, ci='sd', col_wrap=4)

    def view(self, input=[], images=None, labels=None, settings={}):
        """View equalization image and segmentations with napari."""

        filepath = input or self.filepaths[0]
        filepath = filepath.replace(os.path.splitext(filepath)[-1], '.h5')

        super().view(filepath, images, labels, settings)


def load_image(inputpath):

    im = Image(inputpath)
    im.load()
    data = im.ds[:]
    props = im.get_props()
    im.close()

    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
        props['shape'] = [1] + props['shape']
        props['elsize'] = [1] + props['elsize']
        props['axlab'] = 'z' + props['axlab']
        props['slices'] = [slice(0, 1)] + props['slices']

    return data, props


def write_image(data, props, attrs={}):

    props['dtype'] = data.dtype
    mo = Image(**props)
    mo.create()
    mo.write(data)
    for attr, val in attrs.items():
        mo.ds.attrs[attr] = val
    mo.close()


def merge_csvs(csvs, outputpath):
    df = pd.DataFrame()
    for csvfile in csvs:
        df0 = pd.read_csv(csvfile)
        df = pd.concat([df, df0], axis=0)
    df = df.set_index('sample_id')
    df.to_csv(outputpath, index_label='sample_id')


def merge_parameters(ymls, outputpath, **kwargs):
    """Merge specific parameters from a set of yml files."""

    trees = kwargs['trees']

    def replace(d, path, replacement):
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = replacement

    with open(ymls[0], 'r') as ymlfile:
        cfg_out = yaml.safe_load(ymlfile)

    for tree in trees:
        aggr = {}
        for ymlpath in ymls:
            with open(ymlpath, 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            for item in tree:
                cfg = cfg[item]
            # FIXME: abusing instance for aggregationVSreplace switch
            if isinstance(cfg, dict):
                aggr.update(cfg)
            else:
                aggr = cfg
        replace(cfg_out, tree, aggr)

    if outputpath:
        with open(outputpath, 'w') as f:
            yaml.dump(cfg_out, f, default_flow_style=False)

    return yaml.dump(cfg_out, default_flow_style=False)


if __name__ == "__main__":
    main(sys.argv[1:])
