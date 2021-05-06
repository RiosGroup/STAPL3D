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
from stapl3d.reporting import merge_reports, get_centreslices

logger = logging.getLogger(__name__)


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
        equaliz3r._fun_selector[step]()


class Equaliz3r(Stapl3r):
    """Calculate metrics for mLSR-3D equalization assay."""

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
                'ppar': ('sigma',),
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
            'otsu_factor_noise': 'Multiplication factor for noise threshold',
            'otsu_factor_tissue': 'Multiplication factor for tissue threshold',
            'segment_quantile': 'Quantile used for segmentation',
            'segment_min_size': 'Minimal connected component size',
            'quantiles': 'Quantiles separating signal and tissue background',
            }

        default_attr = {
            'filepat': '*.*',
            'filepaths': [],
            'outputformat': '.h5',
            'sigma': 60,
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
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

    def _init_paths(self):

        self.set_filepaths()

        vols_d = ['data', 'smooth']
        vols_m = ['noise_mask', 'tissue_mask', 'segmentation']

        if 'h5' in self.outputformat:
            fstring = '{}.h5/{}'
        elif 'nii' in self.outputformat:
            fstring = '{}_{}.nii.gz'

        stem = self._build_path()
        fpat = self._build_path(suffixes=[{'f': 'p'}])

        self._paths = {
            'smooth': {
                'inputs': {
                    'data': f'{fpat}',
                    },
                'outputs': {
                    ods: fstring.format(fpat, ods) for ods in vols_d
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
                    'yml_segment': f'{fpat}_segment.yml',
                    'yml_metrics': f'{fpat}_metrics.yml',
                    'report': f'{fpat}.pdf',
                    },
                'outputs': {
                    'csv': f'{stem}.csv',
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

    def smooth(self, **kwargs):
        """Smooth images with a gaussian kernel."""

        arglist = self._prep_step('smooth', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._smooth_image, arglist)

    def _smooth_image(self, filepath):
        """Smooth an image with a gaussian kernel."""

        filestem = os.path.splitext(os.path.basename(filepath))[0]
        inputs = self._prep_paths(self.inputs, reps={'f': filestem})
        outputs = self._prep_paths(self.outputs, reps={'f': filestem})

        data, props = load_image(filepath)

        data_smooth = gaussian_filter(data.astype('float'), self.sigma)

        if 'nii' in self.outputformat:
            props = transpose_props(props)
            data = np.atleast_3d(data)
            data_smooth = np.atleast_3d(data_smooth)

        vols = {'data': data, 'smooth': data_smooth}
        for ids, out in vols.items():
            props['path'] = outputs[ids]
            write_image(out, props)

    def segment(self, **kwargs):
        """Segment the noise and tissue region in the image."""

        arglist = self._prep_step('segment', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._segment_regions_image, arglist)

    def _segment_regions_image(self, filepath):

        filestem = os.path.splitext(os.path.basename(filepath))[0]
        inputs = self._prep_paths(self.inputs, reps={'f': filestem})
        outputs = self._prep_paths(self.outputs, reps={'f': filestem})

        data, props = load_image(inputs['data'])
        smooth, props = load_image(inputs['smooth'])

        masks, otsu, thrs = self._region_masks(smooth, filestem)
        masks = self._tissue_segmentation(data, masks)

        for ids, out in masks.items():
            props['path'] = outputs[ids]
            write_image(out, props)

        self.thresholds = {filestem: thrs}
        self._otsus = {filestem: otsu}

        self.dump_parameters(self.step, outputs['yml'])

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
        signal_mask[~tissue_mask] = False
        signal_mask &= ~clipping_mask
        remove_small_objects(signal_mask, min_size=self.segment_min_size, in_place=True)

        segmentation = np.zeros_like(tissue_mask, dtype='uint8')
        segmentation[noise_mask]  = 1
        segmentation[tissue_mask] = 2
        segmentation[signal_mask] = 3
        segmentation[clipping_mask] = 0

        masks['segmentation'] = segmentation

        return masks

    def metrics(self, **kwargs):

        arglist = self._prep_step('metrics', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._calculate_metrics_image, arglist)

    def _calculate_metrics_image(self, filepath):

        def get_measures(data, quantiles=[0.5, 0.9]):
            quantiles = np.quantile(data, quantiles)
            score = quantiles[1] / quantiles[0]
            return quantiles, score

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
            cnr, noise_sd = None, None
            return pd.DataFrame([vals[0], vals[1], score, cnr, noise_sd]).T

        def q_clip(image, quantiles, segpath):
            """Simple quantiles of non-clipping."""
            infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
            mask = np.logical_and(image > infun(image.dtype).min,
                                  image < infun(image.dtype).max)
            data = np.ravel(image[mask])
            vals, score = get_measures(data, quantiles)
            cnr, noise_sd = None, None
            return pd.DataFrame([vals[0], vals[1], score, cnr, noise_sd]).T

        def q_mask(image, quantiles, segpath):
            """Simple quantiles of image[tissue_mask]."""
            segmentation, _ = load_image(segpath)
            tissue_mask = segmentation >= 2
            infun = np.iinfo if image.dtype.kind in 'ui' else np.finfo
            mask = np.logical_and(tissue_mask, image < infun(image.dtype).max)
            data = np.ravel(image[mask])
            #mode = stats.mode(data)[0][0]
            vals, score = get_measures(data, quantiles)
            cnr, noise_sd = get_cnr(image, segmentation==1, vals[1], vals[0])
            return pd.DataFrame([vals[0], vals[1], score, cnr, noise_sd]).T

        def seg(image, quantiles, segpath):
            """Three-comp segmentation."""
            segmentation, _ = load_image(segpath)
            signal = np.median(np.ravel(image[segmentation==2]))
            tissue = np.median(np.ravel(image[segmentation==3]))
            contrast = tissue / signal
            cnr, noise_sd = get_cnr(image, segmentation==1, tissue, signal)
            return pd.DataFrame([signal, tissue, contrast, cnr, noise_sd]).T

        filestem = os.path.splitext(os.path.basename(filepath))[0]
        inputs = self._prep_paths(self.inputs, reps={'f': filestem})
        outputs = self._prep_paths(self.outputs, reps={'f': filestem})

        data, props = load_image(inputs['data'])
        smooth, props = load_image(inputs['smooth'])

        df = pd.DataFrame()
        metrics = ['q1', 'q2', 'contrast', 'cnr', 'noise_sd']
        meths = {'q_base': q_base, 'q_clip': q_clip, 'q_mask': q_mask, 'seg': seg}
        for method, fun in meths.items():
            if method in self.methods:
                df0 = fun(data, self.quantiles, inputs['segmentation'])
                df0.columns=['{}-{}'.format(method, metric) for metric in metrics]
                df = pd.concat([df, df0], axis=1)

        df.index = [filestem]

        # NB: need single-file output for HCP distributed system
        df.to_csv(outputs['csv'], index_label='sample_id')

        self._metrics = {filestem: df.to_dict(orient='list')}
        self.dump_parameters(self.step, outputs['yml'])

        # Pars of this step
        pars = self._step_pars(self._parameter_sets[self.step], vars(self))['params']
        # Add 'sigma' parameter of previous step
        pars = self._load_dumped_step(self._module_id, self._module_id, 'smooth', pars)
        # Load thresholds from filestem-specific yml
        with open(inputs['yml'], 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        pars = {**pars, **cfg['equalization']['segment']['params']}

        # NOTE: doing this for 'seg' method only => TODO
        self.report(outputpath=outputs['report'],
                    name=filestem, filestem=filestem,
                    inputs=inputs, outputs=outputs,
                    parameters=pars,
                    # threshold_noise=pars['thresholds'][filestem][0],
                    # threshold_tissue=pars['thresholds'][filestem][1],
                    # threshold_otsu=pars['_otsus'][filestem],
                    # cnr=self._metrics[filestem]['seg-cnr'][0],
                    # contrast=self._metrics[filestem]['seg-contrast'][0],
                    # median_bg=self._metrics[filestem]['seg-q1'][0],
                    # median_fg=self._metrics[filestem]['seg-q2'][0],
                    )

    def postprocess(self, **kwargs):

        arglist = self._prep_step('postprocess', kwargs)
        self._postprocess()

    def _postprocess(self, basename='equalization_assay'):

        outputs = self._prep_paths(self.outputs)

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
            self._merge(f'yml_{step}', merge_parameters, trees=trees)

        self._merge('csv', merge_csvs)

        self._merge('report', merge_reports)

        self.summary_report(outputpath=outputs['summary'])

    def set_filepaths(self):
        """Set the filepaths by globbing the directory."""

        # directory = os.path.abspath(self.directory)
        directory = os.path.abspath(os.path.dirname(self.image_in))
        self.filepaths = sorted(glob(os.path.join(directory, self.filepat)))

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        filestem = kwargs['filestem']
        kwargs['threshold_noise']  = p['thresholds'][filestem][0]
        kwargs['threshold_tissue'] = p['thresholds'][filestem][1]
        kwargs['threshold_otsu']   = p['_otsus'][filestem]
        # # NOTE: doing this for 'seg' method first => TODO
        kwargs['cnr']              = p['_metrics'][filestem]['seg-cnr'][0]
        kwargs['contrast']         = p['_metrics'][filestem]['seg-contrast'][0]
        kwargs['median_bg']        = p['_metrics'][filestem]['seg-q1'][0]
        kwargs['median_fg']        = p['_metrics'][filestem]['seg-q2'][0]

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
        labs = ['{} = {}'.format(k, '{0: >8.2f}'.format(info_dict[k])) for k in mets]
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

        filepath = self._abs(self.outputs['csv'].format(filestem))
        info_dict['df'] = pd.read_csv(filepath, index_col='sample_id')

        return info_dict

    def view_with_napari(self, filepath='', idss=['data', 'smooth'], ldss=['noise_mask', 'tissue_mask', 'segmentation']):

        if not filepath:
            filestem = os.path.splitext(os.path.basename(self.filepaths[0]))[0]
            outputs = self._prep_paths(self.outputpaths['segment'], reps={'f': filestem})
            input = outputs['stem']
            filepath = f'{input}.h5'
        elif filepath.endswith('.czi'):
            filestem = os.path.splitext(os.path.basename(filepath))[0]
            outputs = self._prep_paths(self.outputpaths['segment'], reps={'f': filestem})
            input = outputs['stem']
            filepath = f'{input}.h5'

        super().view_with_napari(filepath, idss, ldss)


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
