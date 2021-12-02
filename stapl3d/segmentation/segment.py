#!/usr/bin/env python

"""Segment cells from membrane and nuclear channels.

    # TODO: make python=3.7 a requirement (we want OrderedDict)

"""

import os
import sys
import logging
import pickle
import shutil
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from glob import glob

import time

from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes

from skimage.color import label2rgb
from skimage.transform import resize
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import (
    gaussian,
    median,
    threshold_sauvola,
    )
from skimage.morphology import (
    label,
    remove_small_objects,
    opening,
    binary_opening,
    binary_closing,
    binary_dilation,
    binary_erosion,
    ball,
    disk,
    )

from stapl3d import parse_args, Stapl3r, Image, LabelImage, MaskImage, wmeMPI, transpose_props
from stapl3d.blocks import Block3r
from stapl3d import get_paths  # TODO: into Image/Stapl3r
from stapl3d.reporting import (
    gen_orthoplot,
    load_parameters,
    get_centreslice,
    get_centreslices,
    get_zyx_medians,
    get_cslc,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Segment cells from membrane and nuclear channels."""

    steps = ['estimate']  # , 'subsegment'
    args = parse_args('segmentation', steps, *argv)

    segment3r = Segment3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        segment3r._fun_selector[step]()


class Segment3r(Block3r):
    """Segment cells from membrane and nuclear channels."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'segmentation'

        super(Segment3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'estimate': self.estimate,
            })

        self._parallelization.update({
            'estimate': ['blocks'],
            })

        self._parameter_sets.update({
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        # TODO
        self._parameter_table.update({
            })

        # TODO
        default_attr = {
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_segmenter()

        self._init_log()

        self._prep_blocks()

        self._images = ['nucl/prep']
        self._labels = ['segm/labels_full', 'segm/labels_nucl', 'segm/labels_memb', 'segm/labels_csol']

    def _init_paths_segmenter(self):

        prev_path = {
            'moduledir': 'membrane_enhancement', 'module_id': 'membrane_enhancement',
            'step_id': 'membrane_enhancement', 'step': 'estimate',
            'ioitem': 'outputs', 'output': 'blockfiles',
            }
        bpat = self._get_inpath(prev_path)
        if bpat == 'default':
            os.makedirs('blocks', exist_ok=True)
            bpat = self._build_path(moduledir='blocks', prefixes=[self.prefix, 'blocks'], suffixes=[{'b': 'p'}], ext='h5')

        self._paths.update({
            'estimate': {
                'inputs': {
                    'blockfiles': f'{bpat}',
                    },
                'outputs': {
                    'blockfiles': f'{bpat}',
                    'report': f'{bpat}'.replace('.h5', '.pdf'),
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def estimate(self, **kwargs):
        """Segment cells from membrane and nuclear channels."""

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, block):
        """Segment cells from membrane and nuclear channels."""

        block = self._blocks[block]
        inputs = self._prep_paths(self.inputs, reps={'b': block.idx})
        outputs = self._prep_paths(self.outputs, reps={'b': block.idx})

        step = 'estimate'
        params = self._cfg[self.step_id][step]

        filepath = inputs['blockfiles']
        save_steps = True

        fundict = {
            'prep': prep_volume,
            'mask': mask_volume,
            'combine': combine_volumes,
            'seed': seed_volume,
            'segment': segment_volume,
            'filter': filter_segments,
            'iterative_split': resegment_largelabels,
        }

        for step_key, pars in params.items():

            t = time.time()

            # im = fundict[](filepath, step_key, pars, save_steps)
            if step_key.startswith('prep'):
                im = prep_volume(filepath, step_key, pars, save_steps)
            elif step_key.startswith('mask'):
                im = mask_volume(filepath, step_key, pars, save_steps)
            elif step_key.startswith('combine'):
                im = combine_volumes(filepath, step_key, pars, save_steps)
            elif step_key == 'seed':
                im = seed_volume(filepath, step_key, pars, save_steps)
            elif step_key == 'segment':
                im = segment_volume(filepath, step_key, pars, save_steps)
            elif step_key == 'filter':
                im = filter_segments(filepath, step_key, pars, save_steps)
            elif step_key == 'iterative_split':
                im = resegment_largelabels(filepath, step_key, pars, save_steps)

            elapsed = time.time() - t
            print('{} took {:1f} s'.format(step_key, elapsed))

        # self.dump_parameters(self.step, outputs['parameters'])
        self.report(outputs['report'], inputs=inputs, outputs=outputs)

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        filepath = kwargs['outputs']['blockfiles']
        # kwargs['props'] = get_imageprops(filepath)
        kwargs['paths'] = get_paths(f'{filepath}/mean')
        kwargs['centreslices'] = get_centreslices(kwargs)

        return kwargs

    def _gen_subgrid(self, f, gs, channel=None):
        """Generate the axes for printing the report."""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(4, 2)

        # TODO: see what is available in info_dict, or force saving intermediates, or gather cslcs in over the process
        t = [
            'nucl', 'memb',
            'nucl_mask', 'memb_mask',
            'seeds_mask', 'seeds_mask_dil',
            'segm_seeds', 'segm',
            ]
        voldict = {name: {'idx': i, 'title': name} for i, name in enumerate(t)}

        for k, vd in voldict.items():
            titles[k] = (vd['title'], 'lcm', 0)
            axdict[k] = gen_orthoplot(f, gs01[vd['idx']], 5, 1)

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict={}):
        """Plot graph with shading image."""

        def plot_imgs(img, labels, axdict, axkey, axidx, aspect, cmap='gray', alpha=0.3, colors=None):

            ax = axdict[axkey][axidx]
            ax.axis('off')

            if img is None and labels is None:
                return
            elif img is None:
                img = np.zeros(labels.shape)

            if dim == 'x':
                img = img.transpose()

            if labels is not None:
                if dim == 'x':
                    labels = labels.transpose()
                data = label2rgb(labels, image=img, alpha=alpha, bg_label=0, colors=colors)
            else:
                data = img

            ax.imshow(data, aspect=aspect, cmap=cmap)

        def get_data(prefered, fallback, dimfac):
            try:
                slc = centreslices[prefered][dim] * dimfac
                return slc
            except (TypeError, KeyError):
                print('{} not found: falling back to {}'.format(prefered, fallback))
                try:
                    slc = centreslices[fallback][dim] * dimfac
                    return slc
                except (TypeError, KeyError):
                    print('{} not found: falling back to empty'.format(fallback))
                    # TODO: empty image of correct dim
                    return None

        centreslices = info_dict['centreslices']

        # aspects = ['equal', 'equal', 'equal']
        aspects = ['equal', 'auto', 'auto']
        for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

            data_nucl = get_data('nucl/prep', 'nucl/mean', dimfac=3)
            data_memb = get_data('memb/prep', 'memb/mean', dimfac=5)

            plot_imgs(data_nucl, None, axdict, 'nucl', i, aspect, 'gray')
            plot_imgs(data_memb, None, axdict, 'memb', i, aspect, 'gray')

            # segmentation:mask_dset:ods_mask
            # segmentation:mask_nucl:ods_mask
            try:
                labels = centreslices['mask'][dim].astype('uint8')
                labels[centreslices['nucl/mask'][dim]] = 2
                plot_imgs(data_nucl, labels, axdict, 'nucl_mask', i, aspect, None, 0.5, colors=[[1, 0, 0], [0, 1, 0]])
            except (TypeError, KeyError):
                ax = axdict['nucl_mask'][i]
                ax.axis('off')

            # segmentation:mask_dset:ods_mask
            # segmentation:mask_memb:ods_mask
            try:
                labels = centreslices['mask'][dim].astype('uint8')
                labels[centreslices['memb/mask'][dim]] = 2
                plot_imgs(data_memb, labels, axdict, 'memb_mask', i, aspect, None, 0.5, colors=[[1, 0, 0], [0, 1, 0]])
            except (TypeError, KeyError):
                ax = axdict['memb_mask'][i]
                ax.axis('off')

            # segmentation:seed:ids_mask
            try:
                labels = centreslices['segm/seeds_mask'][dim].astype('uint8')
                labels[centreslices['segm/seeds_mask_dil'][dim]] = 2
                plot_imgs(data_nucl, labels, axdict, 'seeds_mask', i, aspect, None, 0.5, colors=[[1, 0, 0], [0, 1, 0]])
            except (TypeError, KeyError):
                ax = axdict['seeds_mask'][i]
                ax.axis('off')

            try:
                data_edt = centreslices['segm/seeds_mask_edt'][dim]
                labels = centreslices['segm/seeds_mask_dil'][dim]
                plot_imgs(data_edt, labels, axdict, 'seeds_mask_dil', i, aspect, None, 0.5, colors=None)
            except (TypeError, KeyError):
                ax = axdict['seeds_mask_dil'][i]
                ax.axis('off')

            # segmentation:seed:ods_labels
            try:
                labels = centreslices['segm/labels_edt'][dim]
                plot_imgs(data_nucl, labels, axdict, 'segm_seeds', i, aspect, None, 0.7, colors=None)
            except (TypeError, KeyError):
                ax = axdict['segm_seeds'][i]
                ax.axis('off')

            # segmentation:filter:ods_labels
            try:
                labels = centreslices['segm/labels'][dim]
                plot_imgs(data_memb, labels, axdict, 'segm', i, aspect, None, 0.3, colors=None)
            except (TypeError, KeyError):
                ax = axdict['segm'][i]
                ax.axis('off')

    def view(self, input=[], images=[], labels=[], settings={}):

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, (int, float)):
            input = self._blocks[input].path.replace('/{ods}', '')
        else:
            input = input or [0, 1]

        super().view(input, images, labels, settings)


def prep_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()

    try:
        p = pars['downsample']
        print(p)
    except KeyError:
        data = im.slice_dataset()
        im.close()
    else:
        im2 = im.downsampled(p['factors'])
        data = im2.slice_dataset()
        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im2)
        im2.close()
        im = im2


    # if 'shift_planes' in pars.keys():
    #     p = pars['shift_planes']
    try:
        p = pars['shift_planes']
        print(p)
    except KeyError:
        pass
    else:
        data = shift_channel(data, n_planes=p['n_planes'])
        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im)

    try:
        p = pars['opening']
        print(p)
    except KeyError:
        pass
    else:
        try:
            selem = p['selem']
        except KeyError:
            selem = None
        data = opening(data, selem=selem, out=data)
        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im)

    try:
        p = pars['filter']
        print(p)
    except KeyError:
        pass
    else:

        if p['type'] in ['median', 'gaussian']:
            if p['inplane']:
                data = smooth_channel_inplane(data, p['sigma'], p['type'])
            else:
                # TODO
                data = smooth_channel(data, p['sigma'], p['type'])
        elif p['type'] == 'dog':
            data = smooth_dog(data, im.elsize, p['sigma1'], p['sigma2'])
            data = data.astype('float')

        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im)

    im = write(data, '{}/'.format(filepath), pars['ods_image'], im, 'Image')

    return im


def mask_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    try:
        p = pars['threshold']
        print(p)
    except KeyError:
        pass
    else:
        if pars['threshold'] is None:
            from skimage.filters import threshold_otsu
            thr = threshold_otsu(data)
        else:
            thr = pars['threshold']
        print(f'thresholding at {thr} for {filepath}')
        mask = data > thr

    try:
        p = pars['sauvola']
        print(p)
    except KeyError:
        pass
    else:

        if 'absmin' in p.keys():
            mask = data > p['absmin']
        else:
            mask = np.zeros_like(data, dtype='bool')

        if 'k' in p.keys():
            thr = threshold_sauvola(data, window_size=p['window_size'], k=p['k'])
            mask_sauvola = data > thr
            mask &= mask_sauvola

        if 'threshold' in p.keys():
            mask_threshold = data > p['threshold']
            mask_threshold = binary_closing(mask_threshold)
            mask |= mask_threshold

    try:
        p = pars['fill']
        print(p)
    except KeyError:
        pass
    else:
        binary_fill_holes(mask, output=mask)
        if 'postfix' in p.keys(): write(mask, image_in, p['postfix'], im, 'Mask')

    try:
        p = pars['erode']
        print(p)
    except KeyError:
        pass
    else:
        # FIXME: this may or may not be the desired primary output
        disk_erosion = disk(p['disk'])
        mask_ero = np.zeros_like(mask, dtype='bool')
        for i, slc in enumerate(mask):  # FIXME: assuming zyx here
            mask_ero[i, :, :] = binary_erosion(slc, disk_erosion)
        if 'postfix' in p.keys(): write(mask_ero, image_in, p['postfix'], im, 'Mask')

    im = write(mask, '{}/'.format(filepath), pars['ods_mask'], im, 'Mask')

    return im


def combine_volumes(filepath, step_key, pars, save_steps=True):
    # TODO: make more general and versatile

    if step_key.endswith('masks'):

        image_in = '{}/{}'.format(filepath, pars['ids_nucl'])
        im = MaskImage(image_in)
        im.load()
        mask_nucl = im.slice_dataset().astype('bool')
        im.close()

        image_in = '{}/{}'.format(filepath, pars['ids_memb'])
        im = MaskImage(image_in)
        im.load()
        mask_memb = im.slice_dataset().astype('bool')
        im.close()

        try:
            p = pars['erode_nucl']
        except KeyError:
            pass
        else:
            disk_erosion = disk(p['disk'])
            for i, slc in enumerate(mask_nucl):  # FIXME: assuming zyx here
                mask_nucl[i, :, :] = binary_erosion(slc, disk_erosion)

        mask = np.logical_and(mask_nucl, ~mask_memb)

        try:
            p = pars['opening_footprint']
        except KeyError:
            pass
        else:
            footprint = create_footprint(p)
            mask = binary_opening(mask, footprint, out=mask)

    im = write(mask, '{}/'.format(filepath), pars['ods_mask'], im, 'Mask')

    return im


def seed_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    image_in = '{}/{}'.format(filepath, pars['ids_mask'])
    im = Image(image_in)
    im.load()
    mask = im.slice_dataset().astype('bool')
    im.close()

    elsize = np.absolute(im.elsize)

    try:
        p = pars['edt']
    except KeyError:
        if 'ids_edt' in pars.keys():
            image_in = '{}/{}'.format(filepath, pars['ids_edt'])
            im = Image(image_in)
            im.load()
            edt = im.slice_dataset()
            im.close()
    else:
        edt = distance_transform_edt(mask, sampling=elsize)
        # mask = im.ds[:].astype('uint32')
        # edt = edt.edt(mask, anisotropy=im.elsize, black_border=True, order='F', parallel=1)
        # TODO?: leverage parallel
        try:
            threshold = p['threshold']
        except KeyError:
            pass
        else:
            edt[edt > threshold] = 0
        if 'postfix' in p.keys(): write(edt, image_in, p['postfix'], im, 'Image')

    try:
        p = pars['modulate']
    except KeyError:
        if 'ids_dog' in pars.keys():
            image_in = '{}/{}'.format(filepath, pars['ids_dog'])
            im = Image(image_in)
            im.load()
            edt = im.slice_dataset()
            im.close()
    else:
        dog = smooth_dog(data, elsize, p['sigma1'], p['sigma2'])
        edt *= normalize_data(dog, a=p['min'], b=p['max'])[0]
        if 'postfix' in p.keys(): write(dog, image_in, p['postfix'], im, 'Image')  # edt and/or dog?

    try:
        p = pars['edt_threshold']
        print(p)
    except KeyError:
        pass
    else:
        mask = edt > pars['edt_threshold']

    try:
        p = pars['peaks']
    except KeyError:
        pass
    else:
        # find peaks in the distance transform
        mask = find_local_maxima(edt, p['window'], p['threshold'])
        try:
            footprint = p['dilate']['footprint']
        except KeyError:
            pass
        else:
            footprint = create_footprint(footprint)
            mask_dil = binary_dilation(mask, selem=footprint)
            if 'postfix' in p['dilate'].keys(): write(mask_dil, image_in, p['dilate']['postfix'], im, 'Mask')

    try:
        p = pars['label']
    except KeyError:
        pass
    else:
        seeds = ndi.label(mask)[0]
        write(seeds, '{}/'.format(filepath), p['ods_peaks'], im, 'Label')

    try:
        p = pars['seeds']
    except KeyError:
        pass
    else:
        if 'threshold' in p.keys():
            thr = p['threshold']
        elif 'threshold' in pars['seeds'].keys():
            thr = pars['seeds']['threshold']
        else:
            thr = 0
        seeds = watershed(-edt, seeds, mask=edt > thr)

    im = write(seeds, '{}/'.format(filepath), pars['ods_labels'], im, 'Label')

    return im


def seed_volume_data(pars, data, mask, elsize):
    # TODO: unduplicate

    try:
        p = pars['edt']
    except KeyError:
        pass
    else:
        edt = distance_transform_edt(mask, sampling=elsize)
        try:
            threshold = p['threshold']
        except KeyError:
            pass
        else:
            edt[edt > threshold] = 0

    try:
        p = pars['modulate']
    except KeyError:
        pass
    else:
        dog = smooth_dog(data, elsize, p['sigma1'], p['sigma2'])
        edt *= normalize_data(dog, a=p['min'], b=p['max'])[0]

    try:
        p = pars['edt_threshold']
    except KeyError:
        pass
    else:
        mask = edt > pars['edt_threshold']

    try:
        p = pars['peaks']
    except KeyError:
        pass
    else:
        mask = find_local_maxima(edt, p['window'], p['threshold'])
        try:
            footprint = p['dilate']['footprint']
        except KeyError:
            pass
        else:
            footprint = create_footprint(footprint)
            mask_dil = binary_dilation(mask, selem=footprint)

    try:
        p = pars['label']
    except KeyError:
        pass
    else:
        seeds = ndi.label(mask)[0]

    try:
        p = pars['seeds']
    except KeyError:
        pass
    else:
        if 'threshold' in p.keys():
            thr = p['threshold']
        elif 'threshold' in pars['seeds'].keys():
            thr = pars['seeds']['threshold']
        else:
            thr = 0
        seeds = watershed(-edt, seeds, mask=edt > thr)

    return seeds


def segment_volume(filepath, step_key, pars, save_steps=True):

    if 'ids_mask' in pars.keys():
        image_in = '{}/{}'.format(filepath, pars['ids_mask'])
        im = Image(image_in)
        im.load()
        mask = im.slice_dataset()
        im.close()
    else:
        mask = None

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    image_in = '{}/{}'.format(filepath, pars['ids_labels'])
    im = Image(image_in)
    im.load()
    seeds = im.slice_dataset()
    im.close()

    if 'ids_neg_seed' in pars.keys():
        image_in = '{}/{}'.format(filepath, pars['ids_neg_seed'])
        im = Image(image_in)
        im.load()
        nseed = im.slice_dataset()
        im.close()
        seeds[~nseed] = max(np.unique(seeds)) + 1

    elsize = np.absolute(im.elsize)

    try:
        p = pars['watershed']
    except KeyError:
        pass
    else:

        ws = segment_volume_data(pars, data, seeds, mask, elsize)

        if 'ids_mask_post' in p.keys():
            image_in = '{}/{}'.format(filepath, p['ids_mask_post'])
            im = Image(image_in)
            im.load()
            mask = im.slice_dataset().astype('bool')
            im.close()
            ws[~mask] = 0

        if 'postfix' in p.keys(): write(ws, image_in, p['postfix'], im, 'Mask')

    im = write(ws, '{}/'.format(filepath), pars['ods_labels'], im, 'Label')

    return im


def segment_volume_data(pars, data, seeds, mask, elsize):

    try:
        p = pars['watershed']
    except KeyError:
        pass
    else:

        if 'invert_data' in p.keys():
            data = -data

        if 'voxel_spacing' in p.keys():
            spacing = p['voxel_spacing']
        else:
            spacing = elsize

        if 'compactness' in p.keys():
            compactness = p['compactness']
        else:
            compactness = 0.0

        try:
            ws = watershed(data, seeds, mask=mask, compactness=compactness, spacing=spacing)
        except TypeError:
            print('WARNING: possibly not using correct spacing for compact watershed')
            ws = watershed(data, seeds, mask=mask, compactness=compactness)

        return ws


def filter_segments(filepath, step_key, pars, save_steps=True):

    # TODO: option to pass image?
    image_in = '{}/{}'.format(filepath, pars['ids_labels'])
    im = Image(image_in)
    im.load()
    ws = im.slice_dataset()
    im.close()

    labelset = set([])

    if 'ids_mask' in pars.keys():
        image_in = '{}/{}'.format(filepath, pars['ids_mask'])
        im = MaskImage(image_in)
        im.load()
        mask = im.slice_dataset().astype('bool')
        im.close()
        labelset |= set(np.unique(ws[~mask]))

    if 'max_size' in pars.keys():
        nvoxels = int(pars['max_size'] / np.prod(im.elsize))
        bc = np.bincount(np.ravel(ws))
        labelset |= set(np.where(bc > nvoxels)[0])

    # TODO: dump deleted labelset?
    maxlabel = max(np.unique(ws))
    if maxlabel:
        fwmap = [True if l in labelset else False for l in range(0, maxlabel + 1)]
        ws[np.array(fwmap)[ws]] = 0

    im = write(ws, '{}/'.format(filepath), pars['ods_labels'], im, 'Label')

    # generate_report('{}/mean'.format(filepath, 'mean'))

    return im


def resegment_largelabels(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_labels'])
    im = Image(image_in)
    im.load()
    labels = im.slice_dataset()
    im.close()

    try:
        p = pars['thresholds']
    except KeyError:
        pass
    else:

        thresholds = p
        elsize = np.absolute(im.elsize)

        if 'ero_distances_1' in pars.keys():
            ero_distances = pars['ero_distances_1']
            dil_distances = [ero_dist - elsize[im.axlab.index('z')] for ero_dist in ero_distances]
            labels = iterative_label_splitter(labels, elsize, thresholds, ero_distances, dil_distances)

        if 'ids_memb' in pars.keys():
            ids_memb = pars['ids_memb']
            ero_thresholds = pars['ero_thresholds']
            dil_distances = pars['dil_distances']
            labels = iterative_label_splitter(labels, elsize, thresholds, ero_thresholds, dil_distances, filepath, ids_memb)

        if 'ero_distances_2' in pars.keys():
            ero_distances = pars['ero_distances_2']
            dil_distances = [ero_dist - elsize[im.axlab.index('z')] for ero_dist in ero_distances]
            labels = iterative_label_splitter(labels, elsize, thresholds, ero_distances, dil_distances)

        # if 'postfix' in p.keys(): write(ws, image_in, p['postfix'], im, 'Mask')

    im = write(labels, '{}/'.format(filepath), pars['ods_labels'], im, 'Label')

    return im


def iterative_label_splitter(labels, elsize, thresholds, ero_distances, dil_distances, filepath='', ids=''):

    for ero_dist, dil_dist in zip(ero_distances, dil_distances):

        labels, labels_large = split_labels(labels, thresholds[1], thresholds[2])

        mask = reduced_mask(labels_large, elsize, ero_dist, filepath, ids)
        relabeled = ndi.label(mask)[0]
        relabeled = remove_small_objects(relabeled, thresholds[0])
        relabeled = expand_labels(relabeled, elsize, dil_dist)

        mask = relabeled.astype('bool')
        relabeled[mask] += np.amax(labels)
        labels[mask] = relabeled[mask]

    return labels


def split_labels(labels, threshold_small, threshold_large):
    """Split labels in normal-sized and large-sized; discard small."""

    labels = remove_small_objects(labels, threshold_small)
    labels_large = remove_small_objects(labels, threshold_large)
    labels[labels_large.astype('bool')] = 0
    return labels, labels_large


def reduced_mask(labels, elsize, ero_val, filepath='', ids=''):
    if filepath:  # ero_val is membrane threshold
        mask_memb = read_vol(filepath, ids) > ero_val
        mask = np.logical_and(labels.astype('bool'), ~mask_memb)
    else:  # ero_val is distance
        fp_zyx = dist_to_extent(ero_val, elsize)
        print('fp_extent', fp_zyx)
        selem = create_footprint(fp_zyx)
        mask = binary_erosion(labels.astype('bool'), selem=selem)
    return mask


def dist_to_extent(dist, elsize):
    ero_vox = [np.ceil(dist / es) * 2 for es in elsize]
    ero_vox_odd = [int(np.floor(ev / 2) * 2 + 1) for ev in ero_vox]
    return ero_vox_odd


# def get_distances(fp_size_z, elsize):
#     anis_fac = np.floor(elsize['z'] / elsize['x'])
#     print('anis_fac', anis_fac)
#     extent = [int(fp_size_z), int(fp_size_z * anis_fac), int(fp_size_z * anis_fac)]
#     distance = ( (extent[2] - anis_fac * 2) / 2) * elsize['x']
#     # FIXME: may not be general: choose distance a bit smaller than erosion footprint
#     return extent, distance


# def seg_large(mask, elsize, dil_dist, thresholds=[50, 3000, 50000]):
#     seeds = ndi.label(mask)[0]  # relabel
#     seeds = remove_small_objects(seeds, thresholds[0])  # remove specks
#     seeds = expand_labels(seeds, elsize, dil_dist)  # refill mask
#     seeds, labels_large = split_labels(seeds, thresholds[1], thresholds[2])  # split normal / large
#     return seeds#, labels_large


# def add_labels(labels, labels_new):
#     """Add new labels to a label_image."""
#     mask = labels_new.astype('bool')
#     labels_new[mask] += np.amax(labels)
#     labels[mask] = labels_new[mask]
#     return labels


def read_vol(filepath, ids):
    im = Image('{}/{}'.format(filepath, ids), permission='r')
    im.load()
    vol = im.slice_dataset()
    im.close()
    return vol


# def write(vol, filepath, ids, props):
#     im = Image('{}/{}'.format(filepath, ids), permission='r+', **props)
#     im.create()
#     im.write(vol)
#     im.close()


def expand_labels(label_image, sampling, distance=1):
    # adapted for anisotropy from scikit-image expand_labels
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, sampling=sampling, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


def normalize_data(data, a=1.00, b=1.01):
    """Normalize data."""

    data = data.astype('float64')
    datamin = np.amin(data)
    datamax = np.amax(data)
    data -= datamin
    data *= (b-a)/(datamax-datamin)
    data += a

    return data, [datamin, datamax]


def gen_outpath(im, pf):
    """Fix us a postfixed output path."""

    comps = im.split_path()
    if im.format == '.nii':
        outpath = "{}{}{}".format(comps['base'], pf, comps['ext'])
    elif im.format == '.h5':
        outpath = "{}{}{}".format(comps['file'], comps['int'], pf)

    return outpath


def write(out, outstem, postfix, ref_im, imtype='Image'):
    """Write an image to disk."""

    outstem = outstem or gen_outpath(ref_im, '')
    outpath = '{}{}'.format(outstem, postfix)

    props = ref_im.get_props()
    props['dtype'] = out.dtype

    if imtype == 'Label':
        mo = LabelImage(outpath, **props)
    elif imtype == 'Mask':
        mo = MaskImage(outpath, **props)
    else:
        mo = Image(outpath, **props)

    mo.create()
    mo.write(out)

    if imtype == 'Label':
        mo.set_maxlabel()
        mo.ds.attrs.create('maxlabel', mo.maxlabel, dtype='uint32')

    return mo


def find_local_maxima(data, size=[13, 13, 3], threshold=0.05, noise_sd=0.0):
    """Find peaks in image."""

    if threshold == -float('Inf'):
        threshold = img.min()

    def add_noise(data, sd=0.001):

        mask = data == 0
        data += np.random.normal(0, sd, (data.shape))
        data[mask] = 0

        return data

    """
    NOTE: handle double identified peaks within the footprint region
    by adding a bit of noise
    (with same height within maximum_filtered image)
    this happens a lot for edt, because of discrete distances
    This has now been handled through modulation of distance transform.
    """
    if noise_sd:
        data = add_noise(data, noise_sd)

    footprint = create_footprint(size)
    image_max = ndi.maximum_filter(data, footprint=footprint, mode='constant')

    mask = data == image_max
    mask &= data > threshold

    coordinates = np.column_stack(np.nonzero(mask))[::-1]

    peaks = np.zeros_like(data, dtype=np.bool)
    peaks[tuple(coordinates.T)] = True

    return peaks


def create_footprint(size=[5, 21, 21]):
    """Create a 3D ellipsoid-like structure element for anisotropic data.

    FIXME: why don't I just get an isotropic ball and delete some slices?
    r = int(footprint[2] / 2)
    b = ball(r, dtype='bool')
    slc = [0, ..., r, ..., -1]
    e = b[slc, :, :]
    #e = np.delete(b, , axis=0)  # TODO flexible z
    """

    footprint = np.zeros(size)
    c_idxs = [int(size[0] / 2), int(size[1] / 2), int(size[2] / 2)]
    disk_ctr = disk(c_idxs[1])
    footprint[int(size[0]/2), :, :] = disk_ctr
    for i in range(c_idxs[0]):
        j = 2 + i + 1
        r = int(size[1] / j)
        d = disk(r)
        slc = slice(c_idxs[1]-r, c_idxs[1]+r+1, 1)
        footprint[c_idxs[0]-i-1, slc, slc] = d
        footprint[c_idxs[0]+i+1, slc, slc] = d

    return footprint


def shift_channel(data, n_planes=0, zdim_idx=0):

    if n_planes:
        if zdim_idx == 0:
            data[n_planes:, :, :] = data[:-n_planes, :, :]
            data[:n_planes, :, :] = 0
        elif zdim_idx == 2:
            data[:, :, n_planes:] = data[:, :, :-n_planes]
            data[:, :, :n_planes] = 0

    return data


def smooth_channel(data, sigma=3, filter='median'):

    if filter == 'median':
        k = ball(sigma)  # TODO footprint
        data_smooth = median(data, k)
    elif filter == 'gaussian':
        data_smooth = gaussian(data, sigma=sigma, preserve_range=True)

    return data_smooth


def smooth_channel_inplane(data, sigma=3, filter='median'):

    k = disk(sigma)
    data_smooth = np.zeros_like(data)
    for i, slc in enumerate(data):
        if filter == 'median':
            data_smooth[i, :, :] = median(slc, k)
        elif filter == 'gaussian':
            data_smooth[i, :, :] = gaussian(slc, sigma=sigma, preserve_range=True)

    return data_smooth


def smooth_dog(data, elsize, sigma1, sigma2):

    elsize = np.absolute(elsize)
    s1 = [sigma1 / es for es in elsize]
    s2 = [sigma2 / es for es in elsize]
    dog = gaussian(data, s1) - gaussian(data, s2)

    return dog


# UNUSED?
def upsample_to_block(mask_ds, block_us, dsfacs=[1, 16, 16, 1], order=0):
    """Upsample a low-res mask to a full-res block."""

    comps = block_us.split_path()
    from stapl3d import split_filename
    block_info = split_filename(comps['file'])[0]
    slices_us = [slice(block_info['z'], block_info['Z'], None),
                 slice(block_info['y'], block_info['Y'], None),
                 slice(block_info['x'], block_info['X'], None)]

    mask_ds.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                      for slc, ds in zip(slices_us, dsfacs)]
    mask_block = mask_ds.slice_dataset()  #.astype('float32')
    mask_us = resize(mask_block, block_us.dims, preserve_range=True, order=order)

    return mask_us.astype(mask_ds.dtype)


def delete_labels_in_mask(labels, mask, maxlabel=0):
    """Delete the labels found within mask."""

    labels_del = np.copy(labels)

    # TODO: save deleted labelset
    maxlabel = maxlabel or np.amax(labels[:])
    if maxlabel:
        labelset = set(np.unique(labels[mask]))
        fwmap = [True if l in labelset else False for l in range(0, maxlabel + 1)]
        labels_del[np.array(fwmap)[labels]] = 0

    return labels_del


# UNUSED?
def find_border_segments(im):

    segments = set([])
    for idx_z in [0, -1]:
        for idx_y in [0, -1]:
            for idx_x in [0, -1]:
                segments = segments | set(im.ds[idx_z, idx_y, idx_x].ravel())

    return segments


def memb_mask(labels, footprint=None):
    """Create membrane mask from label boundaries."""

    mask = find_boundaries(labels)

    if footprint is None:
        from skimage.morphology import ball
        footprint = ball(1, dtype='bool')
        footprint[0, :, :] = False
        footprint[-1, :, :] = False
    else:
        footprint = create_footprint(footprint)
    mask = binary_dilation(mask, selem=footprint)

    # constrain within cells
    mask[~labels.astype('bool')] = False

    return mask


def nucl_mask(labels):
    """Create nuclear mask."""
    # TODO
    return labels.astype('bool')


def csol_mask(labels, mask_nucl, mask_memb):
    """Create cytoplasm mask from labels and nuclear/membrane masks."""

    mask_csol = labels.astype('bool')
    mask_csol &= ~mask_nucl
    mask_csol &= ~mask_memb

    return mask_csol


class Subsegment3r(Block3r):
    """Subdivide cells in compartments."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'subsegmentation'

        super(Subsegment3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'estimate': self.estimate,
            })

        self._parallelization.update({
            'estimate': ['blocks'],
            })

        self._parameter_sets.update({
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        # TODO
        self._parameter_table.update({
            })

        default_attr = {
            'ids_labels': 'segm/labels_zip',
            'ids_nucl_mask': 'nucl/mask',
            'ids_memb_mask': '',
            'ids_csol_mask': '',
            'ods_full': '',
            'ods_memb': '',
            'ods_nucl': '',
            'ods_csol': '',
            'footprint_memb': None,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_subsegmenter()

        self._init_log()

        self._prep_blocks()

        self._images = ['nucl/prep']
        self._labels = ['segm/labels_full', 'segm/labels_nucl', 'segm/labels_memb', 'segm/labels_csol']

    def _init_paths_subsegmenter(self):

        prev_path = {
            'moduledir': 'zipping', 'module_id': 'zipping',
            'step_id': 'zipping', 'step': 'estimate',
            'ioitem': 'outputs', 'output': 'blockfiles',
            }
        bpat = self._get_inpath(prev_path)
        if bpat == 'default':
            os.makedirs('blocks', exist_ok=True)
            bpat = self._build_path(moduledir='blocks', prefixes=[self.prefix, 'blocks'], suffixes=[{'b': 'p'}])

        self._paths.update({
            'estimate': {
                'inputs': {
                    'blockfiles': f'{bpat}',
                    },
                'outputs': {
                    'blockfiles': f'{bpat}',
                    'report': f'{bpat}_subsegment'.replace('.h5', '.pdf'),
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def estimate(self, **kwargs):
        """Subdivide cells in compartments."""

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, block):
        """Subdivide cells in compartments."""

        block = self._blocks[block]
        inputs = self._prep_paths(self.inputs, reps={'b': block.idx})
        outputs = self._prep_paths(self.outputs, reps={'b': block.idx})

        inputfile = inputs['blockfiles'].format(b=block)

        label_im = LabelImage('{}/{}'.format(inputfile, self.ids_labels))
        label_im.load(load_data=False)
        try:
            del label_im.file[self.ods_full]
        except KeyError:
            pass
        label_im.file[self.ods_full] = label_im.file[self.ids_labels]
        labels = label_im.slice_dataset()
        label_im.close()

        def read_mask(inputfile, ids):
            im = MaskImage('{}/{}'.format(inputfile, ids), permission='r')
            im.load()
            mask = im.slice_dataset().astype('bool')
            im.close()
            return mask

        def write_masked(label_im, inputfile, ods, labels, mask):
            outstem = '{}/{}'.format(inputfile, ods)
            labs = np.copy(labels)
            labs[~mask.astype('bool')] = 0
            write(labs, outstem, '', label_im, imtype='Label')

        if self.ods_memb:
            if self.ids_memb_mask:
                mask_memb = read_mask(inputfile, self.ids_memb_mask)
            else:
                mask_memb = memb_mask(labels, self.footprint_memb)
            write_masked(label_im, inputfile, self.ods_memb, labels, mask_memb)

        if self.ods_nucl:
            if self.ids_nucl_mask:
                mask_nucl = read_mask(inputfile, self.ids_nucl_mask)
            else:
                mask_nucl = nucl_mask(labels)
            write_masked(label_im, inputfile, self.ods_nucl, labels, mask_nucl)

        if self.ods_csol:
            if self.ids_csol_mask:
                mask_csol = read_mask(inputfile, self.ids_csol_mask)
            else:
                mask_csol = csol_mask(labels, mask_nucl, mask_memb)
            write_masked(label_im, inputfile, self.ods_csol, labels, mask_csol)

        # self.dump_parameters(self.step, outputs['parameters'])
        self.report(outputs['report'], inputs=inputs, outputs=outputs)

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        filepath = kwargs['outputs']['blockfiles']
        # kwargs['props'] = get_imageprops(filepath)
        kwargs['paths'] = get_paths(f'{filepath}/mean')
        kwargs['centreslices'] = get_centreslices(kwargs)

        return kwargs

    def _gen_subgrid(self, f, gs, channel=None):
        """Generate the axes for printing the report."""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(4, 2)

        t = [
            'nucl', 'memb',
            'nucl_mask', 'memb_mask',
            'seeds_mask', 'seeds_mask_dil',
            'segm_seeds', 'segm',
            ]
        voldict = {name: {'idx': i, 'title': name} for i, name in enumerate(t)}

        for k, vd in voldict.items():
            titles[k] = (vd['title'], 'lcm', 0)
            axdict[k] = gen_orthoplot(f, gs01[vd['idx']], 5, 1)

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict={}):
        """Plot graph with shading image."""

        def plot_imgs(img, labels, axdict, axkey, axidx, aspect, cmap='gray', alpha=0.3, colors=None):

            if dim == 'x':
                img = img.transpose()

            if labels is not None:
                if dim == 'x':
                    labels = labels.transpose()
                data = label2rgb(labels, image=img, alpha=alpha, bg_label=0, colors=colors)
            else:
                data = img

            ax = axdict[axkey][axidx]
            ax.imshow(data, aspect=aspect, cmap=cmap)
            ax.axis('off')

        def get_data(prefered, fallback, dimfac):
            try:
                slc = centreslices[prefered][dim] * dimfac
                return slc
            except (TypeError, KeyError):
                print('{} not found: falling back to {}'.format(prefered, fallback))
                try:
                    slc = centreslices[fallback][dim] * dimfac
                    return slc
                except (TypeError, KeyError):
                    print('{} not found: falling back to empty'.format(fallback))
                    # TODO: empty image of correct dim
                    return None

        centreslices = info_dict['centreslices']

        # aspects = ['equal', 'equal', 'equal']
        aspects = ['equal', 'auto', 'auto']
        for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

            data_nucl = get_data('nucl/prep', 'nucl/mean', dimfac=3)
            data_memb = get_data('memb/prep', 'memb/mean', dimfac=5)
            if data_memb is None:
                data_memb = np.zeros(data_nucl.shape)

            plot_imgs(data_nucl, None, axdict, 'nucl', i, aspect, 'gray')
            plot_imgs(data_memb, None, axdict, 'memb', i, aspect, 'gray')

            try:

                data_edt = centreslices['segm/seeds_mask_edt'][dim]

                labels = centreslices['mask'][dim].astype('uint8')
                labels[centreslices['nucl/mask'][dim]] = 2
                plot_imgs(data_nucl, labels, axdict, 'nucl_mask', i, aspect, None, 0.5, colors=[[1, 0, 0], [0, 1, 0]])

                labels = centreslices['mask'][dim].astype('uint8')
                labels[centreslices['memb/mask'][dim]] = 2
                plot_imgs(data_memb, labels, axdict, 'memb_mask', i, aspect, None, 0.5, colors=[[1, 0, 0], [0, 1, 0]])

                labels = centreslices['segm/seeds_mask'][dim].astype('uint8')
                labels[centreslices['segm/seeds_mask_dil'][dim]] = 2
                plot_imgs(data_nucl, labels, axdict, 'seeds_mask', i, aspect, None, 0.5, colors=[[1, 0, 0], [0, 1, 0]])

                labels = centreslices['segm/seeds_mask_dil'][dim]
                plot_imgs(data_edt, labels, axdict, 'seeds_mask_dil', i, aspect, None, 0.5, colors=None)

                labels = centreslices['segm/labels_edt'][dim]
                plot_imgs(data_nucl, labels, axdict, 'segm_seeds', i, aspect, None, 0.7, colors=None)

                labels = centreslices['segm/labels'][dim]
                plot_imgs(data_memb, labels, axdict, 'segm', i, aspect, None, 0.3, colors=None)

            except (TypeError, KeyError):
                print('not all steps were found: generating simplified report')

                labels = centreslices['segm/labels'][dim]
                plot_imgs(data_memb, labels, axdict, 'segm', i, aspect, None, 0.3, colors=None)

    def view(self, input=[], images=[], labels=[], settings={}):

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, (int, float)):
            input = self._blocks[input].path.replace('/{ods}', '')
        else:
            input = input or [0, 1]

        super().view(input, images, labels, settings)



if __name__ == "__main__":
    main(sys.argv[1:])
