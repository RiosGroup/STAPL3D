#!/usr/bin/env python

"""Segment cells from membrane and nuclear channels.

"""

import os
import sys
import logging
import pickle
import shutil
import multiprocessing

from inspect import signature, stack
from importlib import import_module

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
from skimage.transform import resize, rescale
from skimage.feature import blob_log
from skimage.segmentation import (
    find_boundaries,
    watershed,
    slic,
    join_segmentations,
)
from skimage.filters import (
    gaussian,
    median,
    threshold_sauvola,
    threshold_otsu,
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

from stapl3d import parse_args, Image, MaskImage, LabelImage
from stapl3d import get_paths  # TODO: into Image/Stapl3r
from stapl3d.blocks import Block, Block3r
# from stapl3d.segmentation.segment import Zipset
from stapl3d.reporting import (
    gen_orthoplot,
    get_centreslices,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Segment cells from membrane and nuclear channels."""

    steps = ['estimate', 'postprocess']  # , 'subsegment'
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
            'postprocess': self.postprocess,
            })

        self._parallelization.update({
            'estimate': ['blocks'],
            'postprocess': [],
            })

        self._parameter_sets.update({
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        # TODO
        self._parameter_table.update({

            })

        # TODO: presets / derive from yaml
        default_attr = {
            '_plotlayout': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_segmenter()

        self._init_log()

        self._set_plotlayout()

        self._prep_blocks()

        self._images = ['nucl/prep']
        self._labels = ['segm/labels_full', 'segm/labels_nucl',
                        'segm/labels_memb', 'segm/labels_csol']

    def _init_paths_segmenter(self):

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        stem = self._build_path(moduledir=self._logdir)
        bpat = self.outputpaths['blockinfo']['blockfiles']  #self._l2p(['blocks', '{f}.h5'])
        bmat = self._pat2mat(bpat)  # <>_blocks_B{b:05d}.h5  => <>_blocks_B*.h5
        # TODO: for filepaths, blocks, ...

        self._paths.update({
            'estimate': {
                'inputs': {
                    'blockfiles': f'{bpat}',
                    },
                'outputs': {
                    'blockfiles': f'{bpat}',
                    'report': f'{bpat}'.replace('.h5', '.pdf'),
#                    'parameters': f'{bpat}'.replace('.h5', '.yml'),
                    }
                },
            'postprocess': {
                'inputs': {
                    'report': f'{bmat}'.replace('.h5', '.pdf'),
                    },
                'outputs': {
                    'report': f'{stem}.pdf',
                    },
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

    def _estimate_block(self, block_idx):
        """Segment cells from membrane and nuclear channels.
        
        prep(ids_image) -> ods_image
        mask(ids_image) -> ods_mask
        combine_images(ids_image1, ids_image2) -> ods_image
        combine_masks(ids_nucl, ids_memb) -> ods_mask
        combine_labels(ids_label1, ids_label2) -> ods_label
        seed(ids_mask) -> ids_label
        segment()
        filter()
        """

        block = self._blocks[block_idx]

        inputs = self._prep_paths_blockfiles(self.inputs, block)
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        params = self._cfg[self.step_id]['estimate']

        fundict = {
            'generic': generic_test,
            'prep': prep_volume,
            'mask': mask_volume,
            'combine_masks': combine_masks,
            'combine_labels': combine_labels,
            # 'combine': combine_volumes,
            'seed': seed_volume,
            'segment': segment_volume,
            'filter': filter_segments,
            'iterative_split': resegment_largelabels,
            '_plotlayout': None,
        }

        for step_key, pars in params.items():

            t = time.time()

            fun_key = fun_for_step(fundict, step_key)

            if fun_key is None:
                pass
            elif fun_key == '_plotlayout':
                name = os.path.basename(block.path.split('.h5')[0])
                #self.dump_parameters(self.step, outputs['parameters'])
                self.report(outputs['report'], name, inputs=inputs, outputs=outputs)
            else:
                #fundict[fun_key](inputs['blockfiles'], step_key, pars)
                fundict[fun_key](block, step_key, pars)

            elapsed = time.time() - t
            print('{} took {:1f} s for block {}'.format(step_key, elapsed, block.id))

    def postprocess(self, **kwargs):
        """Merge block reports."""

        self._prep_step('postprocess', kwargs)

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        pdfs = glob(inputs['report'])
        pdfs.sort()
        self._merge(pdfs, outputs['report'], self._merge_reports)

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        filepath = kwargs['outputs']['blockfiles']
        # kwargs['props'] = get_imageprops(filepath)
        kwargs['paths'] = get_paths(f'{filepath}/mean')  # FIXME: assumes dataset is generated under that name

        idss = []
        for r in self._plotlayout:
            for key in ['data', 'labelkey']:
                for v in r:
                    if isinstance(v[key], str):
                        idss += [v[key]]
                    elif isinstance(v[key], list):
                        idss += v[key]

        kwargs['centreslices'] = get_centreslices(kwargs, idss=idss)

        return kwargs

    def _gen_subgrid(self, f, gs, channel=None):
        """Generate the axes for printing the report."""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(len(self._plotlayout), len(self._plotlayout[0]))



        t = [d for pl in self._plotlayout for d in pl]
        for i, vd in enumerate(t):
            k = vd['name']

            vd = {**{'plottype': 'ortho'}, **vd}

            if vd['plottype'] == 'hist':
                titles[k] = (k, 'rc', 0)
                axdict[k] = f.add_subplot(gs01[i])
                axdict[k].tick_params(axis='both', direction='in')
                for l in ['top', 'right']:
                    axdict[k].spines[l].set_visible(False)
#            elif vd['plottype'] == 'ortho':
            else:
                titles[k] = (k, 'lcm', 0)
                axdict[k] = gen_orthoplot(f, gs01[i], 5, 1)




        self._add_titles(axdict, titles)

        return axdict

    def _set_plotlayout(self):

        if self._plotlayout:
            dummy = {
                'data': None,
                'labelkey': None,
                'cmap': 'gray',
                'alpha': 0.3,
                'colors': None,
                'vmin': None,
                'vmax': None,
                'labelmode': '',
                }
            self._plotlayout = [[{**dummy, **p} for p in r] for r in self._plotlayout]
            return

        params_defaults = {
            'prep_dset': {'ods_image': 'prep'},
            'mask_dset': {'ods_mask': 'mask'},
            'prep_nucl': {'ods_image': 'nucl/prep'},
            'mask_nucl': {'ods_mask': 'nucl/mask'},
            'prep_memb': {'ods_image': 'memb/prep'},
            'mask_memb': {'ods_mask': 'memb/mask'},
            'combine_masks': {'ods_mask': 'segm/seeds_mask'},
            'seed': {'ods_labels': 'segm/labels_edt'},
            'segment': {'ods_labels': 'segm/labels_raw'},
            'filter': {'ods_labels': 'segm/labels'},
        }

        params = self._cfg[self.step_id]['estimate'] if self._cfg else {}
        params = {**params_defaults, **params}

        try:
            dilkey = params['combine_masks']['ods_mask']
            if 'postfix' in params['combine_masks']['dilate'].keys():
                dilkey += params['combine_masks']['dilate']['postfix']
        except:
            dilkey = 'segm/seeds_mask_dil'

        try:
            edtkey =  params['seed']['ids_mask']
            if 'postfix' in params['seed']['edt'].keys():
                edtkey += params['seed']['edt']['postfix']
        except:
            edtkey = 'segm/seeds_mask_edt'

        self._plotlayout = [
            [
                {
                    'name': 'nucl',
                    'data': params['prep_nucl']['ods_image'],
                },
                {
                    'name': 'memb',
                    'data': params['prep_memb']['ods_image'],
                },
            ],
            [
                {
                    'name': 'nucl_mask',
                    'data': params['prep_nucl']['ods_image'],
                    'labelkey': [params['mask_dset']['ods_mask'], params['mask_nucl']['ods_mask']],
                    'cmap': None,
                    'alpha': 0.5,
                    'colors': [[1, 0, 0], [0, 1, 0]],
                },
                {
                    'name': 'memb_mask',
                    'data': params['prep_memb']['ods_image'],
                    'labelkey': [params['mask_dset']['ods_mask'], params['mask_memb']['ods_mask']],
                    'cmap': None,
                    'alpha': 0.5,
                    'colors': [[1, 0, 0], [0, 1, 0]],
                },
            ],
            [
                {
                    'name': 'seeds_mask',
                    'data': params['prep_nucl']['ods_image'],
                    'labelkey': [params['combine_masks']['ods_mask'], dilkey],
                    'cmap': None,
                    'alpha': 0.5,
                    'colors': [[1, 0, 0], [0, 1, 0]],
                },
                {
                    'name': 'seeds_mask_dil',
                    'data': edtkey,
                    'labelkey': dilkey,
                    'cmap': None,
                    'alpha': 0.5,
                },
            ],
            [
                {
                    'name': 'segm_seeds',
                    'data': params['prep_nucl']['ods_image'],
                    'labelkey': params['seed']['ods_labels'],
                    'cmap': None,
                    'alpha': 0.7,
                },
                {
                    'name': 'segm',
                    'data': params['prep_nucl']['ods_image'],
                    'labelkey': params['filter']['ods_labels'],
                    'cmap': None,
                    'alpha': 0.3,
                },
            ],
        ]

        dummy = {
            'data': None,
            'labelkey': None,
            'cmap': 'gray',
            'alpha': 0.3,
            'colors': None,
            'vmin': None,
            'vmax': None,
            'labelmode': '',
            }
        self._plotlayout = [[{**dummy, **p} for p in r] for r in self._plotlayout]

    def _plot_images(self, f, axdict, info_dict={}):
        """Plot graph with shading image."""

        def foo(plotdict, centreslices, dim, ax, aspect):

            try:

                data = get_data(plotdict['data'], centreslices, dim)
                labels = get_labels(plotdict['labelkey'], centreslices, dim)

                plot_imgs(data, labels, ax, aspect,
                          plotdict['cmap'], plotdict['alpha'], plotdict['colors'],
                          plotdict['vmin'], plotdict['vmax'], plotdict['labelmode'])

            except (TypeError, KeyError):

                ax.axis('off')

        def get_data(key, centreslices, dim, dimfac=1, dtype_max=65535):

            try:
                return centreslices[key][dim]  # * dimfac
                #slc = centreslices[key][dim] / dtype_max
                #return slc * dimfac
            except (TypeError, KeyError):
                # TODO: empty image of correct dim
                return None

        def get_labels(keys, centreslices, dim):

            if keys is None:
                return None
            elif isinstance(keys, list):  # interpret as stacked masks
                labels = np.zeros_like(centreslices[keys[0]][dim], dtype='uint8')
                for i, key in enumerate(keys):
                    mask = centreslices[key][dim].astype('bool')
                    labels[mask] = i + 1
            else:  # single mask or labels
                labels = centreslices[keys][dim]

            return labels.astype('int')

        def plot_imgs(img, labels, ax, aspect,
                      cmap='gray', alpha=0.3, colors=None,
                      vmin=None, vmax=None, labelmode=''):

            ax.axis('off')

            if img is None and labels is None:
                return
            elif img is None:
                img = np.zeros(labels.shape)

            if dim == 'x':
                img = img.transpose()
                if labels is not None:
                    labels = labels.transpose()

            if labels is not None:

                if vmax is not None:
                    scalefac = 65535 / vmax
                    img *= scalefac
                img = img / 256  # uint16 to uint8
                img = img.astype('uint8')

                if labelmode == 'outline':
                    bound = find_boundaries(labels)
                    labels[~bound] = 0

                img = label2rgb(labels, image=img, bg_label=0, alpha=alpha, colors=colors)
                ax.imshow(img, aspect=aspect, cmap=cmap)

            else:

                ax.imshow(img, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)

        for d in [p for pl in self._plotlayout for p in pl]:

            d = {**{'plottype': 'ortho'}, **d}
            if 'hist' in d['plottype']:

                ax = axdict[d['name']]
                logscale = True

                h5_path = info_dict['paths']['file']
                ids = d['data']
                im = Image(f'{h5_path}/{ids}', permission='r')
                im.load(load_data=False)
                data = im.slice_dataset()
                im.close()

                if 'mask' in d.keys():
                    h5_path = info_dict['paths']['file']
                    ids = d['mask']
                    im = Image(f'{h5_path}/{ids}', permission='r')
                    im.load(load_data=False)
                    mask = im.slice_dataset()
                    im.close()

                    data = [np.ravel(data[~mask]), np.ravel(data[mask])]
                    segcolors = [[0, 0, 0], [0, 0, 0], [1, 0, 0]]
                    ax.hist(data, bins=256, log=logscale, histtype='bar', stacked=True, color=segcolors[1:])
                else:
                    ax.hist(np.ravel(data), bins=256, log=logscale, color=[0, 0, 0])

#            if d['plottype'] == 'ortho':
            else:
                dims = {0: ('z', 'equal'), 1: ('y', 'auto'), 2: ('x', 'auto')}
                for i, (dim, aspect) in dims.items():
                    foo(d, info_dict['centreslices'], dim, axdict[d['name']][i], aspect)


#                infun = np.iinfo if data.dtype.kind in 'ui' else np.finfo
#                dmax = infun(data.dtype).max
#                ax.set_xlim([0, dmax])


    def view(self, input=[], images=[], labels=[], settings={}):

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, (int, float)):
            input = self._blocks[input].path
        else:
            input = input or [0, 1]

        super().view(input, images, labels, settings)


def fun_for_step(fundict, step_key):
    for fun_key in fundict.keys():
        if step_key.startswith(fun_key):
            return fun_key


def write_output(out, outpath, im, imtype='Image'):

    props = im.get_props()

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


def write_image(data, p, im):

    outpath = p['basepath'] + p['suffix'] if 'suffix' in p.keys() else ''
    # outpath = gen_outpath(im, p['suffix']) if 'suffix' in p.keys() else ''

    props = im.get_props()
    props['dtype'] = data.dtype

    # FIXME: indexing [1] may not work for all calls?
    daddy = stack()[1][3]  # caller function name
    sig = signature(eval(daddy))
    imclass = sig.return_annotation

    mo = imclass(outpath, **props)
    mo.create()
    mo.write(data)

    if isinstance(imclass, LabelImage):
        mo.set_maxlabel()
        mo.ds.attrs.create('maxlabel', mo.maxlabel, dtype='uint32')

    return mo


def read_block(block, ids):

    if isinstance(block, Block):
        # block.create_dataset(ids)
        block.create_dataset(ids, axlab='zyx')  # TMP
        block_ds_in = block.datasets[ids]
        block_ds_in.read(from_block=True)
        im = block_ds_in.image
    else:
        block.read_margins(ids)
        block.assemble_dataset(ids)
        im = block.assembled.datasets[ids].image

    return im


def write_block(block, out, ods, axlab, imtype=''):

    if isinstance(block, Block):
        block.create_dataset(
            ods,
            axlab=axlab,
            dtype=out.dtype,
            imtype=imtype,
            create_image=True,
            )
        block.datasets[ods].write(out)
    else:
        block.assembled.create_dataset(ods, axlab=axlab, imtype=imtype)
        block.assembled.datasets[ods]
        block.assembled.datasets[ods].dtype = out.dtype
        block.assembled.datasets[ods].create_image(data=out)
        block.assembled.datasets[ods].image.ds[:] = out
        # block.assembled.datasets[ods].image.close()




# SIGNATURES
def prep_foo(im: Image, key: str, p: dict) -> Image:
    """"""
    data = im.ds[:]
    data = gaussian(data)
    im = write_image(data, p, im, 'Image')
    return im
def mask_foo(im: Image, key: str, p: dict) -> MaskImage:
    """"""
    data = im.ds[:]
    data = data > 0
    im = write_image(data, p, im, 'MaskImage')
    return im
def seed_foo(im: MaskImage, key: str, p: dict) -> LabelImage:
    """"""
    data = im.ds[:]
    data = ndi.label(data)[0]
    im = write_image(data, p, im, 'LabelImage')
    return im
def segment_foo(im1: Image, im2: LabelImage, key: str, p: dict) -> LabelImage:
    """"""
    data = im1.ds[:]
    seeds = im2.ds[:]
    data = watershed(data, seeds)
    im = write_image(data, p, im, 'LabelImage')
    return im
def filter_foo(im: LabelImage, key: str, p: dict) -> LabelImage:
    """"""
    data = im.ds[:]
    from skimage.segmentation import clear_border
    data = clear_border(data)
    im = write_image(data, p, im, 'LabelImage')
    return im
def combine_images_foo(im1: Image, im2: Image, key: str, p: dict) -> Image:
    """"""
    data1 = im1.ds[:]
    data2 = im2.ds[:]
    data = np.add(data1, data2)
    im = write_image(data, p, im, 'Image')
    return im
def combine_masks_foo(im1: MaskImage, im2: MaskImage, key: str, p: dict) -> MaskImage:
    """"""
    data1 = im1.ds[:]
    data2 = im2.ds[:]
    data = np.logical_and(data1, data2)
    im = write_image(data, p, im, 'MaskImage')
    return im
def combine_labels_foo(im1: LabelImage, im2: LabelImage, key: str, p: dict) -> LabelImage:
    """"""
    data1 = im1.ds[:]
    data2 = im2.ds[:]
    data = join_segmentations(data1, data2)
    im = write_image(data, p, im, 'LabelImage')
    return im




def generic_test(block: Block, step_key: str, pars: dict) -> None:
    """"""

    for funpath, parameters in pars.items():
        out = external_fun(block, funpath, parameters)


def external_fun(block, funpath, parameters):
    """Apply an external function to the input image."""

    parameters = {**{'args': [], 'kwargs': {}, 'return_map': []}, **parameters}

    # get function
    modpath, funname = funpath.rsplit('.', 1)
    mod = import_module(modpath)
    fun = getattr(mod, funname)

    # replace special string with numpy arrays
    def str2array(block, imstring):
        c = imstring.split(':')
        imclass = c[0]
        ids = c[1]
        attr_name = c[2] if len(c) == 3 else 'ds'
        im = read_block(block, ids)
        im.load()
        data = getattr(im, attr_name)
        im.close()
        return data

    def replace_imstring(block, arg):
        is_imstring = isinstance(arg, str) and ':' in arg
        return str2array(block, arg) if is_imstring else arg

    args = [replace_imstring(block, arg) for arg in parameters['args']]
    kwargs = {k: replace_imstring(block, arg) for k, arg in parameters['kwargs'].items()}
    return_map = parameters['return_map']

    retvals = fun(*args, **kwargs)

    for ret, retval in zip(return_map, retvals):
        if ret is not None:  # TODO: more options
            imclass, ods = ret.split(':')
            write_block(block, retval, ods, block.axlab[:3], imtype=imclass.replace('Image', ''))
              # assuming 3D -> 3D here for axlab



def prep_volume(block: Block, step_key: str, pars: dict) -> None:
    """
    ids_image
    ods_image

    skimage:
    unet:
        comp: memb  # => squeeze (4D czyx to 3D zyx)
        comp: nucl  # => split
    downsample:
    shift_planes:
    opening:
    filter:
    norm:

    """

    fundict = {
        'skimage': prep_external,
        'filter': prep_filter,
        'unet': prep_unet,
        'downsample': prep_downsample,
        'shift_planes': prep_shift_planes,
        'opening': prep_opening,
        'norm': prep_norm,
    }

    im = read_block(block, pars['ids_image'])
    basepath = f'{block.path}/{pars["ids_image"]}'  # TODO: for other steps

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        p['basepath'] = basepath

        im.load()
        im = fundict[fun_key](im, key, p)
        im.close()

    im.load()  # if intermediate is written, the h5 has to be reopened...
    write_block(block, im.ds[:], pars['ods_image'], im.axlab, 'Image')


def prep_external(im: Image, key: str, p: dict) -> Image:
    """Apply a scikit-image function to the input image.

    the function has to take an array as first argument and return an array.
    """

    data = im.ds[:]

    modpath, funname = key.rsplit('.', 1)
    mod = import_module(modpath)
    fun = getattr(mod, funname)

    data = fun(data, **p)

    im = write_image(data, p, im)

    return im


def prep_unet(im: Image, key: str, p: dict) -> Image:
    """Extract 3D volumes from 4D UNET output."""

    ext = '.h5'
    fstem=im.path.split(ext)[0]
    filepath = f'{fstem}{ext}'

    comp = p['comp']
    dir = f'predictions_{comp}_generic'  # TODO: flexible
    p, b = os.path.split(filepath)
    fpath = os.path.join(p, dir, b)
    fpath = fpath.replace('.h5', '_predictions.h5/predictions')
    im2 = Image(fpath)
    im2.load()
    data = im2.slice_dataset()
    im2.close()

    if comp == 'nucl':
        # 2-ch UNET output with centre and surround
        # output 1. centre, 2. surround, 3. mean
        outpath = '{fp}/{ids}'.format(fp=filepath, ids='nucl/unet_cent')
        im = write_output(np.squeeze(data[0, :, :, :]), outpath, im)
        outpath = '{fp}/{ids}'.format(fp=filepath, ids='nucl/unet_peri')
        im = write_output(np.squeeze(data[1, :, :, :]), outpath, im)
        im = write_output(np.mean(data, axis=0), '', im)
    elif comp == 'memb':
        # single-ch UNET output
        # output squeezed (1st axis) to 3D
        im = write_output(np.squeeze(data), '', im)

    return im


def prep_downsample(im: Image, key: str, p: dict) -> Image:
    """Downsample the input image."""

    outpath = gen_outpath(im, p['postfix']) if 'postfix' in p.keys() else ''
    im_ds = im.downsampled(p['factors'], outputpath=outpath)

    return im_ds


def prep_shift_planes(im: Image, key: str, p: dict) -> Image:

    data = im.ds[:]

    n_planes = p['n_planes']
    zdim_idx = 0
    if n_planes:
        if zdim_idx == 0:
            data[n_planes:, :, :] = data[:-n_planes, :, :]
            data[:n_planes, :, :] = 0
        elif zdim_idx == 2:
            data[:, :, n_planes:] = data[:, :, :-n_planes]
            data[:, :, :n_planes] = 0

    im = write_image(data, p, im)

    return im


def prep_opening(im: Image, key: str, p: dict) -> Image:
    """Perform grayscale opening of the input image.

    footprint:
    """

    data = im.ds[:]

    try:
        footprint = p['selem']
    except KeyError:
        footprint = None
    data = opening(data, footprint=footprint)

    im = write_image(data, p, im)

    return im


def prep_filter(im: Image, key: str, p: dict) -> Image:
    """Filter the input image.

    type : str
        median, gaussian, or dog (difference-of-gaussians)
    inplane: bool
        Perform inplane (xy) smoothing, instead of 3D.
    sigma : float  # TODO list-like
        Smoothing kernel for 'median' and 'gaussian', in voxels.  # TODO: um
    sigma1 : float
        Smoothing kernel for 'dog', in um.
    sigma2 : float
        Smoothing kernel for 'dog', in um.
    """

    data = im.ds[:]

    if p['type'] in ['median', 'gaussian']:

        if p['inplane']:
            data = smooth_channel_inplane(data, p['sigma'], p['type'])
        else:
            data = smooth_channel(data, p['sigma'], p['type'])

    elif p['type'] == 'dog':

        data = smooth_dog(data, im.elsize, p['sigma1'], p['sigma2'])
        data = data.astype('float')

    im = write_image(data, p, im)

    return im


def prep_norm(im: Image, key: str, p: dict) -> Image:
    """Normalize the input between 0 and 1 on percentile.

    perc : list, default [0, 100]
        Percentile values to map to [0, 1].
        Values outside the range are clipped to [0, 1].
    postfix : str
        Suffix the inputpath to save the output to file.
    """

    data = im.ds[:]

    perc = p['perc'] if 'perc' in p.keys() else [0, 100]
    mi, ma = np.percentile(data, perc)
    data = (data - mi) / (ma - mi + np.finfo(float).eps)
    data = np.clip(data, 0, 1)

    im = write_image(data, p, im)

    return im




def mask_volume(block: Block, step_key: str, pars: dict) -> None:
    """
    ids_image
    ods_mask

    image-to-mask ==>
    threshold:
    otsu:
    suavola:

    mask-operations ==>
    size_filter_vx:
    fill:
    erode:

    """

    fundict = {
        'threshold': mask_threshold,
        'otsu': mask_otsu,
        'sauvola': mask_sauvola,
        'size_filter_vx': mask_size_filter_vx,
        'fill': mask_fill,
        'erode': mask_erode,
    }

    im = read_block(block, pars['ids_image'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        im.load()
        im = fundict[fun_key](im, key, p)
        im.close()

    write_block(block, im.ds[:], pars['ods_mask'], im.axlab, 'Mask')

    return im


def mask_threshold(im: Image, key: str, p: dict) -> MaskImage:

    data = im.ds[:]

    if isinstance(p, dict):
        mask = data > p['threshold']
        outpath = gen_outpath(im, p['postfix']) if 'postfix' in p.keys() else ''
    else:
        mask = data > p
        outpath = ''

    im = write_output(mask, outpath, im)

    return im


def mask_otsu(im: Image, key: str, p: dict) -> MaskImage:

    data = im.ds[:]

    if 'clip_range' in p.keys():
        datah = data[data>=p['clip_range'][0]]
        datah = datah[datah<p['clip_range'][1]]
    elif 'perc_range' in p.keys():
        r = np.percentile(data, p['perc_range'])
        datah = data[data>=0]  # FIXME: implement pars['perc_range'][0]
        datah = datah[datah<r[1]]
    else:
        datah = data.flatten()

    thr = threshold_otsu(datah)

    if 'factor' in p.keys():
        thr = thr * p

    mask = data > thr

    im = write_image(mask, p, im)

    return im


def mask_sauvola(im: Image, key: str, p: dict) -> MaskImage:

    data = im.ds[:]

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

    im = write_image(mask, p, im)

    return im


def mask_size_filter_vx(im: Image, key: str, p: dict) -> MaskImage:

    mask = im.ds[:]

    min_size = p['min_size'] if isinstance(p, dict) else p
    mask = remove_small_objects(label(mask), min_size=p).astype('bool')

    im = write_image(mask, p, im)

    return im




def mask_fill(im: MaskImage, key: str, p: dict) -> MaskImage:

    mask = im.ds[:]

    if p == '2D':
        mask_fill = np.zeros_like(mask, dtype='bool')
        for i, slc in enumerate(mask):  # FIXME: assuming zyx here
            mask_fill[i, :, :] = binary_fill_holes(slc)
        mask = mask_fill
    else:
        binary_fill_holes(mask, output=mask)

    im = write_image(mask, p, im)

    return im


def mask_erode(im: MaskImage, key: str, p: dict) -> MaskImage:

    mask = im.ds[:]

    if 'disk' in p.keys():
        disk_erosion = disk(p['disk'])
        mask_ero = np.zeros_like(mask, dtype='bool')
        for i, slc in enumerate(mask):  # FIXME: assuming zyx here
            mask_ero[i, :, :] = binary_erosion(slc, disk_erosion)
    else:
        mask = binary_erosion(mask)

    im = write_image(mask, p, im)

    # FIXME: this may or may not be the desired primary output

    return im


def mask_dilate(im: MaskImage, key: str, p: dict) -> MaskImage:

    mask = im.ds[:]

    if 'disk' in p.keys():
        disk_dilation = disk(p['disk'])
        mask_dil = np.zeros_like(mask, dtype='bool')
        for i, slc in enumerate(mask):  # FIXME: assuming zyx here
            mask_dil[i, :, :] = binary_dilation(slc, disk_dilation)
    elif 'footprint' in p.keys():
        footprint = create_footprint(p['footprint'])
        mask = binary_dilation(mask, selem=footprint)
    else:
        mask = binary_dilation(mask)

    im = write_image(mask, p, im)

    # FIXME: this may or may not be the desired primary output

    return im


def mask_opening(im: MaskImage, key: str, p: dict) -> MaskImage:
    """Perform binary opening of the input image.

    footprint:
    """

    mask = im.ds[:]

    try:
        # footprint = p['selem']
        footprint = create_footprint(p['selem'])
    except KeyError:
        footprint = None
    mask = binary_opening(mask, footprint=footprint)

    im = write_image(mask, p, im)

    return im


def mask_invert(im: MaskImage, key: str, p: dict) -> MaskImage:
    """Return invert of the input mask."""

    mask = im.ds[:]

    mask = ~mask

    im = write_image(mask, p, im)

    return im




def combine_masks(block: Block, step_key: str, pars: dict) -> None:
    """
    ids_nucl
    ids_memb
    ods_mask

    erode
    dilate
    invert

    combine

    opening
    fill

    """

    fundict = {
        'otsu': mask_otsu,
        'sauvola': mask_sauvola,
        'size_filter_vx': mask_size_filter_vx,
        'erode': mask_erode,
        'dilate': mask_dilate,
        'invert': mask_invert,
        'opening': mask_opening,
        'fill': mask_fill,
        'combine': mask_join,
    }

    im_nucl = read_block(block, pars['ids_nucl'])
    im_memb = read_block(block, pars['ids_memb'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        if key.startswith('combine'):
            im = fundict[fun_key](im_nucl, im_memb, key, p)
        elif 'nucl' in key:
            im_nucl.load()
            im_nucl = fundict[fun_key](im_nucl, key, p)
            im_nucl.close()
        elif 'memb' in key:
            im_memb.load()
            im_memb = fundict[fun_key](im_memb, key, p)
            im_memb.close()
        else:
            im.load()
            im = fundict[fun_key](im, key, p)
            im.close()

    write_block(block, im.ds[:], pars['ods_mask'], im.axlab, 'Mask')


def mask_join(im1: MaskImage, im2: MaskImage, key: str, p: dict) -> MaskImage:

    mask1 = im1.ds[:]
    mask2 = im2.ds[:]

    fun = eval(p['fun']) if 'fun' in p.keys() else np.logical_and

    mask = fun(mask1, mask2)

    im = write_image(mask, p, im1)

    return im


def combine_labels(block: Block, step_key: str, pars: dict) -> None:
    """
    ids_labels1
    ids_labels2
    ods_labels

    join
    """

    fundict = {
        'join': label_join,
    }

    im1 = read_block(block, pars['ids_labels1'])
    im2 = read_block(block, pars['ids_labels2'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        im = fundict[fun_key](im_nucl, im_memb, key, p)

    write_block(block, im.ds[:], pars['ods_labels'], im.axlab, 'Label')


def label_join(im1: LabelImage, im2: LabelImage, key: str, p: dict) -> LabelImage:

    labels1 = im1.ds[:]
    labels2 = im2.ds[:]

    labels = join_segmentations(labels1, labels2)

    im = write_image(labels, p, im1)

    return im


def combine_images(block: Block, step_key: str, pars: dict) -> None:
    """
    ids_image1
    ids_image2
    ods_image

    sum
    ...
    etc
    """

    fundict = {
        'sum': image_sum,
    }

    im1 = read_block(block, pars['ids_image1'])
    im2 = read_block(block, pars['ids_image2'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        im = fundict[fun_key](im1, im2, key, p)

    write_block(block, im.ds[:], pars['ods_image'], im.axlab)


def image_sum(im1: Image, im2: Image, key: str, p: dict) -> Image:

    data1 = im1.ds[:]
    data2 = im2.ds[:]

    data = np.add(data1, data2)

    im = write_image(data, p, im1)

    return im




def seed_volume(block: Block, step_key: str, pars: dict) -> None:
    """
    IN:  ids_mask
    OUT: ods_labels

    AUX:
    ids_edt  # TODO
    ids_image
    ids_dog  # TODO

    """

    fundict = {
        'edt': seed_edt,
        'modulate': seed_modulate,
        'threshold': seed_threshold,
        'peaks_window': seed_peaks_window,
        'peaks_blobs': seed_peaks_blobs,
        'label': seed_label,
        'seeds': seed_seeds,
    }

    im_mask = read_block(block, pars['ids_mask'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        # TODO: simplify and streamline  this
        if key.startswith('edt'):
            if 'ids_edt' in p.keys():
                im = read_block(block, p['ids_edt'])
            else:
                im_mask.load()
                im = fundict[fun_key](im_mask, key, p)
                im_mask.close()
        if key.startswith('modulate'):
            im_data = read_block(block, p['ids_image'])
            im.load()
            im_data.load()
            im = fundict[fun_key](im, im_data, key, p)
            im.close()
            im_data.close()
        if key.startswith('threshold'):
            im.load()
            im = fundict[fun_key](im, key, p)
            im.close()
        if key.startswith('peaks_window'):
            im.load()
            im = fundict[fun_key](im, key, p)
            im.close()
        if key.startswith('peaks_blobs'):
            pass
        if key.startswith('label'):
            im.load()
            im = fundict[fun_key](im, key, p)
            im.close()
        if key.startswith('seeds'):
            im_data = read_block(block, p['ids_image'])
            im.load()
            im_data.load()
            im = fundict[fun_key](im, im_data, key, p)
            im.close()
            im_data.close()

    write_block(block, im.ds[:], pars['ods_labels'], im.axlab, 'Label')


def seed_edt(im: MaskImage, key: str, p: dict) -> Image:

    mask = im.ds[:]

    data = distance_transform_edt(mask, sampling=im.elsize)

    if 'threshold' in p.keys():
        data[data > p['threshold']] = 0

    im = write_image(data, p, im)

    return im


def seed_modulate(im1: Image, im2: Image, key: str, p: dict) -> Image:

    edt = im1.ds[:]
    data = im2.ds[:]

    # TODO: option to supply dog directly
    dog = smooth_dog(data, im2.elsize, p['sigma1'], p['sigma2'])

    edt *= normalize_data(dog, a=p['min'], b=p['max'])[0]

    im = write_image(edt, p, im1)  # edt and/or dog?

    return im


def seed_threshold(im: Image, key: str, p: dict) -> Image:
    # TODO: unduplicate with prep_threshold...

    data = im.ds[:]

    if not isinstance(p, dict):
        p = {'threshold': p}
    data[data > p['threshold']] = 0

    im = write_image(data, p, im)

    return im


def seed_peaks_window(im: Image, key: str, p: dict) -> MaskImage:

    data = im.ds[:]

    w = p['window']
    if isinstance(w, list):
        window = w
    elif isinstance(w, int):
        window = [w] * 3
    elif isinstance(w, float):
        window = [int(w / es) * 2 + 1 for es in im.elsize]

    mask, im_max = find_local_maxima(data, window, p['threshold'], 0.001)
    # write(im_max, image_in, '_max', im, 'Image') . # TODO

    im = write_image(mask, p, im)

    return im


def seed_peaks_blobs(im: Image, key: str, p: dict) -> MaskImage:

    data = im.ds[:]

    min_sigmas_vx = [max(1, int(p['sigmas'][0] / es)) for es in im.elsize]
    max_sigmas_vx = [max(1, int(p['sigmas'][1] / es)) for es in im.elsize]

    pad_width = tuple((ms, ms) for ms in max_sigmas_vx)
    data = np.pad(data, pad_width, mode='reflect')

    blobs_log = blob_log(
        data,
        min_sigma=min_sigmas_vx,
        max_sigma=max_sigmas_vx,
        num_sigma=10,
        threshold=p['threshold'],
        )

    dilate_footprint = p['dilate_footprint'] if 'dilate_footprint' in p.keys() else []
    mask = blobs_to_mask(blobs_log, data.shape, dilate_footprint)

    mask = unpad(mask, pad_width)
    # data = unpad(data, pad_width)

    im = write_image(mask, p, im)

    return im


def blobs_to_mask(blobs, shape, dilate_footprint=[]):

    mask = np.zeros(shape, dtype='bool')
    for blob_idx, blob in enumerate(blobs):

        # 3D blob coordinates in voxels bounded to dataset extent
        c = [min(d, max(0, int(k))) for k, d in zip(blob[:3], shape)]
        mask[c[0], c[1], c[2]] = True

        if dilate_footprint:
            # The radius of each blob is approximately ‾√2𝜎 for a 2-D image and ‾√3𝜎 for a 3-D image.
            sigma_blob_z = blob[3]
            footprint = ball(int(sigma_blob_z / 2), dtype='bool')
            mask2 = np.zeros(shape, dtype='bool')
            mask2[c[0], c[1], c[2]] = True
            mask2 = binary_dilation(mask2, selem=footprint)
            mask |= mask2

    return mask


def unpad(data, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return data[tuple(slices)]


def seed_label(im: MaskImage, key: str, p: dict) -> LabelImage:

    mask = im.ds[:]

    labels = ndi.label(mask)[0]

    im = write_image(labels, p, im)

    return im


def seed_seeds(im1: LabelImage, im2: Image, key: str, p: dict) -> LabelImage:

    seeds = im1.ds[:]
    data = im2.ds[:]

    thr = p['threshold'] if 'threshold' in p.keys() else 0
    labels = watershed(-data, seeds, mask=data > thr)

    im = write_image(labels, p, im1)

    return im



#### TRIED starting with passing blocks to the functions.
# def seed_volume(block, step_key, pars):
#     """
#     ids_image
#     ids_mask
#     ids_edt  # TODO
#     ids_dog  # TODO
#     ods_label

#     """

#     # block.create_dataset(pars['ids_image'])
#     # block.datasets[pars['ids_image']].read(from_source=True)

#     # block.create_dataset(pars['ids_mask'])
#     # block.datasets[pars['ids_mask']].read(from_source=True)

#     fundict = {
#         'edt': seed_edt,
#         'modulate': seed_modulate,
#         'edt_threshold': seed_edt_threshold,
#         'peaks': seed_peaks,
#         'label': seed_label,
#         'seeds': seed_seeds,
#     }

#     im_nucl = read_block(block, pars['ids_nucl'])
#     im_memb = read_block(block, pars['ids_memb'])

#     for key, p in pars.items():

#         fun_key = fun_for_step(fundict, key)

#         if fun_key is None:
#             continue

#         im.load()
#         im = fundict[fun_key](block, pars, p)
#         im.close()

#     write_block(block, im.ds[:], pars['ods_mask'], im.axlab, 'Mask')

#     # im.load()
#     # ods, axlab, imtype, datatype = pars['ods_label'], 'zyx', 'Label', 'uint32'
#     # block.create_dataset(
#     #     ods,
#     #     dtype=datatype,
#     #     axlab=axlab,
#     #     imtype=imtype,
#     #     create_image=True,
#     #     )
#     # block.datasets[ods].write(im.ds[:])
#     # im.close()

#     return block


# def seed_edt(block, pars, p):

#     if 'ids_edt' in p.keys():
#         block.create_dataset(p['ids_edt'])
#         block.datasets[p['ids_edt']].read(from_source=True)
#     else:
#         im_mask = block.datasets[pars['ids_mask']].image
#         im_mask.load()
#         mask = im_mask.ds[:]
#         edt = distance_transform_edt(mask, sampling=im.elsize)

#     if 'threshold' in p.keys():
#         edt[edt > p['threshold']] = 0

#     if 'postfix' in p.keys():
#         ids_edt = 'foo'  # gen_outpath(im, p['postfix']) if 'postfix' in p.keys() else ''
#         block.create_dataset(ids_edt)
#         # write to block
#     else:
#         ids_edt = 'edt'
#         block.create_dataset(ids_edt)

#     return block


# def seed_modulate(im, key, p):

#     edt = im.ds[:]

#     dog = smooth_dog(data, elsize, p['sigma1'], p['sigma2'])

#     edt *= normalize_data(dog, a=p['min'], b=p['max'])[0]

#     outpath = gen_outpath(im, p['postfix']) if 'postfix' in p.keys() else ''
#     im = write_output(edt, outpath, im, imtype='Image')  # edt and/or dog?

#     return im




#### THIS IS FOR FGS: TRYING 2D SEEDS: TODO: CHECK IF THIS IS STILL ON THE BRAIN

#         # find blobs in the data  # TODO: try on edt
#         elif '2Dseeds' in p.keys():  # for FGS project
#             from skimage.feature import blob_log

#             def round_up_to_odd(f):
#                 return np.ceil(f) // 2 * 2 + 1

#             def blobs_to_mask(blobs, shape, dilate_footprint=[], idxs_max_proj=[]):

#                 seeds = np.zeros(shape, dtype='uint32')
#                 mask = np.zeros(shape, dtype='bool')
#                 for blob_idx, blob in enumerate(blobs_log):

#                     # 3D blob coordinates in voxels bounded to dataset extent
#                     c = [min(d, max(0, int(k))) for k, d in zip(blob[:2], shape[1:3])]
#                     z_idx = idxs_max_proj[c[0], c[1]]
#                     mask[z_idx, c[0], c[1]] = True

#                     if dilate_footprint:

#                         # The radius of each blob is approximately ‾√2𝜎 for a 2-D image and ‾√3𝜎 for a 3-D image.
#                         radius_blob_y = blob[2] * np.sqrt(2)
#                         radius_blob_x = blob[3] * np.sqrt(2)
#                         footprint_size = [
#                             5,
#                             int(round_up_to_odd(radius_blob_y*2)),
#                             int(round_up_to_odd(radius_blob_x*2)),
#                             ]
#                         footprint = create_footprint(footprint_size)

#                         mask2 = np.zeros(shape, dtype='bool')
#                         mask2[z_idx, c[0], c[1]] = True
#                         mask2 = binary_dilation(mask2, footprint=footprint)
#                         mask |= mask2
#                         seeds[mask2] = blob_idx + 1

#                 return mask, seeds

#             def unpad(data, pad_width):
#                 slices = []
#                 for c in pad_width:
#                     e = None if c[1] == 0 else -c[1]
#                     slices.append(slice(c[0], e))
#                 return data[tuple(slices)]

#             #min_sigmas_vx = [max(1, int(p['sigmas'][0] / es)) for es in im.elsize]
#             #max_sigmas_vx = [max(1, int(p['sigmas'][1] / es)) for es in im.elsize]
#             min_sigmas_vx = [7, 7, 7]
#             max_sigmas_vx = [11, 11, 11]

#             # Create maximaum intensity projection over z
#             pad_width = tuple((ms, ms) for ms in max_sigmas_vx)
#             data = np.pad(data, pad_width, mode='reflect')

#             data_max_proj = np.max(data, axis=im.axlab.index('z'))
#             idxs_max_proj = np.argmax(data, axis=im.axlab.index('z'))

#             # Detect blobs in 2D image
#             blobs_log = blob_log(
#                 data_max_proj,
#                 min_sigma=min_sigmas_vx[1:],
#                 max_sigma=max_sigmas_vx[1:],
#                 num_sigma=10,
#                 threshold=p['threshold'],
# #                threshold_rel=p['threshold'],
#                 )

#             # Project the blob coordinate to the index of the max value in Z
#             dilate_footprint = []
#             if 'dilate' in p.keys():
#                 dilate_footprint = p['dilate']['footprint']
#             mask, seeds = blobs_to_mask(blobs_log, data.shape, dilate_footprint, idxs_max_proj)
#             mask = unpad(mask, pad_width)
#             seeds = unpad(seeds, pad_width)
#             data = unpad(data, pad_width)
#             edt = data


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
        mask, im_max = find_local_maxima(edt, p['window'], p['threshold'])
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


def segment_volume(block: Block, step_key: str, pars: dict) -> None:
    """
    IN:  ids_labels
    OUT: ods_labels

    AUX:
    ids_image
    ids_mask

    """

    fundict = {
        'negative_seed': segment_negative_seed,
        'slic': segment_slic,
        'watershed': segment_watershed,
    }

    im = read_block(block, pars['ids_labels'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        # TODO: simplify and streamline  this
        if key.startswith('negative_seed'):
            im_neg = read_block(block, p['ids_mask'])
            im.load()
            im_neg.load()
            im = fundict[fun_key](im, im_neg, key, p)
            im_neg.close()
            im.close()
        if key.startswith('slic'):
            im_data = read_block(block, p['ids_image'])
            im_mask = read_block(block, p['ids_mask'])
            im_data.load()
            im_mask.load()
            im = fundict[fun_key](im_data, im_mask, key, p)
            im_mask.close()
            im_data.close()
        if key.startswith('watershed'):
            im_data = read_block(block, p['ids_image'])
            if 'ids_mask' in p.keys():
                im_mask = read_block(block, p['ids_mask'])
                im_mask.load()
            else:
                im_mask = None
            im_data.load()
            im = fundict[fun_key](im_data, im, im_mask, key, p)
            im_data.close()
            if 'ids_mask' in p.keys():
                im_mask.close()

    write_block(block, im.ds[:], pars['ods_labels'], im.axlab, 'Label')


def segment_negative_seed(im1: LabelImage, im2: MaskImage, key: str, p: dict) -> LabelImage:
    # TODO: maybe move to seed step

    labels = im1.ds[:]
    mask = im2.ds[:]

    labels[~mask] = max(np.unique(labels)) + 1

    im = write_image(labels, p, im1)

    return im


def segment_slic(im1: Image, im2: MaskImage, key: str, p: dict) -> LabelImage:
    # TODO: this could simply be done in skimage external function
    # TODO: kwargs

    data = im1.ds[:]
    mask = im2.ds[:]

    ws = slic(
        np.expand_dims(data, 3),
        n_segments=100,
        compactness=10.0,
        max_num_iter=10,
        sigma=0,
        spacing=None,
        multichannel=True,
        convert2lab=None,
        enforce_connectivity=True,
        min_size_factor=0.5,
        max_size_factor=3,
        slic_zero=False,
        start_label=1,
        mask=mask,
        channel_axis=3,
    )

    im = write_image(labels, p, im1)

    return im


def segment_watershed(im1: Image, im2: LabelImage, im3: MaskImage, key: str, p: dict) -> LabelImage:

    data = im1.ds[:]
    labels = im2.ds[:]
    if im3 is not None:
        mask = im3.ds[:]
    else:
        mask = None

    elsize = np.absolute(im1.elsize)
    labels = segment_volume_data(p, data, labels, mask, elsize)

    im = write_image(labels, p, im1)

    return im


def segment_volume_data(p, data, seeds, mask, elsize):

    if 'invert_data' in p.keys():
        data = -data

    if 'edt_data' in p.keys():
        data = distance_transform_edt(data.astype('bool'), sampling=elsize)

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
        if compactness:
            print('WARNING: possibly not using correct spacing for compact watershed')
        ws = watershed(data, seeds, mask=mask, compactness=compactness)

        return ws


def filter_segments(block: Block, step_key: str, pars: dict) -> None:
    """
    IN:  ids_labels
    OUT: ods_labels

    AUX:
    ids_image
    ids_mask

    """

    fundict = {
        'mask_labels': filter_mask_labels,
        'delete_labels': filter_delete_labels,
        'min_size': filter_min_size,
        'max_size': filter_max_size,
        'expand': filter_expand_labels,
        'upsample': filter_upsample,
    }

    im = read_block(block, pars['ids_labels'])

    for key, p in pars.items():

        fun_key = fun_for_step(fundict, key)

        if fun_key is None:
            continue

        labelset = set([])

        # # TODO: simplify and streamline  this
        if key.startswith('mask_labels') or key.startswith('delete_labels'):
            im_mask = read_block(block, p['ids_mask'])
            im_mask.load()
            im = fundict[fun_key](im, im_mask, key, p)
            im_mask.close()
            im.close()
        elif key.startswith('upsample'):
            im_ref = read_block(block, p['ids_ref'])
            im_ref.load()
            im = fundict[fun_key](im, im_ref, key, p)
            im_ref.close()
            im.close()
        else:
            im.load()
            im = fundict[fun_key](im, key, p)
            im.close()

    write_block(block, im.ds[:], pars['ods_labels'], im.axlab, 'Label')


def filter_mask_labels(im1: LabelImage, im2: MaskImage, key: str, p: dict) -> LabelImage:

    labels = im1.ds[:]
    mask = im2.ds[:]

    if 'invert' in p.keys():
        mask = ~mask
    labels[mask] = 0

    im = write_image(labels, p, im1)

    return im


def filter_delete_labels(im1: LabelImage, im2: MaskImage, key: str, p: dict) -> LabelImage:

    labels = im1.ds[:]
    mask = im2.ds[:]

    labelset = set([])


    if 'invert' in p.keys():
        mask = ~mask

    if 'threshold' in p.keys():
        delete_labels = set(np.argwhere(np.bincount(labels[mask]) > p['threshold']).flatten())
    else:  # all labels overlapping mask
        delete_labels = set(np.unique(labels[mask]))

    delete_labels -= set([0])
    labelset |= delete_labels


    labels = forward_map_labelset(labels, labelset)

    im = write_image(labels, p, im1)

    return im


def filter_min_size(im: LabelImage, key: str, p: dict) -> LabelImage:

    labels = im.ds[:]

    labelset = set([])

    p = {'min_size': p} if not isinstance(p, dict) else p
    nvoxels = int(p['min_size'] / np.prod(im.elsize))
    bc = np.bincount(np.ravel(labels))
    delete_labels = set(np.where(bc < nvoxels)[0])
    labelset |= delete_labels

    labels = forward_map_labelset(labels, labelset)

    im = write_image(labels, p, im)

    return im


def filter_max_size(im: LabelImage, key: str, p: dict) -> LabelImage:

    labels = im.ds[:]

    labelset = set([])

    p = {'max_size': p} if not isinstance(p, dict) else p
    nvoxels = int(p['max_size'] / np.prod(im.elsize))
    bc = np.bincount(np.ravel(labels))
    delete_labels = set(np.where(bc > nvoxels)[0])
    labelset |= delete_labels

    labels = forward_map_labelset(labels, labelset)

    im = write_image(labels, p, im)

    return im


def forward_map_labelset(labels, labelset):

    if labelset:
        maxlabel = max(np.unique(labels)).astype('uint32')
        if maxlabel:
            fwmap = [True if l in labelset else False for l in range(0, maxlabel + 1)]
            labels[np.array(fwmap).astype('bool')[labels]] = 0

    return labels


def filter_expand_labels(im: LabelImage, key: str, p: dict) -> LabelImage:

    labels = im.ds[:]

    labels = expand_labels(labels, im.elsize, p['expand'])

    im = write_image(labels, p, im)

    return im


def filter_upsample(im1: LabelImage, im2: Image, key: str, p: dict) -> LabelImage:

    labels = im1.ds[:]
    # ref = im2.ds[:]

    order = p['order'] if 'order' in p.keys() else 0

    if 'ids_ref' in p.keys():
        labels = resize(labels, im2.dims, preserve_range=True, order=order)
    elif 'factors' in p.keys():
        factors = [p['factors'][al] for al in im1.axlab]
        labels = rescale(labels, factors, order=order)

    # TODO: edit the im_props!?

    im = write_image(labels, p, im1)

    return im


def resegment_largelabels(filepath, step_key, pars):

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

        im = Image('{}/{}'.format(filepath, ids), permission='r')
        im.load()
        vol = im.slice_dataset()
        im.close()

        mask_memb = vol > ero_val
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


def expand_labels(label_image, sampling, distance=1):
    # adapted for anisotropy from scikit-image expand_labels

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

    return peaks, image_max


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

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        stem = self._build_path(moduledir=self._logdir)
        bpat = self.outputpaths['blockinfo']['blockfiles']  #self._l2p(['blocks', '{f}.h5'])
        bmat = self._pat2mat(bpat)  # <>_blocks_B{b:05d}.h5  => <>_blocks_B*.h5
        # TODO: for filepaths, blocks, ...

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

    def _estimate_block(self, block_idx):
        """Subdivide cells in compartments."""

        block = self._blocks[block_idx]

        inputs = self._prep_paths_blockfiles(self.inputs, block)
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        step = 'estimate'
        params = self._cfg[self.step_id][step]

        filepath = inputs['blockfiles']

        label_im = LabelImage(f'{filepath}/{self.ids_labels}')
        label_im.load(load_data=False)
        try:
            del label_im.file[self.ods_full]
        except KeyError:
            pass
        label_im.file[self.ods_full] = label_im.file[self.ids_labels]
        labels = label_im.slice_dataset()
        label_im.close()

        def read_mask(filepath, ids):
            im = MaskImage(f'{filepath}/{ids}', permission='r')
            im.load()
            mask = im.slice_dataset().astype('bool')
            im.close()
            return mask

        def write_masked(label_im, filepath, ods, labels, mask):
            labs = np.copy(labels)
            labs[~mask.astype('bool')] = 0
            write(labs, f'{filepath}/{ods}', '', label_im, imtype='Label')

        if self.ods_memb:
            if self.ids_memb_mask:
                mask_memb = read_mask(filepath, self.ids_memb_mask)
            else:
                mask_memb = memb_mask(labels, self.footprint_memb)
            write_masked(label_im, filepath, self.ods_memb, labels, mask_memb)

        if self.ods_nucl:
            if self.ids_nucl_mask:
                mask_nucl = read_mask(filepath, self.ids_nucl_mask)
            else:
                mask_nucl = nucl_mask(labels)
            write_masked(label_im, filepath, self.ods_nucl, labels, mask_nucl)

        if self.ods_csol:
            if self.ids_csol_mask:
                mask_csol = read_mask(filepath, self.ids_csol_mask)
            else:
                mask_csol = csol_mask(labels, mask_nucl, mask_memb)
            write_masked(label_im, filepath, self.ods_csol, labels, mask_csol)

        # self.dump_parameters(self.step, outputs['parameters'])
        #self.report(outputs['report'], inputs=inputs, outputs=outputs)

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
            input = self._blocks[input].path
        else:
            input = input or [0, 1]

        super().view(input, images, labels, settings)


if __name__ == "__main__":
    main(sys.argv[1:])
