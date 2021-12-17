#!/usr/bin/env python

"""Segment nuclei by stardist DL.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import numpy as np

from glob import glob

from csbdeep.utils import normalize, normalize_mi_ma, axes_check_and_normalize, axes_dict

from stardist import fill_label_holes, calculate_extents, Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D
from stardist.big import _grid_divisible, BlockND, OBJECT_KEYS

from stapl3d import parse_args, Stapl3r, Image
from stapl3d.blocks import Block3r
from stapl3d.segmentation import zipping

logger = logging.getLogger(__name__)


def main(argv):
    """Segment nuclei with StarDist."""

    steps = ['train', 'predict', 'merge']
    args = parse_args('stardist', steps, *argv)

    stardist3r = StarDist3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        stardist3r._fun_selector[step]()


class StarDist3r(Block3r):
    """Segment nuclei with StarDist."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'stardist'

        super(StarDist3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'train': self.train,
            'predict': self.predict,
            'merge': self.merge,
            })

        self._parallelization.update({
            'train': [],
            'predict': ['blocks'],
            'merge': [],
            })

        self._parameter_sets.update({
            'train': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            'predict': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            'merge': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        # TODO
        self._parameter_table.update({})

        # TODO: parameters controlling stardist training epochs
        default_attr = {
            'modelname': 'stardist',
            'ids_raw': 'raw_nucl',
            'ids_lbl': 'label_nucl',
            'train_patch_size': (40, 96, 96),
            'axis_norm': (0, 1, 2),
            'grid': (),
            'n_rays': 96,
            'augmenter': {
                'rotflip_axis': (1, 2),
                'int_fac': [0.8, 1.2],
                'int_add': [-0.1, 0.1],
                'scale_axis': (),
                'scale_fac': [],
                },
            'normalizer_intensities': [],
            'normalizer_percentages': [1.0, 99.8],
            'config': {},
            'axes': 'ZYX',
            'block_size': [None, None, None],
            'min_overlap': [32, 128, 128],
            'context': [0, 64, 64],
            'ids_label': 'labels',
            'print_nblocks': False,
            'blocks': [],
            '_blocks': [],
            'channel': None,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_stardist()

        self._init_log()

        self._prep_blocks_stardist()

        self._images = []
        self._labels = []

    def _init_paths_stardist(self):

        self._paths.update({
            'train': {
                'inputs': {
                    'traindir': ['stardist', 'data', 'train'],
                    'valdir': ['stardist', 'data', 'val'],
                    },
                'outputs': {
                    'modeldir': ['stardist', 'models'],
                    }
                },
            'predict': {
                'inputs': {
                    'data': '.',
                    'modeldir': ['stardist', 'models'],
                    },
                'outputs': {
                    'nblocks': ['blocks_stardist', 'nblocks.txt'],
                    'blockfiles': ['blocks_stardist', '{f}.h5'],
                    }
                },
            'merge': {
                'inputs': {
                    'blockfiles': ['blocks_stardist', '{f}.h5'],
                    'data': '.',
                    },
                'outputs': {
                    'prediction': 'stardist_prediction.h5/stardist',
                    'polys': 'stardist_prediction.pickle',
                    }
                },
            })

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def train(self, **kwargs):
        """Train StarDist model."""

        arglist = self._prep_step('train', kwargs)
        self._train_model()

    def _train_model(self):
        """Train StarDist model."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        X_trn, Y_trn = self._load_data_directory(inputs['traindir'])
        X_val, Y_val = self._load_data_directory(inputs['valdir'])

        n_channels = 1 if X_trn[0].ndim == 3 else X_trn[0].shape[-1]

        self._set_stardist_config(n_channels, Y_trn + Y_val)

        model = StarDist3D(self.config, name=self.modelname, basedir=outputs['modeldir'])

        model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)

        model.optimize_thresholds(X_val, Y_val)

    def _load_data_directory(self, datadir):
        """Load data/label pairs of all hdf5 files in a directory."""

        X, Y = [], []
        filepaths = sorted(glob(os.path.join(datadir, '*.h5')))
        for filepath in filepaths:
            X.append(load_data(filepath, self.ids_raw))
            Y.append(load_data(filepath, self.ids_lbl))

        X = [self._normalize_stardist_data(x) for x in X]

        Y = [fill_label_holes(y) for y in Y]

        return X, Y

    def _set_stardist_config(self, n_channels, Y):
        """Set the StarDist configuration."""

        extents = calculate_extents(Y)
        anisotropy = tuple(np.max(extents) / extents)
        grid = self.grid or tuple(1 if a > 1.5 else 2 for a in anisotropy)
        rays = Rays_GoldenSpiral(self.n_rays, anisotropy=anisotropy)
        self.config = Config3D (
            rays             = rays,
            grid             = grid,
            anisotropy       = anisotropy,
            use_gpu          = False,
            n_channel_in     = n_channels,
            train_patch_size = self.train_patch_size,
            train_batch_size = 2,
            )

    def print_nblocks(self, **kwargs):
        """Predict nuclei with StarDist model."""

        arglist = self._prep_step('print_nblocks', kwargs)
        self._print_nblocks(0)

    def _print_nblocks(self):
        """Print the number of blocks to file (for parallel hpc prediction)."""

        blockdir = os.path.dirname(self.outputs['blockfiles'])
        os.makedirs(blockdir, exist_ok=True)
        with open(os.path.join(blockdir, 'nblocks.txt'), 'w') as f:
            f.write(str(len(self._blocks)))

    def predict(self, **kwargs):
        """Predict nuclei with StarDist model."""

        arglist = self._prep_step('predict', kwargs)

        if self.print_nblocks:
            self._print_nblocks()
            return

#        with multiprocessing.Pool(processes=self._n_workers) as pool:
#            pool.starmap(self._predict_block, arglist)
        for block_idx in self.blocks:
            self._predict_block(block_idx)

    def _predict_block(self, block_idx):
        """Predict nuclei with StarDist model for a block."""

        filepath, block = self._blocks[block_idx]
        filestem = os.path.basename(os.path.splitext(filepath)[0])

        inputs = self._prep_paths(self.inputs)

        reps = self._find_reps(self.outputs['blockfiles'], filestem, block_idx)
        # reps = {'f': filestem, 'b': block_idx}
        outputs = self._prep_paths(self.outputs, reps=reps)

        blockdir = os.path.dirname(outputs['blockfiles'])
        os.makedirs(blockdir, exist_ok=True)
        blockstem = outputs['blockfiles'].split('.h5')[0]

        model = load_model(inputs['modeldir'], self.modelname)

        # Load data
        im = Image(filepath, permission='r')
        im.load(load_data=False)
        comps = im.split_path()
        if filepath.endswith('.ims'):
            h5ds = im.file['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
        elif '.h5' in filepath:
            h5ds = im.file[comps['int']]
        else:  # czi
            if self.channel is not None:
                im.slices[im.axlab.index('c')] = slice(self.channel, self.channel + 1)
            h5ds = im.slice_dataset()

        props = im.get_props()
        for al in 'ct':
            if al in im.axlab:
                props = im.squeeze_props(props, dim=im.axlab.index(al))

        data = block.read(h5ds, axes=self.axes)
        data = self._normalize_stardist_data(data)

        labels, polys = self._run_prediction(model, block, data)

        write_labels(f'{blockstem}.h5/{self.ids_lbl}', props, labels)
        with open(f'{blockstem}.pickle', 'wb') as f:
            pickle.dump([model._axes_out.replace('C', ''), polys, block], f)

        im.close()

    def _run_prediction(self, model, block, data):
        """Run stardist prediction on block."""

        axes_out = model._axes_out.replace('C', '')

        labels, polys = model.predict_instances(data)
        labels = block.crop_context(labels, axes=axes_out)
        labels, polys = block.filter_objects(labels, polys, axes=axes_out)

        return labels, polys

    def merge(self, **kwargs):
        """Merge stardistblocks to single volume."""

        arglist = self._prep_step('merge', kwargs)
        self._mergeblocks()

    def _mergeblocks(self):

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        filemat = self._pat2mat(inputs['blockfiles'], mat='*')
        blockpaths = sorted(glob(filemat))

        maxlabels = zipping.get_maxlabels_from_attribute(blockpaths, self.ids_lbl, '')

        im = Image(inputs['data'], permission='r')
        im.load(load_data=False)
        props = im.get_props()
        props['dtype'] = 'int32'
        for al in 'ct':
            if al in im.axlab:
                props = im.squeeze_props(props, dim=im.axlab.index(al))
        im.close()

        mo = Image(outputs['prediction'], **props)
        mo.create()

        polys_all = {}
        for blockpath in blockpaths:

            # read polys
            picklepath = blockpath.replace('.h5', '.pickle')
            with open(picklepath, 'rb') as f:
                (axes_out, polys, block) = pickle.load(f)
            for k,v in polys.items():
                polys_all.setdefault(k,[]).append(v)

            # read labels
            h5_path = f'{blockpath}/{self.ids_lbl}'
            im = Image(h5_path, permission='r')
            im.load()
            labels = im.slice_dataset().astype(props['dtype'])
            im.close()

            # relabel labels
            maxlabel = np.sum(maxlabels[:block.id]).astype(props['dtype'])
            bg_label = 0
            mask = labels == bg_label
            labels[~mask] += maxlabel

            # write labels
            block.write(mo.ds, labels, axes=axes_out)

        mo.ds.attrs.create('maxlabel', maxlabel, dtype='uint32')
        mo.close()

        polys_all = {k: (np.concatenate(v) if k in OBJECT_KEYS else v[0])
                     for k,v in polys_all.items()}
        with open(outputs['polys'], 'wb') as f:
            pickle.dump(polys_all, f)

    def _prep_blocks(self):
        pass

    def _prep_blocks_stardist(self):
        """Generate StarDist blocks for parallel processing."""

        inputs = self._prep_paths(self.inputpaths['predict'])

        try:
            model = load_model(inputs['modeldir'], self.modelname)
        except:
            return

        inpaths = inputs['data']
        if '{b' in inpaths or '{f' in inpaths:
            self.filepaths = self.get_filepaths(inpaths)
        elif os.path.isdir(inpaths):  # TODO: flexible extension
            self.filepaths = sorted(glob(os.path.join(inpaths, '*.czi')))
        else:
            # should be a single file (or a list of files???)
            self.filepaths = [inpaths]

        self._blocks = []
        for filepath in self.filepaths:

            im = Image(filepath, permission='r')
            im.load(load_data=False)
            dims = [im.dims[im.axlab.index(d) for d in self.axes.lower()]
            im.close()

            blocks = stardist_blocks(model, dims, self.axes, self.block_size, self.min_overlap, self.context)
            for b in blocks:
                self._blocks.append((filepath, b))

    def _normalize_stardist_data(self, x):

        if self.normalizer_intensities:
            mi = np.array([[[self.normalizer_intensities[0]]]])
            ma = np.array([[[self.normalizer_intensities[1]]]])
            x = normalize_mi_ma(x, mi, ma)
        elif self.normalizer_percentages:
            mi = self.normalizer_percentages[0]
            ma = self.normalizer_percentages[1]
            x = normalize(x, mi, ma, axis=self.axis_norm)

        return x


def random_fliprot(img, mask, axis=None):
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric

    if axis is None:
        return img, mask
        # axis = tuple(range(mask.ndim))

    axis = tuple(axis)

    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)

    return img, mask


def random_intensity_change(img, fac=[], add=[]):

    if fac:
        img *= np.random.uniform(fac[0], fac[1])
    if add:
        img += np.random.uniform(add[0], add[1])

    return img


def random_scale(img, mask, axis=None, fac=[]):

    # TODO# np.random.uniform(0.8, 1.5)
    # np.random.uniform(fac[0], fac[1])
    return img, mask


def augmenter(x, y,
              rotflip_axis=(),
              int_fac=[], int_add=[],
              scale_axis=[], scale_fac=[],
              ):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y, rotflip_axis)
    x = random_intensity_change(x)
    x, y = random_scale(x, y, scale_axis, scale_fac)
    return x, y


def stardist_normalization_range(image_in, pmin=1, pmax=99.8, axis_norm=(0, 1, 2)):
    """Find percentiles of dataset (for big dataset normalization presets)."""

    im = Image(image_in)
    im.load()
    props = im.get_props()
    X = im.slice_dataset()
    im.close()

    mi = np.percentile(X, pmin, axis=axis_norm, keepdims=True)
    ma = np.percentile(X, pmax, axis=axis_norm, keepdims=True)

    return mi, ma


def stardist_blocks(model, imdims, axes='ZYX', block_size=[None, 1024, 1024], min_overlap=[32, 128, 128], context=[0, 64, 64]):
    """Generate StarDist blocks for parallel processing."""

    # TODO: dataset-dependent automatic blocksize

    n = len(axes)
    dims = imdims[:n]

    block_size = [b if b is not None else d for b, d in zip(block_size, dims)]

    assert 0 <= min_overlap[0] + 2 * context[0] < block_size[0] <= dims[0]
    assert 0 <= min_overlap[1] + 2 * context[1] < block_size[1] <= dims[1]
    assert 0 <= min_overlap[2] + 2 * context[2] < block_size[2] <= dims[2]

    axes = axes_check_and_normalize(axes, length=n)
    grid = model._axes_div_by(axes)
    axes_out = model._axes_out.replace('C', '')

    if context is None:
        context = model._axes_tile_overlap(axes)

    if np.isscalar(block_size):
        block_size  = n * [block_size]
    if np.isscalar(min_overlap):
        min_overlap = n * [min_overlap]
    if np.isscalar(context):
        context = n * [context]
    block_size, min_overlap, context = list(block_size), list(min_overlap), list(context)
    assert n == len(block_size) == len(min_overlap) == len(context)

    if 'C' in axes:
        i = axes_dict(axes)['C']
        block_size[i] = imdims[i]
        min_overlap[i] = context[i] = 0

    block_size = tuple(_grid_divisible(g, v, name='block_size', verbose=False)
                       for v, g, a in zip(block_size, grid, axes))
    min_overlap = tuple(_grid_divisible(g, v, name='min_overlap', verbose=False)
                        for v, g, a in zip(min_overlap, grid, axes))
    context = tuple(_grid_divisible(g, v, name='context', verbose=False)
                    for v,g,a in zip(context, grid, axes))

    # Define blocks
    blocks = BlockND.cover(dims, axes, block_size, min_overlap, context, grid)

    return blocks


def load_model(self, modeldir, modelname):
    """Load StarDist model."""

    if modelname in ['3D_demo']:
        model = StarDist3D.from_pretrained(modelname)
    else:
        model = StarDist3D(None, name=modelname, basedir=modeldir)

    return model


def load_data(filepath, ids, z_range=[]):
    """Load data from file."""

    im = Image(f'{filepath}/{ids}', permission='r')
    im.load()
    z_slc = slice(z_range[0], z_range[1]) if z_range else slice(None)
    im.slices[im.axlab.index('z')] = z_slc
    data = im.slice_dataset()
    im.close()

    return data


def write_labels(filepath, props, labels):
    """Write labels to file."""

    props['shape'] = labels.shape
    mo = Image(filepath, **props)
    mo.create()
    mo.write(labels)
    mo.ds.attrs.create('maxlabel', np.amax(labels), dtype='uint32')
    mo.close()


if __name__ == "__main__":
    main(sys.argv[1:])
