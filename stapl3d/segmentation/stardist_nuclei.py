#!/usr/bin/env python

"""Segment nuclei by stardist DL.

"""

from __future__ import print_function, unicode_literals, absolute_import, division

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import numpy as np

from glob import glob

import warnings
import math
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path

from csbdeep.models.base_model import BaseModel
from csbdeep.utils import Path, normalize, normalize_mi_ma, _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
from csbdeep.utils.tf import export_SavedModel, keras_import, IS_TF_1, CARETensorBoard
from csbdeep.internals.predict import tile_iterator
from csbdeep.internals.train import RollingSequence
from csbdeep.data import Resizer

from stardist import random_label_cmap, fill_label_holes, calculate_extents, Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D, StarDistData3D
from stardist.sample_patches import get_valid_inds
from stardist.utils import _is_power_of_2, optimize_threshold
from stardist.matching import relabel_sequential
from stardist.big import _grid_divisible, BlockND, OBJECT_KEYS#, repaint_labels

import tensorflow as tf
K = keras_import('backend')
Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')

from stapl3d import Image
from stapl3d.pipelines import stardist_testing_library


def stardist_train(basedir, modelname='stardist'):

    datadir = os.path.join(basedir, 'data')
    modeldir = os.path.join(basedir, 'models')

    runs = stardist_testing_library.get_runs()
    trainsets = stardist_testing_library.get_trainsets()

    run = runs[modelname]

    td = {}
    for trainset_name in run['trainsets']:
        ts = trainsets[trainset_name]
        td[trainset_name] = get_training_data(datadir, **ts, **run)

    X_trn, Y_trn, X_val, Y_val, i_trn, i_val = [], [], [], [], [], []
    for tsname, (X_t, X_v, Y_t, Y_v, i_t, i_v) in td.items():
        X_trn += X_t
        X_val += X_v
        Y_trn += Y_t
        Y_val += Y_v
        i_trn += list(i_t)
        i_val += list(i_v)

    n_channels = 1 if X_trn[0].ndim == 3 else X_trn[0].shape[-1]

    conf = get_config(run, n_channels, Y_trn + Y_val)
    model = StarDist3D(conf, name=modelname, basedir=modeldir)

    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    model.optimize_thresholds(X_val, Y_val)

    return model


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


def get_training_data(datadir, filestem_train, h5_path_pat, z_range, trainblock_list, ids_dapi, ids_label, ids_memb, use_memb, axis_norm, ind_val, normalizer, **kwargs):
    X, Y = [], []
    for trainblock in trainblock_list:
        z_slc = slice(z_range[0], z_range[1])
        h5_path = h5_path_pat.format(filestem_train, trainblock)
        img_path = os.path.join(datadir, '{}/{}'.format(h5_path, ids_dapi))
        img_im = Image(img_path, permission='r')
        img_im.load(load_data=True)
        data = img_im.ds[z_slc, :, :]
        if use_memb:
            img_path = os.path.join(datadir, '{}/{}'.format(h5_path, ids_memb))
            img_im.path = img_path
            img_im.load(load_data=True)
            data = np.stack([data, img_im.ds[z_slc, :, :]], axis=-1)
        X.append(data)
        img_im.close()

        lbl_path = os.path.join(datadir, '{}/{}'.format(h5_path, ids_label))
        lbl_im = Image(lbl_path, permission='r')
        lbl_im.load(load_data=True)
        Y.append(lbl_im.ds[z_slc, :, :])
        lbl_im.close()

    if not normalizer:
        X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
    else:
        X = [normalize_mi_ma(x, normalizer[0], normalizer[1]) for x in X]
    Y = [fill_label_holes(y) for y in Y]

    X_trn, Y_trn, X_val, Y_val, ind_train, ind_val = split_training_data(X, Y, ind_val)

    return X_trn, X_val, Y_trn, Y_val, ind_train, ind_val


def split_training_data(X, Y, ind_val=[]):
    if ind_val:
        ind_train = [ind for ind in range(len(X)) if ind not in ind_val]
    else:
        rng = np.random.RandomState(42)
        ind = rng.permutation(len(X))
        n_val = max(1, int(round(0.15 * len(ind))))
        ind_train, ind_val = ind[:-n_val], ind[-n_val:]

    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]

    return X_trn, Y_trn, X_val, Y_val, ind_train, ind_val


def get_config(run, n_channels, Y):

    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    grid = run['grid'] or tuple(1 if a > 1.5 else 2 for a in anisotropy)
    rays = Rays_GoldenSpiral(run['n_rays'], anisotropy=anisotropy)
    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = False,
        n_channel_in     = n_channels,
        train_patch_size = run['train_patch_size'],
        train_batch_size = 2,
        )

    return conf


def stardist_normalization_range(image_in, pmin=1, pmax=99.8, axis_norm=(0, 1, 2)):

    im = Image(image_in)
    im.load()
    props = im.get_props()
    X = im.slice_dataset()
    im.close()

    mi = np.percentile(X, pmin, axis=axis_norm, keepdims=True)
    ma = np.percentile(X, pmax, axis=axis_norm, keepdims=True)

    return mi, ma


def stardist_blocks(model, imdims, axes='ZYX', block_size=[106, 1024, 1024], min_overlap=[32, 128, 128], context=[0, 64, 64]):

    n = len(axes)
    dims = imdims[:n]

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


def stardist_predict(basedir, modelname, image_in, idx, normalization=[],
                     axes = 'ZYX', block_size=[106, 1024, 1024], min_overlap=[32, 128, 128], context=[0, 64, 64],
                     ids_label='labels', outputstem='', print_nblocks=False):

    # Load model
    if modelname in ['3D_demo']:
        model = StarDist3D.from_pretrained(modelname)
        #model.optimize_thresholds(X_val, Y_val)
    else:
        modeldir = os.path.join(basedir, 'models')
        model = StarDist3D(None, name=modelname, basedir=modeldir)

    # Load data
    im = Image(image_in, permission='r')
    im.load(load_data=False)
    if image_in.endswith('.ims'):
        h5ds = im.file['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
        props = im.squeeze_props(im.squeeze_props(dim=4), dim=3)

    # Outputdir
    comps = im.split_path()
    blockdir = os.path.join(comps['dir'], 'blocks_stardist')
    os.makedirs(blockdir, exist_ok=True)
    outputstem = outputstem or os.path.join(blockdir, comps['fname'])

    # Blocks
    blocks = stardist_blocks(model, im.dims, axes, block_size, min_overlap, context)
    if print_nblocks:
        with open(os.path.join(blockdir, 'nblocks.txt'), 'w') as f:
            f.write("{}".format(len(blocks)))
        im.close()
        return

    block = blocks[idx]
    blockstem = os.path.join(blockdir, '{}_block{:05d}'.format(outputstem, block.id))
    print('# of blocks: {}; processing block: {:05d}'.format(len(blocks), idx))

    # Normalize
    if not normalization:
        img = normalize(block.read(h5ds, axes=axes), 1, 99.8, axis=axis_norm)
    else:
        mi, ma = np.array([[[normalization[0]]]]), np.array([[[normalization[1]]]])
        img = normalize_mi_ma(block.read(h5ds, axes=axes), mi, ma)

    # Predict
    axes_out = model._axes_out.replace('C', '')
    predict_kwargs = {}
    labels, polys = model.predict_instances(img, **predict_kwargs)
    # print(labels.shape, len(np.unique(labels)), len(polys))
    labels = block.crop_context(labels, axes=axes_out)
    # print(labels.shape, len(np.unique(labels)), len(polys))
    labels, polys = block.filter_objects(labels, polys, axes=axes_out)
    # print(labels.shape, len(np.unique(labels)), len(polys))

    # Save
    maxlabel = np.amax(labels)
    props['shape'] = labels.shape
    mo = Image('{}.h5/{}'.format(blockstem, ids_label), **props)
    mo.create()
    mo.write(labels)
    mo.ds.attrs.create('maxlabel', maxlabel, dtype='uint32')
    mo.close()

    with open('{}.pickle'.format(blockstem), 'wb') as f:
        pickle.dump([axes_out, polys, block], f)

    im.close()


def stardist_mergeblocks(blockdir, image_in_ref, ids_label='labels', postfix='_stardist', dtype='int32'):

    blockpaths = glob(os.path.join(blockdir, '*.pickle'))
    blockpaths.sort()

    blockstem = blockpaths[0].split('_block')[0]
    dataset = os.path.split(blockstem)[1]

    maxlabelfile = os.path.join(blockdir, '{}_maxlabels.txt'.format(dataset))
    maxlabels = np.loadtxt(maxlabelfile, dtype=np.uint32)

    datadir = os.path.split(image_in_ref)[0]
    filestem = os.path.join(datadir, dataset)

    im = Image(image_in_ref, permission='r')
    im.load(load_data=False)
    props = im.squeeze_props(im.squeeze_props(dim=4), dim=3)
    props['dtype'] = dtype
    im.close()

    outputstem = os.path.join(datadir, '{}{}'.format(filestem, postfix))
    outpath = '{}.h5/{}'.format(outputstem, ids_label)
    mo = Image(outpath, **props)
    mo.create()

    polys_all = {}
    for blockpath in blockpaths:
        # read polys
        with open(blockpath, 'rb') as f:
            (axes_out, polys, block) = pickle.load(f)
        for k,v in polys.items():
            polys_all.setdefault(k,[]).append(v)
        # read labels
        blockstem = os.path.splitext(blockpath)[0]
        blockpath = os.path.join(datadir, '{}.h5/{}'.format(blockstem, ids_label))
        im = Image(blockpath, permission='r')
        im.load()
        labels = im.slice_dataset().astype(dtype)
        im.close()
        # relabel labels
        maxlabel = np.sum(maxlabels[:block.id])
        bg_label = 0
        mask = labels == bg_label
        labels[~mask] += maxlabel
        # write labels
        block.write(mo.ds, labels, axes=axes_out)

    mo.ds.attrs.create('maxlabel', maxlabel, dtype='uint32')
    mo.close()

    polys_all = {k: (np.concatenate(v) if k in OBJECT_KEYS else v[0])
                 for k,v in polys_all.items()}
    outpath = '{}.pickle'.format(outputstem)
    with open(outpath, 'wb') as f:
        pickle.dump(polys_all, f)
