#!/usr/bin/env python

"""Write segment backprojection.

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
import pandas as pd
from random import shuffle

import h5py
import pathlib

from stapl3d import parse_args, Stapl3r, Image, LabelImage, wmeMPI, transpose_props
from stapl3d.blocks import Block3r

logger = logging.getLogger(__name__)


def main(argv):
    """Write segment backprojection."""

    steps = ['backproject', 'postprocess']
    args = parse_args('backproject', steps, *argv)

    backproject3r = Backproject3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        backproject3r._fun_selector[step]()


class Backproject3r(Block3r):
    """Write segment backprojection."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'backproject'

        super(Backproject3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'backproject': self.backproject,
            'postprocess': self.postprocess,
            })

        self._parallelization.update({
#            'backproject': ['features'],
            'backproject': ['blocks'],
            'postprocess': [],
            })

        self._parameter_sets.update({
            'backproject': {
                'fpar': self._FPAR_NAMES,
                'ppar': (''),
#                'spar': ('_n_workers', 'features'),
                'spar': ('_n_workers', 'blocks'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'features': {},
            'labelkey': 'label',
            'name': '',
            'maxlabel': 0,
            'normalize': False,
            'scale_dtype': False,
            'replace_nan': True,
            'dtype': 'uint16',
            'channel': -1,
            'blocksize_tmp': [],
            'ims_ref': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths_backproject()

        self._init_log()

        self._prep_blocks()

        self._images = []
        self._labels = []


    def _init_paths_backproject(self):

        lbl_path = self._build_basename(
            prefixes=[self.prefix],
            suffixes=['segm-labels_full'],
            ext='h5/segm/labels_full',
            )
        ftr_path = self._build_basename(
            prefixes=[self.prefix],
            suffixes=['features'],
            ext='csv',
            )
        ims_path = self._build_basename(
            prefixes=[self.prefix],
            suffixes=['shading_stitching_ref'],
            ext='ims',
            )

        apat = self._build_basename(suffixes=['{a}'], ext='h5/{a}')
        #apat = self._build_basename(suffixes=['{a}'], ext='ims')

        self._paths.update({
            'backproject': {
                'inputs': {
                    'label_image': lbl_path,
                    'feature_csv': ftr_path,
                    'ims_ref': '',
                    },
                'outputs': {
                    'backproject': f'{apat}',
                    },
                },
            'postprocess': {
                'inputs': {
                    'filepat': self._build_basename(suffixes=['{a}'], ext='h5'),
                    },
                'outputs': {
                    'aggregate': self._build_basename(ext='h5'),
#                    'aggregate': self._build_path(prefixes=['backproject'], ext='h5'),
                    },
                },
        })

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def backproject(self, **kwargs):
        """Write segment backprojection."""

        arglist = self._prep_step('backproject', kwargs)
        for feat in self.features.keys():
            print(feat)
            arglist_aug = [(feat, block_idx[0]) for block_idx in arglist]
            with multiprocessing.Pool(processes=self._n_workers) as pool:
                pool.starmap(self._backproject_feature, arglist_aug)

        """
        arglist = self._prep_step('backproject', kwargs)
        b = self.blocks or list(range(len(self._blocks)))
        for block_idx in b:
            arglist_aug = [(feat[0], block_idx) for feat in arglist]
            with multiprocessing.Pool(processes=self._n_workers) as pool:
                pool.starmap(self._backproject_feature, arglist_aug)
        """

    def _backproject_feature(self, key, block_idx):
        """Backproject feature values to segments."""

        block = self._blocks[block_idx]

        inputs = self._prep_paths_blockfiles(self.inputs, block)
        outputs = self._prep_paths_blockfiles(self.outputs, block, reps={'a': key})

        image_in = inputs['label_image']
        feat_path = inputs['feature_path']
        ims_ref = inputs['ims_ref']
        outpath = outputs['backproject']

        labels = LabelImage(image_in, permission='r')
        labels.load(load_data=False)
        props = labels.get_props()

        maxlabel = self.prep_maxlabel(labels, maxlabel=self.maxlabel)
        labels.close()

        mpi = wmeMPI(usempi=False)
        bs = self.blocksize_tmp or labels.dims
        mpi.set_blocks(labels, bs)
        mpi.scatter_series()

        if feat_path.endswith('.csv'):
            df = pd.read_csv(feat_path)
            df = df.astype({self.labelkey: int})
        elif feat_path.endswith('.h5ad'):
            import scanpy as sc
            adata = sc.read(feat_path)

            adata.X = adata.raw.X

            cols = [self.labelkey, 'block']
            df = adata.obs[cols].astype(int)

            if key in adata.obs.columns:
                df2 = adata.obs[key]
            else:
                df2 = adata[:, key].to_df()

            df = pd.concat([df, df2], axis=1)
            df = df[df['block'] == block_idx]

        print(feat_path, maxlabel, block_idx)

        fw = np.zeros(maxlabel + 1, dtype='float')
        for index, row in df.iterrows():
            fw[int(row[self.labelkey])] = row[key]

        maxval = 65535  # uint16 for imaris. FIXME: generalize to other datatypes
        default = {
            'normalize': self.normalize,
            'scale_dtype': self.scale_dtype,
            'replace_nan': self.replace_nan,
            'dtype': self.dtype,
            }
        featdict = {**default, **self.features[key]}
        fw = scale_fwmap(fw, maxval, **featdict)

        if outpath.endswith('.ims'):
            if ims_ref:  # single-channel empty imaris image with the correct dimensions
                shutil.copy2(ims_ref, outpath)
                self.channel = 0

            mo = Image(outpath, permission='r+')
            mo.load(load_data=False)

            if self.channel >= 0 and self.channel < mo.dims[3]:
                ch = self.channel
            else:
                mo.create()
                ch = mo.dims[3] - 1
            mo.slices[3] = slice(ch, ch + 1, 1)
            cpath = 'DataSetInfo/Channel {}'.format(ch)
            name = self.name or key
            mo.file[cpath].attrs['Name'] = np.array([c for c in name], dtype='|S1')

        elif '.h5' in outpath:
            props['dtype'] = featdict['dtype']
            props['permission'] = 'r+'
            mo = Image(outpath, **props)
            mo.create()
            mo.close()

        fw = list(fw)
        for i in mpi.series:

            #print('{}: processing {}: block {:03d} with coords: {}'.format(block.path, key, i, mpi.blocks[i]['id']))
            block = mpi.blocks[i]
            labels = LabelImage(image_in, permission='r+')
            labels.load(load_data=False)
            labels.slices = block['slices']
            data = labels.slice_dataset().astype('int')
            out = labels.forward_map(fw, ds=data)
            labels.close()

            if outpath.endswith('.ims'):
                mo.slices[:3] = block['slices']
                mo.write(out.astype(mo.dtype))

            elif '.h5' in outpath:
                mo = Image(outpath)
                mo.load()
                mo.slices[:3] = block['slices']
                mo.write(out.astype(mo.dtype))
                mo.close()

            elif outpath.endswith('.nii.gz'):
                # props = labels.get_props()
                # labels.close()
                if not labels.path.endswith('.nii.gz'):
                    props = transpose_props(props, outlayout='xyz')
                    out = out.transpose()
                mo = write_output(outpath, out, props)

            else:
                # props = labels.get_props()
                #TODO
                #outpath = outpath or gen_outpath(labels, key)
                mo = write_output(outpath, out, props)

        mo.close()

    def prep_maxlabel(self, labels, maxlabel=0):

        if not maxlabel:
            if 'maxlabel' in labels.ds.attrs.keys():
                maxlabel = labels.ds.attrs['maxlabel']

        try:
            maxlabel = int(maxlabel)
        except ValueError:
            if maxlabel.endswith('.npy'):
                ulabels = np.load(maxlabel, allow_pickle=True)
                maxlabel = max(np.amax(ulabels))
            else:
                maxlabels = np.loadtxt(maxlabel, dtype=np.uint32)
                maxlabel = max(maxlabels)

        if not maxlabel:
            labels.set_maxlabel()
            maxlabel = int(labels.maxlabel)

        return maxlabel

    def postprocess(self):

        #TODO: unduplicate from imarisfiles.py and blocks.py: Merger
        def create_copy(f, tgt_loc, ext_file, ext_loc):
            """Copy the imaris DatasetInfo group."""
            try:
                del f[tgt_loc]
            except KeyError:
                pass
            g = h5py.File(ext_file, 'r')
            f.copy(g[ext_loc], tgt_loc)
            g.close()

        def create_ext_link(f, tgt_loc, ext_file, ext_loc):
            """Create an individual link."""
            try:
                del f[tgt_loc]
            except KeyError:
                pass
            f[tgt_loc] = h5py.ExternalLink(ext_file, ext_loc)

        self._prep_step('postprocess', {})

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        tgt_file = outputs['aggregate']
        tgt_dir  = os.path.dirname(tgt_file)
        f = h5py.File(tgt_file, 'w')

        for ids in self.features.keys():
            inputs = self._prep_paths(self.inputs, reps={'a': ids})
            linked_path = os.path.relpath(inputs['filepat'], tgt_dir)
            ext_file = pathlib.Path(linked_path).as_posix()
            create_ext_link(f, ids, ext_file, ids)


def scale_fwmap(fw, maxval=65535, replace_nan=False, normalize=False, scale_dtype=False, dtype='uint16'):

    if replace_nan:
        fw = np.nan_to_num(fw)

    if normalize:
        def normalize_data(data):
            """Normalize data between 0 and 1."""
            data = data.astype('float64')
            datamin = np.amin(data)
            datamax = np.amax(data)
            data -= datamin
            data *= 1/(datamax-datamin)
            return data, [datamin, datamax]
        fw_n, fw_minmax = normalize_data(fw)
        fw_n *= maxval
        fw = fw_n
    elif scale_dtype:  # for e.g. pseudotime / FA / etc / any [0, 1] vars
        fw *= maxval

    return fw


def ulabelset(image_in, blocksize=[], filepath=''):
    """Find the set of unique labels."""

    if isinstance(image_in, Image):
        labels = image_in
    else:
        labels = LabelImage(image_in)
        labels.load(load_data=False)

    mpi = wmeMPI(usempi=False)
    mpi.set_blocks(labels, blocksize or labels.dims)
    mpi.scatter_series()

    ulabels = set([])
    for i in mpi.series:
        print('processing block {:03d} with id: {}'.format(i, mpi.blocks[i]['id']))
        block = mpi.blocks[i]
        labels.slices = block['slices']
        data = labels.slice_dataset()
        ulabels |= set(np.unique(data))

    if filepath:
        np.save(filepath, np.array(list(ulabels)))

    return ulabels


def shuffle_labels(ulabels, wrap=65535, filepath=''):
    """Shuffle a labelset."""

    relabeled_seq = np.array([l for l in range(1, len(ulabels) + 1)])

    relabeled_shuffled = np.copy(relabeled_seq)
    shuffle(relabeled_shuffled)

    relabeled_wrapped = relabeled_shuffled % wrap

    if filepath:
        keys = ['label', 'relabel_seq', 'relabel_shuffled', 'relabel_wrapped']
        vols = [ulabels, relabeled_seq, relabeled_shuffled, relabeled_wrapped]
        df = pd.DataFrame(np.array(vols).transpose(), columns=keys)
        df.to_csv(filepath)

    return relabeled_seq, relabeled_shuffled, relabeled_wrapped


def write_output(outpath, out, props):
    """Write data to image file."""

    props['dtype'] = out.dtype
    mo = Image(outpath, **props)
    mo.create()
    mo.write(out)

    return mo


if __name__ == "__main__":
    main(sys.argv[1:])
