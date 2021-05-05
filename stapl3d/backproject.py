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

# import yaml

# from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
from random import shuffle

# from tifffile import create_output
# from xml.etree import cElementTree as etree

# from skimage import img_as_float
# from skimage.io import imread, imsave

# import czifile

from stapl3d import parse_args, Stapl3r, Image, LabelImage, wmeMPI, transpose_props

#from stapl3d.segmentation.segment import gen_outpath

logger = logging.getLogger(__name__)


def main(argv):
    """Write segment backprojection."""

    steps = ['backproject']
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


class Backproject3r(Stapl3r):
    """Write segment backprojection."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Backproject3r, self).__init__(
            image_in, parameter_file,
            module_id='backproject',
            **kwargs,
            )

        self._fun_selector = {
            'backproject': self.backproject,
            }

        self._parallelization = {
            'backproject': ['features'],
            }

        self._parameter_sets = {
            'backproject': {
                'fpar': self._FPAR_NAMES,
                'ppar': (''),
                'spar': ('_n_workers', 'features'),
                },
            }

        self._parameter_table = {
            }

        default_attr = {
            'label_image': '',
            'feature_path': '',
            'features': {},
            'labelkey': 'label',
            'name': '',
            'maxlabel': 0,
            'normalize': False,
            'scale_dtype': False,
            'replace_nan': True,
            'channel': -1,
            'blocksize': [64, 1280, 1280],
            'ims_ref': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

    def _init_paths(self):

        stem = self._build_path()
        apat = self._build_path(suffixes=[{'a': 'p'}])

        self._paths = {
            'backproject': {
                'inputs': {
                    'label_image': self.label_image,
                    'feature_path': self.feature_path,
                    'ims_ref': self.ims_ref,
                    },
                'outputs': {
                    **{ods: f'{apat}.ims' for ods in self.features},
                    #**{ods: f'{apat}.h5/{ods}' for ods in self.features},
                    # ims... feats as datasets ... etc
                    },
                },
        }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def backproject(self, **kwargs):
        """Write segment backprojection."""

        arglist = self._prep_step('backproject', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._backproject_feature, arglist)

    def _backproject_feature(self, key):
        """Backproject feature values to segments."""

        inputs = self._prep_paths(self.inputs)
        image_in = inputs['label_image']
        feat_path = inputs['feature_path']
        ims_ref = inputs['ims_ref']
        outputs = self._prep_paths(self.outputs, reps={'a': key})
        outpath = outputs[key]

        labels = LabelImage(image_in)
        labels.load(load_data=False)

        mpi = wmeMPI(usempi=False)
        mpi.set_blocks(labels, self.blocksize or labels.dims)
        mpi.scatter_series()

        if not self.maxlabel:  # TODO: read maxlabel from attribute (see zipping)
            labels.set_maxlabel()
            self.maxlabel = labels.maxlabel

        if feat_path.endswith('.csv'):
            df = pd.read_csv(feat_path)
            df = df.astype({self.labelkey: int})
        elif feat_path.endswith('.h5ad'):
            import scanpy as sc
            adata = sc.read(feat_path)
            df = adata.obs[self.labelkey].astype(int)
            df = pd.concat([df, adata[:, key].to_df()], axis=1)

        fw = np.zeros(self.maxlabel + 1, dtype='float')
        for index, row in df.iterrows():
            fw[int(row[self.labelkey])] = row[key]

        maxval = 65535  # uint16 for imaris. FIXME: generalize to other datatypes
        default = {
            'normalize': self.normalize,
            'scale_dtype': self.scale_dtype,
            'replace_nan': self.replace_nan,
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

        fw = list(fw)
        for i in mpi.series:

            print('processing {}: block {:03d} with coords: {}'.format(key, i, mpi.blocks[i]['id']))
            block = mpi.blocks[i]
            labels.slices = block['slices']
            data = labels.slice_dataset()
            out = labels.forward_map(fw, ds=data)

            if outpath.endswith('.ims'):
                mo.slices[:3] = block['slices']
                mo.write(out.astype(mo.dtype))

            elif outpath.endswith('.nii.gz'):
                props = labels.get_props()
                if not labels.path.endswith('.nii.gz'):
                    props = transpose_props(props, outlayout='xyz')
                    out = out.transpose()
                mo = write_output(outpath, out, props)

            else:
                #TODO
                #outpath = outpath or gen_outpath(labels, key)
                mo = write_output(outpath, out, labels.get_props())

        mo.close()

        return mo


def scale_fwmap(fw, maxval=65535, replace_nan=False, normalize=False, scale_dtype=False):

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
