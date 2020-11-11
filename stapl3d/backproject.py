#!/usr/bin/env python

"""Write segment backprojection.

"""

import sys
import argparse

import os
import numpy as np
import pandas as pd
from random import shuffle

from stapl3d import (
    Image,
    LabelImage,
    wmeMPI,
    transpose_props,
    )
from stapl3d.segmentation.segment import gen_outpath


def main(argv):
    """Write segment backprojection."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        'seg_path',
        help='path to segments file',
        )
    parser.add_argument(
        'csv_path',
        help='path to feature file',
        )
    parser.add_argument(
        '--labelkey',
        default='Id',
        help='the header of the label value column in the csv file',
        )
    parser.add_argument(
        '--key',
        default='pseudotime',
        help='the column in the csv file',
        )
    parser.add_argument(
        '--name',
        default='',
        help='a name for new Imaris channel',
        )
    parser.add_argument(
        '--maxlabel',
        type=int,
        default=0,
        help='the maximum label in the dataset',
        )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='apply min-max scaling'
        )
    parser.add_argument(
        '--scale_uint16',
        action='store_true',
        help='multiply by 65535'
        )
    parser.add_argument(
        '--replace_nan',
        action='store_true',
        help='run forward map through np.nan_to_num'
        )
    parser.add_argument(
        '--channel',
        type=int,
        default=-1,
        help='imaris channel to write the key to [-1 appends a new channel]',
        )
    parser.add_argument(
        '--blocksize',
        nargs=3,
        type=int,
        default=[],
        help='imaris channel to write the key to [-1 appends a new channel]',
        )
    parser.add_argument(
        '--outpath',
        help='path to output file',
        )

    args = parser.parse_args()

    csv_to_im(
        args.seg_path,
        args.csv_path,
        args.labelkey,
        args.key,
        args.name,
        args.maxlabel,
        args.normalize,
        args.scale_uint16,
        args.replace_nan,
        args.channel,
        args.blocksize,
        args.outpath,
        )


def csv_to_im(
    image_in,
    csv_path,
    labelkey='label',
    key='dapi',
    name='',
    maxlabel=0,
    normalize=False,
    scale_uint16=False,
    replace_nan=False,
    channel=-1,
    blocksize=[],
    outpath='',
    ):
    """Write segment backprojection."""

    if isinstance(image_in, Image):
        labels = image_in
    else:
        labels = LabelImage(image_in)
        labels.load(load_data=False)

    mpi = wmeMPI(usempi=False)
    mpi.set_blocks(labels, blocksize or labels.dims)
    mpi.scatter_series()

    if not maxlabel:
        labels.set_maxlabel()
        maxlabel = labels.maxlabel

    if csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path)
        df = df.astype({labelkey: int})
    elif csv_path.endswith('.h5ad'):
        import scanpy as sc
        adata = sc.read(csv_path)
        if not csv_path.endswith('_nofilter.h5ad'):
            adata.X = adata.raw.X
        df = adata.obs[labelkey].astype(int)
        df = pd.concat([df, adata[:, key].to_df()], axis=1)

    # for key in keys:  # TODO
    fw = np.zeros(maxlabel + 1, dtype='float')
    for index, row in df.iterrows():
        fw[int(row[labelkey])] = row[key]

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
        fw_n *= 65535
        fw = fw_n
    elif scale_uint16:  # for e.g. pseudotime / FA / etc / any [0, 1] vars
        fw *= 65535

    if outpath.endswith('.ims'):
        mo = Image(outpath, permission='r+')
        mo.load(load_data=False)
        if channel >= 0 and channel < mo.dims[3]:
            ch = channel
        else:
            mo.create()
            ch = mo.dims[3] - 1
        mo.slices[3] = slice(ch, ch + 1, 1)
        cpath = 'DataSetInfo/Channel {}'.format(ch)
        name = name or key
        mo.file[cpath].attrs['Name'] = np.array([c for c in name], dtype='|S1')

    fw = list(fw)
    for i in mpi.series:
        print('processing block {:03d} with id: {}'.format(i, mpi.blocks[i]['id']))
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
            outpath = outpath or gen_outpath(labels, key)
            mo = write_output(outpath, out, labels.get_props())

    mo.close()

    return mo


def ulabelset(
    image_in,
    blocksize=[],
    filepath='',
    ):
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
