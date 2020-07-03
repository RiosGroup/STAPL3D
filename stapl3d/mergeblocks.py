#!/usr/bin/env python

"""Merge blocks of data into a single hdf5 dataset.

example splitting of the full dataset: old
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}.h5 \
$datadir/datastem.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx -p datastem
TODO:
splitting of the full dataset: proposed new
# create group?
# links to subsets of full dataset?
# include coordinates in attributes?
"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import numpy as np
from skimage.segmentation import relabel_sequential

from stapl3d import wmeMPI, Image, LabelImage, split_filename
from stapl3d import (
    get_outputdir,
    get_blockfiles,
    get_blockinfo,
    get_imageprops,
    get_params,
    get_n_workers,
    get_paths,
    Image,
    LabelImage,
    wmeMPI,
    split_filename,
    )


def main(argv):
    """Merge blocks of data into a single hdf5 dataset."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_mergeblocks(parser)

    parser.add_argument(
        'inputfiles',
        nargs='*',
        help="""paths to hdf5 datasets <filepath>.h5/<...>/<dataset>:
                datasets to merge together"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                merged dataset"""
        )

    parser.add_argument(
        '-b', '--blockoffset',
        nargs=3,
        type=int,
        default=[0, 0, 0],
        help='offset of the datablock'
        )
    parser.add_argument(
        '-s', '--fullsize',
        nargs=3,
        type=int,
        default=[],
        help='the size of the full dataset'
        )

    parser.add_argument(
        '-l', '--is_labelimage',
        action='store_true',
        help='flag to indicate labelimage'
        )
    parser.add_argument(
        '-r', '--relabel',
        action='store_true',
        help='apply incremental labeling to each block'
        )
    parser.add_argument(
        '-n', '--neighbourmerge',
        action='store_true',
        help='merge overlapping labels'
        )
    parser.add_argument(
        '-F', '--save_fwmap',
        action='store_true',
        help='save the forward map (.npy)'
        )

    parser.add_argument(
        '-f', '--func',
        default='np.amax',
        help='function used for downsampling'
        )

    parser.add_argument(
        '-d', '--datatype',
        default='',
        help='the numpy-style output datatype'
        )

    parser.add_argument(
        '-D', '--dataslices',
        nargs='*',
        type=int,
        help="""
        Data slices, specified as triplets of <start> <stop> <step>;
        setting any <stop> to 0 or will select the full extent;
        provide triplets in the order of the input dataset.
        """
        )

    parser.add_argument(
        '-M', '--usempi',
        action='store_true',
        help='use mpi4py'
        )

    parser.add_argument(
        '-S', '--save_steps',
        action='store_true',
        help='save intermediate results'
        )

    parser.add_argument(
        '-P', '--protective',
        action='store_true',
        help='protect against overwriting data'
        )

    parser.add_argument(
        '--blocksize',
        nargs='*',
        type=int,
        default=[],
        help='size of the datablock'
        )

    parser.add_argument(
        '--blockmargin',
        nargs='*',
        type=int,
        default=[],
        help='the datablock overlap used'
        )

    parser.add_argument(
        '--blockrange',
        nargs=2,
        type=int,
        default=[],
        help='a range of blocks to process'
        )

    args = parser.parse_args()

    mergeblocks(
        args.inputfiles,
        args.dataslices,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.blockoffset,
        args.fullsize,
        args.is_labelimage,
        args.relabel,
        args.neighbourmerge,
        args.save_fwmap,
        args.blockreduce,
        args.func,
        args.datatype,
        args.usempi & ('mpi4py' in sys.modules),
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def merge(
    image_in,
    parameter_file='',
    outputdir='',
    blocksize=[],
    blockmargin=[],
    blockrange=[],
    blocks=[],
    fullsize=[],
    ):
    """Average membrane and nuclear channels and write as blocks."""

    step_id = 'mergeblocks'

    blockdir = get_outputdir(image_in, parameter_file, outputdir, 'blocks', fallback='blocks')
    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, fallback='')

    params = get_params(locals(), parameter_file, step_id)
    idss = [v['ids'] for ids, v in params.items() if ids.startswith('ids')]
    n_workers = get_n_workers(len(idss), params)

    filepaths, blocks = get_blockfiles(image_in, blockdir, params['blocks'])
    blocksize, blockmargin, _ = get_blockinfo(image_in, parameter_file, params)
    props = get_imageprops(image_in)

    dataset = os.path.splitext(get_paths(image_in)['fname'])[0]

    arglist = [
        (
            ['{}/{}'.format(filepath, ids) for filepath in filepaths],
            None,
            blocksize[:3],
            blockmargin[:3],
            [],
            [0, 0, 0],
            props['shape'][:3],
            False,
            False,
            False,
            False,
            [],
            'np.amax',
            '',
            False,
            os.path.join(outputdir, '{}_{}.h5/{}'.format(dataset, ids.replace('/', '-'), ids)),
            False,
            False,
        )
        for ids in idss]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(mergeblocks, arglist)


def mergeblocks(
        images_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        blockoffset=[0, 0, 0],
        fullsize=[],
        is_labelimage=False,
        relabel=False,
        neighbourmerge=False,
        save_fwmap=False,
        blockreduce=[],
        func='np.amax',
        datatype='',
        usempi=False,
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """Merge blocks of data into a single hdf5 file."""

    if blockrange:
        images_in = images_in[blockrange[0]:blockrange[1]]

    mpi = wmeMPI(usempi)

    im = Image(images_in[0], permission='r')
    im.load(mpi.comm, load_data=False)
    props = im.get_props(protective=protective, squeeze=True)
    ndim = im.get_ndim()

    props['dtype'] = datatype or props['dtype']
    props['chunks'] = props['chunks'] or None

    # get the size of the outputfile
    # TODO: option to derive fullsize from dset_names?
    if blockreduce:
        datasize = np.subtract(fullsize, blockoffset)
        outsize = [int(np.ceil(d/np.float(b)))
                   for d, b in zip(datasize, blockreduce)]
        props['elsize'] = [e*b for e, b in zip(im.elsize, blockreduce)]
    else:  # FIXME: 'zyx(c)' stack assumed
        outsize = np.subtract(fullsize, blockoffset)

    if ndim == 4:
        outsize = list(outsize) + [im.ds.shape[3]]  # TODO: flexible insert

    if outputpath.endswith('.ims'):
        mo = LabelImage(outputpath)
        mo.create(comm=mpi.comm)
    else:
        props['shape'] = outsize
        mo = LabelImage(outputpath, **props)
        mo.create(comm=mpi.comm)

    mpi.blocks = [{'path': image_in} for image_in in images_in]
    mpi.nblocks = len(images_in)
    mpi.scatter_series()

    # merge the datasets
    maxlabel = 0
    for i in mpi.series:

        block = mpi.blocks[i]
        # try:
        maxlabel = process_block(block['path'], ndim, blockreduce, func,
                                 blockoffset, blocksize, blockmargin,
                                 fullsize,
                                 mo,
                                 is_labelimage, relabel,
                                 neighbourmerge, save_fwmap,
                                 maxlabel, mpi)
        print('processed block {:03d}: {}'.format(i, block['path']))
        # except Exception as e:
        #     print('failed block {:03d}: {}'.format(i, block['path']))
        #     print(e)

    im.close()
    mo.close()

    return mo


def process_block(image_in, ndim, blockreduce, func,
                  blockoffset, blocksize, margin, fullsize,
                  mo,
                  is_labelimage, relabel, neighbourmerge, save_fwmap,
                  maxlabel, mpi):
    """Write a block of data into a hdf5 file."""

    # open data for reading
    im = Image(image_in, permission='r')
    im.load(mpi.comm, load_data=False)

    # get the indices into the input and output datasets
    # TODO: get indices from attributes
    # TODO: get from mpi.get_blocks
    set_slices_in_and_out(im, mo, blocksize, margin, fullsize)

    # simply copy the data from input to output
    """NOTE:
    it is assumed that the inputs are not 4D labelimages
    """
    if ndim == 4:
        mo.write(im.slice_dataset())
        im.close()
        return
    if ((not is_labelimage) or
            ((not relabel) and
             (not neighbourmerge) and
             (not blockreduce))):
        data = im.slice_dataset()
        #datatype = 'uint16'
        #from skimage.util.dtype import convert
        #data = convert(data, np.dtype(datatype), force_copy=False)
        mo.write(data)
        im.close()
        return

    # forward map to relabel the blocks in the output
    if relabel:
        # FIXME: make sure to get all data in the block
        fw, maxlabel = relabel_block(im.ds[:], maxlabel, mpi)
        if save_fwmap:
            comps = im.split_path()
            fpath = '{}_{}.npy'.format(comps['base'], comps['int'][1:])
            np.save(fpath, fw)
        if (not neighbourmerge) and (not blockreduce):
            data = im.slice_dataset()
            mo.write(fw[data])
            im.close()
            return
    else:
        ulabels = np.unique(im.ds[:])
        fw = [l for l in range(0, np.amax(ulabels) + 1)]
        fw = np.array(fw)

    # blockwise reduction of input datasets
    if blockreduce is not None:
        pass
    else:
        data = im.slice_dataset()

    # merge overlapping labels
    fw = merge_overlap(fw, im, mo, data, margin)
    mo.write(fw[data])
    im.close()


def relabel_block(ds_in, maxlabel, mpi=None):
    """Relabel the labelvolume with consecutive labels.

    NOTE:
    relabel from 0, because mpi is unaware of maxlabel before gather
    """
    fw = relabel_sequential(ds_in[:])[1]

    if mpi.enabled:
        # FIXME: only terminates properly when: nblocks % size = 0

        num_labels = np.amax(fw)
        num_labels = mpi.comm.gather(num_labels, root=0)

        if mpi.rank == 0:
            add_labels = [maxlabel + np.sum(num_labels[:i])
                          for i in range(1, mpi.size)]
            add_labels = np.array([maxlabel] + add_labels, dtype='i')
            maxlabel = maxlabel + np.sum(num_labels)
        else:
            add_labels = np.empty(mpi.size)

        add_labels = mpi.comm.bcast(add_labels, root=0)
        fw[1:] += add_labels[mpi.rank]

    else:

        fw[1:] += maxlabel
        if len(fw) > 1:
            maxlabel = np.amax(fw)

    return fw, maxlabel


def set_slices_in_and_out(im, mo, blocksize, margin, fullsize, blockoffset=[0, 0, 0]):

    comps = im.split_path()
    _, x, X, y, Y, z, Z = split_filename(comps['file'], blockoffset[:3][::-1])
    (oz, oZ), (iz, iZ) = margins(z, Z, blocksize[0], margin[0], fullsize[0])
    (oy, oY), (iy, iY) = margins(y, Y, blocksize[1], margin[1], fullsize[1])
    (ox, oX), (ix, iX) = margins(x, X, blocksize[2], margin[2], fullsize[2])
    im.slices[0] = slice(iz, iZ, 1)
    im.slices[1] = slice(iy, iY, 1)
    im.slices[2] = slice(ix, iX, 1)
    mo.slices[0] = slice(oz, oZ, 1)
    mo.slices[1] = slice(oy, oY, 1)
    mo.slices[2] = slice(ox, oX, 1)


def margins(fc, fC, blocksize, margin, fullsize):
    """Return lower coordinate (fullstack and block) corrected for margin."""

    if fc == 0:
        bc = 0
    else:
        bc = 0 + margin
        fc += margin

    if fC == fullsize:
        bC = bc + blocksize + (fullsize % blocksize)
#         bC = bc + blocksize + 8 ==>>
#         failed block 001: /Users/mkleinnijenhuis/PMCdata/Kidney/190909_RL57_FUnGI_16Bit_25x_zstack1-Masked_T001_Z001_C01/blocks_0500/190909_RL57_FUnGI_16Bit_25x_zstack1-Masked_T001_Z001_C01_00000-00564_00436-01024_00000-00150.h5/memb/sum
#         Can't broadcast (150, 508, 500) -> (150, 524, 500)
#         WHY 24??? ==>> fullsize is 1024; blocksize is 500; 3x3=9 blocks are created; blocks that fail are 1 3 4 5 7, block sthat succeeed are 0 2 6 8;

    else:
        bC = bc + blocksize
        fC -= margin

    return (fc, fC), (bc, bC)


def get_overlap(side, im, mo, data, ixyz, oxyz, margin=[0, 0, 0]):
    """Return boundary slice of block and its neighbour."""

    ix, iX, iy, iY, iz, iZ = ixyz
    ox, oX, oy, oY, oz, oZ = oxyz
    # FIXME: need to account for blockoffset

    data_section = None
    nb_section = None
    ds_in = data
    ds_out = mo.ds

    if (side == 'xmin') & (ox > 0):
        data_section = ds_in[iz:iZ, iy:iY, :margin[2]]
        nb_section = ds_out[oz:oZ, oy:oY, ox-margin[2]:ox]

#         im.slices[0] = slice(iz, iZ, 1)
#         im.slices[1] = slice(iy, iY, 1)
#         im.slices[2] = slice(0, margin[2], 1)
#
#         mo.slices[0] = slice(oz, oZ, 1)
#         mo.slices[1] = slice(oy, oY, 1)
#         mo.slices[2] = slice(ox-margin[2], ox, 1)

    elif (side == 'xmax') & (oX < ds_out.shape[2]):

        data_section = ds_in[iz:iZ, iy:iY, -margin[2]:]
        nb_section = ds_out[oz:oZ, oy:oY, oX:oX+margin[2]]

    elif (side == 'ymin') & (oy > 0):
        data_section = ds_in[iz:iZ, :margin[1], ix:iX]
        nb_section = ds_out[oz:oZ, oy-margin[1]:oy, ox:oX]

    elif (side == 'ymax') & (oY < ds_out.shape[1]):
        data_section = ds_in[iz:iZ, -margin[1]:, ix:iX]
        nb_section = ds_out[oz:oZ, oY:oY+margin[1], ox:oX]

    elif (side == 'zmin') & (oz > 0):
        data_section = ds_in[:margin[0], iy:iY, ix:iX]
        nb_section = ds_out[oz-margin[0]:oz, oy:oY, ox:oX]

    elif (side == 'zmax') & (oZ < ds_out.shape[0]):
        data_section = ds_in[-margin[0]:, iy:iY, ix:iX]
        nb_section = ds_out[oZ:oZ+margin[0], oy:oY, ox:oX]

#     data_section = im.slice_dataset()
#     nb_section = im.slice_dataset()

    return data_section, nb_section


def merge_overlap(fw, im, mo, data, margin=[0, 0, 0]):
    """Adapt the forward map to merge neighbouring labels."""

    ixyz = (im.slices[2].start,
            im.slices[2].stop,
            im.slices[1].start,
            im.slices[1].stop,
            im.slices[0].start,
            im.slices[0].stop,
            )
    oxyz = (im.slices[2].start,
            im.slices[2].stop,
            im.slices[1].start,
            im.slices[1].stop,
            im.slices[0].start,
            im.slices[0].stop,
            )

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        ds, ns = get_overlap(side, im, mo, data, ixyz, oxyz, margin)

        if ns is None:
            continue

        data_labels = np.trim_zeros(np.unique(ds))
        for data_label in data_labels:

            mask_data = ds == data_label
            bins = np.bincount(ns[mask_data])
            if len(bins) <= 1:
                continue

            nb_label = np.argmax(bins[1:]) + 1
            n_data = np.sum(mask_data)
            n_nb = bins[nb_label]
            if float(n_nb) / float(n_data) < 0.1:
                continue

            fw[data_label] = nb_label

    return fw


def get_sections(side, ds_in, ds_out, xyz):
    """Return boundary slice of block and its neighbour."""

    x, X, y, Y, z, Z = xyz
    nb_section = None

    if side == 'xmin':
        data_section = ds_in[:, :, 0]
        if x > 0:
            nb_section = ds_out[z:Z, y:Y, x-1]
    elif side == 'xmax':
        data_section = ds_in[:, :, -1]
        if X < ds_out.shape[2]:
            nb_section = ds_out[z:Z, y:Y, X]
    elif side == 'ymin':
        data_section = ds_in[:, 0, :]
        if y > 0:
            nb_section = ds_out[z:Z, y-1, x:X]
    elif side == 'ymax':
        data_section = ds_in[:, -1, :]
        if Y < ds_out.shape[1]:
            nb_section = ds_out[z:Z, Y, x:X]
    elif side == 'zmin':
        data_section = ds_in[0, :, :]
        if z > 0:
            nb_section = ds_out[z-1, y:Y, x:X]
    elif side == 'zmax':
        data_section = ds_in[-1, :, :]
        if Z < ds_out.shape[0]:
            nb_section = ds_out[Z, y:Y, x:X]

    return data_section, nb_section


def merge_neighbours(fw, ds_in, ds_out, xyz):
    """Adapt the forward map to merge neighbouring labels."""

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        data_section, nb_section = get_sections(side, ds_in, ds_out, xyz)
        if nb_section is None:
            continue

        data_labels = np.trim_zeros(np.unique(data_section))
        for data_label in data_labels:

            mask_data = data_section == data_label
            bins = np.bincount(nb_section[mask_data])
            if len(bins) <= 1:
                continue

            nb_label = np.argmax(bins[1:]) + 1
            n_data = np.sum(mask_data)
            n_nb = bins[nb_label]
            if float(n_nb) / float(n_data) < 0.1:
                continue

            fw[data_label] = nb_label
            print('%s: mapped label %d to %d' % (side, data_label, nb_label))

    return fw


if __name__ == "__main__":
    main(sys.argv)
