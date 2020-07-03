#!/usr/bin/env python

"""Average membrane and nuclear channels and write as blocks.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import numpy as np

from skimage.transform import resize
from skimage.segmentation import relabel_sequential

from stapl3d import (
    get_outputdir,
    get_params,
    get_blockinfo,
    get_n_blocks,
    get_n_workers,
    get_paths,
    prep_outputdir,
    Image,
    wmeMPI,
    get_blockfiles,
    get_imageprops,
    LabelImage,
    split_filename,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Correct z-stack shading."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-i', '--image_in',
        required=True,
        help='path to image file',
        )
    parser.add_argument(
        '-p', '--parameter_file',
        required=True,
        help='path to yaml parameter file',
        )
    parser.add_argument(
        '-o', '--outputdir',
        required=False,
        help='path to output directory',
        )

    args = parser.parse_args()

    split(args.image_in, args.parameter_file, args.outputdir)


def split(
    image_in,
    parameter_file='',
    outputdir='',
    blocksize=[],
    blockmargin=[],
    blockrange=[],
    bias_image='',
    bias_dsfacs=[1, 64, 64, 1],
    memb_idxs=None,
    memb_weights=[],
    nucl_idxs=None,
    nucl_weights=[],
    mean_idxs=None,
    mean_weights=[],
    output_channels=None,
    datatype='',
    chunksize=[],
    ):
    """Average membrane and nuclear channels and write as blocks."""

    step_id = 'blocks'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, fallback=step_id)

    params = get_params(locals(), parameter_file, step_id)

    blocksize, blockmargin, blocks = get_blockinfo(image_in, par_file, params)

    n_workers = get_n_workers(len(blocks), params)

    arglist = [
        (
            image_in,
            blocksize,
            blockmargin,
            [b_idx, b_idx+1],
            params['bias_image'],
            params['bias_dsfacs'],
            params['memb_idxs'],
            params['memb_weights'],
            params['nucl_idxs'],
            params['nucl_weights'],
            params['mean_idxs'],
            params['mean_weights'],
            params['output_channels'],
            params['datatype'],
            params['chunksize'],
            outputdir,
        )
        for b_idx in blocks]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(combine_channels, arglist)


def combine_channels(
    image_in,
    blocksize,
    blockmargin=[0, 64, 64],
    blockrange=[],
    bias_image='',
    bias_dsfacs=[1, 64, 64, 1],
    memb_idxs=None,
    memb_weights=[],
    nucl_idxs=None,
    nucl_weights=[],
    mean_idxs=None,
    mean_weights=[],
    output_channels=None,
    datatype='',
    chunksize=[],
    outputdir='',
    ):
    """Average membrane and nuclear channels and write as blocks."""

    # Prepare the output.
    step_id = 'blocks'
    postfix = ''

    outputdir = prep_outputdir(outputdir, image_in, subdir=step_id)

    paths = get_paths(image_in)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)

    filestem = '{}'.format(dataset)
    outputstem = os.path.join(outputdir, filestem)
    outputpat = '{}.h5/{}'.format(outputstem, '{}')

    logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)
    report = {'parameters': locals()}

    mpi = wmeMPI(usempi=False)

    im = Image(image_in, permission='r')
    im.load(comm=mpi.comm, load_data=False)

    if bias_image:
        bf = Image(bias_image, permission='r')
        bf.load(comm=mpi.comm, load_data=False)
    else:
        bf = None

    outputtemplate = '{}_{}.h5/ods'.format(outputstem, '{}')
    mpi.set_blocks(im, blocksize, blockmargin, blockrange, outputtemplate)
    mpi.scatter_series()

    props = prep_props(im, datatype, chunksize, blockmargin)
    for i in mpi.series:
        block = mpi.blocks[i]
        print('Processing blocknr {:4d} with id: {}'.format(i, block['id']))
        add_volumes(im, block, props, bf, bias_dsfacs,
                    memb_idxs, memb_weights,
                    nucl_idxs, nucl_weights,
                    mean_idxs, mean_weights,
                    output_channels,
                    )

    im.close()


def add_volumes(im, block, props, bf=None, bias_dsfacs=[1, 64, 64, 1],
                memb_idxs=None, memb_weights=[],
                nucl_idxs=None, nucl_weights=[],
                mean_idxs=None, mean_weights=[],
                output_channels=None,
                ):
    """Read, channel-average and write blocks of of data."""

    c_axis = im.axlab.index('c')
    channel_list = [i for i in range(im.dims[c_axis])]
    size = get_outsize(im, block['slices'])

    if output_channels is not None:

        if output_channels == [-1]:
            output_channels = channel_list

        ch_ids = ["ch{:02d}".format(ch) for ch in output_channels]
        ch_idxs = [[ch] for ch in output_channels]
        ch_weights = [[1.0]] * len(ch_idxs)
        ch_out = ['chan/{}'.format(ch_id) for ch_id in ch_ids]

    zipped = zip(
        ['memb', 'nucl', 'mean'] + ch_ids,
        [memb_idxs, nucl_idxs, mean_idxs] + ch_idxs,
        [memb_weights, nucl_weights, mean_weights] + ch_weights,
        ['memb/mean', 'nucl/mean', 'mean'] + ch_out,
        )

    outputs = {}
    for key, idxs, weights, ods in zipped:

        if idxs is None:
            continue
        elif idxs == [-1]:
            idxs = [i for i in range(im.dims[c_axis])]

        if weights == [-1]:
            weights = [1] * len(idxs)

        outputs[key] = {
            'idxs': idxs,
            'weights': weights,
            'ods': ods,
            'dtype': props['dtype'],
            'data': np.zeros(size, dtype='float'),
            }

    im.slices = block['slices']

    idxs_set = set([l for k, v in outputs.items() for l in v['idxs']])
    for volnr in idxs_set:

        im.slices[c_axis] = slice(volnr, volnr + 1, 1)
        data = im.slice_dataset().astype('float')

        if bf is not None:
            bias = get_bias_field_block(bf, im.slices, data.shape, bias_dsfacs)
            bias = np.reshape(bias, data.shape)
            data /= bias
            data = np.nan_to_num(data, copy=False)

        for name, output in outputs.items():
            if volnr in output['idxs']:
                idx = output['idxs'].index(volnr)
                data *= output['weights'][idx]
                output['data'] += data

    for name, output in outputs.items():
        output['data'] = output['data'] / len(output['idxs'])
        outputpostfix = ".h5/{}".format(output['ods'])
        outpath = block['path'].replace(".h5/ods", outputpostfix)
        write_output(outpath, output['data'].astype(output['dtype']), props)


def write_output(outpath, data, props):
    """Write data to file."""

    props['shape'] = data.shape
    props['dtype'] = data.dtype

    mo = Image(outpath, **props)
    mo.create()
    mo.write(data=data)
    mo.close()


def get_bias_field_block(bf, slices, outdims, dsfacs=[1, 64, 64, 1]):
    """Retrieve and upsample the biasfield for a datablock."""

    bf.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                 for slc, ds in zip(slices, dsfacs)]
    bf_block = bf.slice_dataset().astype('float32')
    bias = resize(bf_block, outdims, preserve_range=True)

    return bias


def squeeze_slices(slices, axis):

    slcs = list(slices)
    del slcs[axis]

    return slcs


def get_outsize(im, slices):
    """Return the dimensions of the image."""

    slcs = list(slices)
    if 't' in im.axlab:
        slcs = squeeze_slices(slices, im.axlab.index('t'))
    if 'c' in im.axlab:
        slcs = squeeze_slices(slcs, im.axlab.index('c'))
    size = list(im.slices2shape(slcs))

    return size


def prep_props(im, datatype, chunksize, blockmargin):
    """Create a block-specific props-dictionary."""

    props = im.get_props()

    props['dtype'] = datatype or props['dtype']

    chunks = list(im.dims)
    for d in 'xy':
        axis = im.axlab.index(d)
        chunks[axis] = blockmargin[axis]
    props['chunks'] = chunksize or chunks

    if len(props['shape']) == 5:
        props = im.squeeze_props(props=props, dim=4)
    if len(props['shape']) == 4:
        props = im.squeeze_props(props=props, dim=3)

    return props


def splitblocks(image_in, blocksize, blockmargin, outputtemplate):
    """Split an image into blocks."""

    mpi = wmeMPI(usempi=False)
    im = Image(image_in, permission='r')
    im.load(load_data=False)

    mpi.set_blocks(im, blocksize, blockmargin, [], outputtemplate)
    mpi.scatter_series()
    props = im.get_props()
    for i in mpi.series:
        block = mpi.blocks[i]
        im.slices = block['slices']
        data = im.slice_dataset()
        write_output(block['path'], data, props)


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
            blocksize[:3],
            blockmargin[:3],
            [],
            [0, 0, 0],
            props['shape'][:3],
            '',
            os.path.join(outputdir, '{}_{}.h5/{}'.format(dataset, ids.replace('/', '-'), ids)),
        )
        for ids in idss]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(mergeblocks, arglist)


def mergeblocks(
        images_in,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        blockoffset=[0, 0, 0],
        fullsize=[],
        datatype='',
        outputpath='',
        ):
    """Merge blocks of data into a single hdf5 file."""

    if blockrange:
        images_in = images_in[blockrange[0]:blockrange[1]]

    mpi = wmeMPI(usempi=False)

    im = Image(images_in[0], permission='r')
    im.load(mpi.comm, load_data=False)
    props = im.get_props(squeeze=True)
    ndim = im.get_ndim()

    props['dtype'] = datatype or props['dtype']
    props['chunks'] = props['chunks'] or None

    # get the size of the outputfile
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
    for i in mpi.series:
        block = mpi.blocks[i]

        im = Image(block['path'], permission='r')
        im.load(mpi.comm, load_data=False)
        set_slices_in_and_out(im, mo, blocksize, blockmargin, fullsize)
        data = im.slice_dataset()
        mo.write(data)
        im.close()

        print('processed block {:03d}: {}'.format(i, block['path']))

    im.close()
    mo.close()


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


if __name__ == "__main__":
    main(sys.argv[1:])
