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

import yaml

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
    get_ims_ref_path,
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
    step_id='splitblocks',
    outputdir='',
    n_workers=0,
    blocksize=[],
    blockmargin=[],
    blocks=[],
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
    outputtemplate='',
    ):
    """Average membrane and nuclear channels and write as blocks."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir, 'blocks', 'blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    blocksize, blockmargin, blocks = get_blockinfo(image_in, parameter_file, params)

    arglist = [
        (
            image_in,
            blocksize,
            blockmargin,
            [b_idx, b_idx + 1],
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
            params['outputtemplate'],
            step_id,
            outputdir,
        )
        for b_idx in blocks]

    n_workers = get_n_workers(len(blocks), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(split_with_combinechannels, arglist)


def split_with_combinechannels(
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
    outputtemplate='',
    step_id='splitblocks',
    outputdir='',
    ):
    """Average membrane and nuclear channels and write as blocks."""

    # Prepare the output.
    postfix = ''

    outputdir = get_outputdir(image_in, '', outputdir, 'blocks', 'blocks')

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

    outputtemplate = outputtemplate or '{}_{}.h5/ods'.format(outputstem, '{}')
    mpi.set_blocks(im, blocksize, blockmargin, blockrange, outputtemplate)
    mpi.scatter_series()

    props = prep_props(im, datatype, chunksize, blockmargin)
    for i in mpi.series:
        block = mpi.blocks[i]
        print('Processing blocknr {:4d} with id: {}'.format(i, block['id']))
        if len(im.dims) == 3:
            data = im.slice_dataset().astype('float')
            if bf is not None:
                bias = get_bias_field_block(bf, im.slices, data.shape, bias_dsfacs)
                bias = np.reshape(bias, data.shape)
                data /= bias
                data = np.nan_to_num(data, copy=False)
            outputpostfix = ".h5/{}".format(output['ods'])
            outpath = block['path'].replace(".h5/ods", outputpostfix)
            write_output(outpath, output['data'].astype(output['dtype']), props)
        else:
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

    ch_idxs = [memb_idxs, nucl_idxs, mean_idxs]
    ch_weights = [memb_weights, nucl_weights, mean_weights]
    ch_out = ['memb/mean', 'nucl/mean', 'mean']
    ch_ids = ['memb', 'nucl', 'mean']

    if output_channels is not None:

        if output_channels == [-1]:
            output_channels = channel_list

        ids = ["ch{:02d}".format(ch) for ch in output_channels]
        ch_idxs += [[ch] for ch in output_channels]
        ch_weights += [[1.0]] * len(output_channels)
        ch_out += ['chan/{}'.format(ch_id) for ch_id in ids]
        ch_ids += ids

    outputs = {}
    for key, idxs, weights, ods in zip(ch_ids, ch_idxs, ch_weights, ch_out):

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


def splitblocks(image_in, blocksize, blockmargin, blockrange, outputtemplate):
    """Split an image into blocks."""

    mpi = wmeMPI(usempi=False)
    im = Image(image_in, permission='r')
    im.load(load_data=False)

    mpi.set_blocks(im, blocksize, blockmargin, blockrange, outputtemplate)
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
    step_id='mergeblocks',
    outputdir='',
    n_workers=0,
    idss_select=[],
    blocksize=[],
    blockmargin=[],
    blockrange=[],
    blocks=[],
    fullsize=[],
    ims_ref_path='',
    datatype='',
    ipf='',
    elsize=[],
    inlayout='',
    squeeze='',
    is_labelimage=False,
    ):
    """Average membrane and nuclear channels and write as blocks."""

    blockdir = get_outputdir(image_in, parameter_file, '', step_id, 'blocks')
    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, '')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    block_postfix = '.h5'
    if params['ipf']:
        block_postfix = '{}{}'.format(params['ipf'], block_postfix)
    blockfiles, blocks = get_blockfiles(image_in, blockdir, params['blocks'], block_postfix)
    blocksize, blockmargin, _ = get_blockinfo(image_in, parameter_file, params)

    paths = get_paths(image_in)
    props = get_imageprops(image_in)
    dataset = os.path.splitext(paths['fname'])[0]
    ims_ref_path = get_ims_ref_path(image_in, parameter_file, params['ims_ref_path'])

    idss_dicts = [v for k, v in sorted(params.items()) if k.startswith('ids0')]  # TODO: better regex matching
    idss_select = params['idss_select'] or list(range(len(idss_dicts)))
    idss_dicts = [ids for i, ids in enumerate(idss_dicts) if i in idss_select]

    idss, outputnames, ulabelpaths = [], [], []
    for d in idss_dicts:
        outname = '{}_{}'.format(dataset, d['ids'].replace('/', '-'))
        ulabelpath = ''
        if 'is_labelimage' in d.keys():
            if d['is_labelimage']:
                ulabelpath = '{}_ulabels.npy'.format(outname)
        if d['format'] == 'h5':
            outname = '{}{}/{}'.format(outname, block_postfix, d['ids'])
        elif d['format'] == 'ims':
            outname = '{}.ims'.format(outname)
        idss.append(d['ids'])
        outputnames.append(outname)
        ulabelpaths.append(ulabelpath)

    arglist = [
        (
            ['{}/{}'.format(blockfile, ids) for blockfile in blockfiles],
            blocksize[:3],
            blockmargin[:3],
            [],
            props['shape'][:3],
            ims_ref_path,
            params['datatype'],
            params['elsize'],
            params['inlayout'],
            params['squeeze'],
            ulabelpath,
            os.path.join(outputdir, outputname),
        )
        for ids, outputname, ulabelpaths in zip(idss, outputnames, ulabelpaths)]

    n_workers = get_n_workers(len(idss), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(mergeblocks, arglist)


def mergeblocks(
        images_in,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        fullsize=[],
        ims_ref_path='',
        datatype='',
        elsize=[],
        inlayout='',
        squeeze='',
        ulabelpath='',
        outputpath='',
        ):
    """Merge blocks of data into a single hdf5 file."""

    if blockrange:
        images_in = images_in[blockrange[0]:blockrange[1]]

    mpi = wmeMPI(usempi=False)

    im = Image(images_in[0], permission='r')
    im.load(mpi.comm, load_data=False)
    props = im.get_props()
    ndim = im.get_ndim()

    inlayout = inlayout or props['axlab']
    props['axlab'] = inlayout
    props['elsize'] = elsize or props['elsize']

    dims = fullsize.copy()
    if ndim == 4:
        c_idx = props['axlab'].index('c')
        dims.insert(c_idx, im.ds.shape[c_idx])
    props['shape'] = dims

    for ax in squeeze:
        props = im.squeeze_props(props, dim=props['axlab'].index(ax))

    props['dtype'] = datatype or props['dtype']
    props['chunks'] = props['chunks'] or None

    if outputpath.endswith('.ims'):
        shutil.copy2(ims_ref_path, outputpath)
        mo = Image(outputpath)
        mo.load()
    else:
        mo = LabelImage(outputpath, **props)
        mo.create(comm=mpi.comm)

    mpi.blocks = [{'path': image_in} for image_in in images_in]
    mpi.nblocks = len(images_in)
    mpi.scatter_series()

    # merge the datasets
    ulabels = set([])  # TODO: handle for parallel MPI
    for i in mpi.series:
        block = mpi.blocks[i]

        im = Image(block['path'], permission='r')
        im.load(mpi.comm, load_data=False)
        set_slices_in_and_out(im, mo, blocksize, blockmargin, fullsize, inlayout)
        data = im.slice_dataset()
        if ulabelpath:
            ulabels |= set(np.unique(data))
        mo.write(data)
        im.close()

        print('processed block {:03d}: {}'.format(i, block['path']))

    if ulabelpath:
        np.save(ulabelpath, np.array(ulabels))

    im.close()
    mo.close()


def set_slices_in_and_out(im, mo, blocksize, margin, fullsize, axlab='zyx'):

    comps = im.split_path()
    _, x, X, y, Y, z, Z = split_filename(comps['file'])
    (oz, oZ), (iz, iZ) = margins(z, Z, blocksize[0], margin[0], fullsize[0])
    (oy, oY), (iy, iY) = margins(y, Y, blocksize[1], margin[1], fullsize[1])
    (ox, oX), (ix, iX) = margins(x, X, blocksize[2], margin[2], fullsize[2])
    im.slices[axlab.index('z')] = slice(iz, iZ)
    im.slices[axlab.index('y')] = slice(iy, iY)
    im.slices[axlab.index('x')] = slice(ix, iX)
    mo.slices[mo.axlab.index('z')] = slice(oz, oZ)
    mo.slices[mo.axlab.index('y')] = slice(oy, oY)
    mo.slices[mo.axlab.index('x')] = slice(ox, oX)


def margins(fc, fC, blocksize, margin, fullsize):
    """Return lower coordinate (fullstack and block) corrected for margin."""

    if fc == 0:
        bc = 0
    else:
        bc = 0 + margin
        fc += margin

    if fC == fullsize:
        bC = bc + blocksize + (fullsize % blocksize)

    else:
        bC = bc + blocksize
        fC -= margin

    return (fc, fC), (bc, bC)


def link_blocks(filepath_in, filepath_out, dset_in, dset_out, delete=True, links=True, is_unet=False):

    def delete_dataset(filepath, dset):
        try:
            im = Image('{}/{}'.format(filepath, dset), permission='r+')
            im.load()
            del im.file[dset_out]
        except OSError:
            pass
        except KeyError:
            pass
        im.close()

    if delete:
        mode = 'w'
    else:
        mode = 'r+'

    if links:
        import h5py
        f = h5py.File(filepath_out, 'a')
        if filepath_in == filepath_out:
            f[dset_out] = f[dset_in]
        else:
            f[dset_out] = h5py.ExternalLink(filepath_in, dset_in)

    else:

        im = Image('{}/{}'.format(filepath_in, dset_in), permission='r')
        im.load(load_data=False)

        props = im.get_props()
        if is_unet:
            props['axlab'] = 'zyx'
            props['shape'] = props['shape'][1:]
            props['slices'] = props['slices'][1:]
            props['chunks'] = props['chunks'][1:]

        data = im.slice_dataset(squeeze=True)

        im.close()

        mo = Image('{}/{}'.format(filepath_out, dset_out), permission=mode, **props)
        mo.create()
        mo.write(data)
        mo.close()


if __name__ == "__main__":
    main(sys.argv[1:])
