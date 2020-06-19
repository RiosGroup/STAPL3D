#!/usr/bin/env python

"""Average membrane and nuclear channels and write as blocks.

"""

import sys
import argparse
import yaml

import numpy as np
import multiprocessing

from skimage.transform import resize

from stapl3d import Image, wmeMPI


def main(argv):
    """Correct z-stack shading."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        'inputfile',
        help='path to ims file',
        )
    parser.add_argument(
        'parameter_file',
        help='path to yaml parameter file',
        )

    args = parser.parse_args()

    process_channels(
        args.inputfile,
        args.parameter_file,
        )


def process_channels(
        filepath,
        parameter_file='',
        blocksize=[],
        blockmargin=[0, 64, 64, 0, 0],
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
        outputprefix='',
        usempi=False,
        ):
    """Average membrane and nuclear channels and write as blocks."""

    params = locals()

    file_params = {}
    if parameter_file:
        with open(parameter_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            file_params = cfg['channels']

    params.update(file_params)

    if 'blocks' not in params.keys():
        n_blocks = get_n_blocks(filepath, params['blocksize'], params['blockmargin'])
        params['blocks'] = list(range(n_blocks))

    n_workers = get_n_workers(len(params['blocks']), params)

    print(params)
    arglist = [
        (
            filepath,
            params['blocksize'],
            params['blockmargin'],
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
            params['outputprefix'],
            False,
        )
        for b_idx in params['blocks']]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(combine_channels, arglist)


def get_n_workers(n_workers, params):
    """Determine the number of workers."""

    n_workers = min(n_workers, multiprocessing.cpu_count())

    try:
        n_workers = min(n_workers, params['n_workers'])
    except:
        pass

    return n_workers


def get_n_blocks(image_in, blocksize, blockmargin):

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    mpi = wmeMPI(usempi=False)
    mpi.set_blocks(im, blocksize, blockmargin)
    im.close()

    return len(mpi.blocks)


def combine_channels(
        image_in,
        blocksize=[],
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
        outputprefix='',
        usempi=False,
        ):
    """Average membrane and nuclear channels and write as blocks."""

    mpi = wmeMPI(usempi)

    im = Image(image_in, permission='r')
    im.load(comm=mpi.comm, load_data=False)

    if bias_image:
        bf = Image(bias_image, permission='r')
        bf.load(comm=mpi.comm, load_data=False)
    else:
        bf = None

    outputtemplate = '{}_{}.h5/ods'.format(outputprefix, '{}')
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


def transpose_props(props, outlayout=''):
    """Transpose the attributes of an image."""

    if not outlayout:
        outlayout = props['axlab'][::-1]
    in2out = [props['axlab'].index(l) for l in outlayout]
    props['elsize'] = np.array(props['elsize'])[in2out]
    props['slices'] = [props['slices'][i] for i in in2out]
    props['shape'] = np.array(props['shape'])[in2out]
    props['axlab'] = ''.join([props['axlab'][i] for i in in2out])
    if 'chunks' in props.keys():
        if props['chunks'] is not None:
            props['chunks'] = np.array(props['chunks'])[in2out]

    return props


def h5_nii_convert(image_in, image_out, datatype=''):
    """Convert between h5 (zyx) and nii (xyz) file formats."""

    im_in = Image(image_in)
    im_in.load(load_data=False)
    data = im_in.slice_dataset()

    props = transpose_props(im_in.get_props())
    if dataype:
        props['dtype'] = datatype

    im_out = Image(image_out, **props)
    im_out.create()
    im_out.write(data.transpose().astype(props['dtype']))
    im_in.close()
    im_out.close()


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


if __name__ == "__main__":
    main(sys.argv[1:])
