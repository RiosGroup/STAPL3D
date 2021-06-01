#!/usr/bin/env python

"""Perform planarity estimation.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

from glob import glob

import pathlib
import h5py

import numpy as np

import yaml
import xml.etree.ElementTree as ET

from stapl3d import (
    # parse_args_common,
    get_n_workers,
    get_outputdir,
    get_imageprops,
    get_params,
    get_paths,
    Image,
    )


def main(argv):
    """"Enhance the membrane with ACME.

    """

    # TODO: aggregate channels
    step_ids = ['split_channels', 'aggregate_channels']
#     fun_selector = {
#         'split': split_channels,
#         'aggregate': aggregate_channels,
#         }
#
#     args, mapper = parse_args_common(step_ids, fun_selector, *argv)
#
#     for step, step_id in mapper.items():
#         fun_selector[step](
#             args.image_in,
#             args.parameter_file,
#             step_id,
#             args.outputdir,
#             args.n_workers,
#             )


def split_channels(
    image_in,
    parameter_file,
    step_id='split_channels',
    outputdir='',
    n_workers=0,
    channels=[],
    image_ref='',
    outputpat='',
    channel_re='_ch{:02d}',
    insert=False,
    replace=False,
    ):
    """Split imarisfile into separate channels."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, 'channels')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    if not subparams['channels']:
        props = get_imageprops(image_in)
        n_channels = props['shape'][props['axlab'].index('c')]
        subparams['channels'] = list(range(n_channels))

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    paths = get_paths(image_in)
    filestem = os.path.splitext(paths['fname'])[0]

    if not params['image_ref']:
        filename = '{}{}.ims'.format(filestem, cfg['dataset']['ims_ref_postfix'])
        params['image_ref'] = os.path.join(paths['dir'], filename)

    if not params['outputpat']:
        params['outputpat'] = '{}{}.ims'.format(filestem, params['channel_re'])
        if params['insert']:
            dataset = cfg['dataset']['name']
            postfix = filestem.split(dataset)[-1]
            params['outputpat'] = '{}{}{}.ims'.format(dataset, params['channel_re'], postfix)

    arglist = [
        (
            image_in,
            params['image_ref'],
            ch,
            os.path.join(outputdir, params['outputpat'].format(ch)),
        )
        for ch in subparams['channels']]

    n_workers = get_n_workers(len(subparams['channels']), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(extract_channel, arglist)

    if params['replace']:
        channel_paths = [os.path.join(outputdir, params['outputpat'].format(ch))
                         for ch in subparams['channels']]
        make_aggregate(channel_paths, image_in, params['image_ref'])


def extract_channel(ims_path, ref_path, channel, outputpath):
    """Split an imarisfile into channels."""

    f = h5py.File(ims_path, 'r')
    n_channels = len(f['/DataSet/ResolutionLevel 0/TimePoint 0'])
    n_reslev = len(f['/DataSet'])

    outputdir = os.path.dirname(outputpath)
    os.makedirs(outputdir, exist_ok=True)
    shutil.copy2(ref_path, outputpath)
    g = h5py.File(outputpath, 'r+')

    diloc = '/DataSetInfo'
    chloc = '{}/Channel {}'.format(diloc, channel)
    g.require_group(diloc)
    try:
        del g['{}/Channel {}'.format(diloc, 0)]
    except KeyError:
        pass
    f.copy(f[chloc], g[diloc], name='Channel 0')

    for rl in range(n_reslev):

        tploc = '/DataSet/ResolutionLevel {}/TimePoint 0'.format(rl)
        chloc = '{}/Channel {}'.format(tploc, channel)
        g.require_group(tploc)
        try:
            del g['{}/Channel {}'.format(tploc, 0)]
        except KeyError:
            pass
        f.copy(chloc, g[tploc], name='Channel 0')

    g.close()

    f.close()


def aggregate_file(tgt_file, channels, ch_offset=0):
    """Create an aggregate file with links to individual channels."""

    for i, ch_dict in enumerate(channels):
        ch = ch_offset + i
        if i != 0:
            copy_dsi_int(tgt_file, ch, ch_dict['filepath'])
        linked_path = os.path.relpath(ch_dict['filepath'], os.path.dirname(tgt_file))
        create_linked_channel(tgt_file, ch, pathlib.Path(linked_path).as_posix())
        set_attr(tgt_file, ch, ch_dict)


def copy_dsi_int(tgt_file, tgt_ch, ext_file, ext_ch=0, prefix='/DataSetInfo'):
    """Copy the imaris DatasetInfo group."""

    f = h5py.File(tgt_file, 'r+')
    chname = 'Channel {}'.format(tgt_ch)
    try:
        del f['{}/{}'.format(prefix, chname)]
    except KeyError:
        pass
    f.copy('/DataSetInfo/Channel 0', '/DataSetInfo/{}'.format(chname))
    f.close()


def create_linked_channel(tgt_file, tgt_ch, ext_file, ext_ch=0):
    """Create a to an externally stored channel."""

    f = h5py.File(tgt_file, 'r+')
    n_reslev = len(f['/DataSet'])
    for rl in range(n_reslev):
        prefix = '/DataSet/ResolutionLevel {}/TimePoint 0'.format(rl)
        create_ext_link(f, tgt_ch, ext_file, ext_ch, prefix)
    f.close()


def create_ext_link(f, tgt_ch, ext_file, ext_ch=0, prefix=''):
    """Create an individual link."""

    tgt_loc = '{}/Channel {}'.format(prefix, tgt_ch)
    ext_loc = '{}/Channel {}'.format(prefix, ext_ch)
    try:
        del f[tgt_loc]
    except KeyError:
        pass
    f[tgt_loc] = h5py.ExternalLink(ext_file, ext_loc)


def set_attr(tgt_file, ch, ch_dict={}):
    """Set attributes of a channel."""

    f = h5py.File(tgt_file, 'r+')
    prefix = '/DataSetInfo'
    ch = f[prefix]['Channel {}'.format(ch)]
    attr_keys = ['Name', 'Color', 'ColorRange', 'ColorMode']
    for key, val in ch_dict.items():
        if key in attr_keys:
            ch.attrs[key] = np.array([c for c in ch_dict[key]], dtype='|S1')
    f.close()


def correct_histogram(infile):
    """Generate a new histogram for the channel."""

    def write_attribute(ch, name, val, formatstring='{:.3f}'):
        arr = [c for c in formatstring.format(val)]
        ch.attrs[name] = np.array(arr, dtype='|S1')

    f = h5py.File(infile)

    data = f['/DataSet/ResolutionLevel 4/TimePoint 0/Channel 0/Data']
    hist = np.histogram(data, bins=256)[0]
    histminmax = [0, 65535]

    for rl_idx in range(0, len(f['/DataSet'])):
        rl = f['/DataSet/ResolutionLevel {}'.format(rl_idx)]
        tp = rl['TimePoint 0']
        chn = tp['Channel 0']
        chn['Histogram'][:] = hist
        attributes = {
            'HistogramMin': (histminmax[0], '{:.3f}'),
            'HistogramMax': (histminmax[1], '{:.3f}'),
        }
        for k, v in attributes.items():
            write_attribute(chn, k, v[0], v[1])

    f.close()


def aggregate_channels(
    image_in,
    parameter_file,
    step_id='aggregate_channels',
    outputdir='',
    n_workers=0,
    channel_pat='_ch??',
    postfix='',
    color=[1, 1, 1],
    crange=[0, 20000],
    ):

    # TODO make option to get params from yml
    # split image_in on channel_pat to get inputstem [and postfix if doesn't exist]

    paths = get_paths(image_in)

    # outputfile =
    # ref_path =
    # inputstem = paths['base']
    # make_aggregate(
    #     outputfile,
    #     ref_path,
    #     inputstem,
    #     channel_pat,
    #     postfix,
    #     color,
    #     crange,
    #     )

def make_aggregate(outputfile, ref_path,
                   inputstem, channel_pat='_ch??', postfix='',
                   color=[1, 1, 1], crange=[0, 20000]):
    """Gather the inputfiles into an imarisfile by symbolic links."""

    inputfiles = glob('{}{}{}.ims'.format(inputstem, channel_pat, postfix))
    inputfiles.sort()

    channels = [
        {
         'filepath': inputfile,
         'Name': 'chan',
         'Color': ' '.join(['{:.3f}'.format(i) for i in color]),
         'ColorRange': ' '.join(['{:.3f}'.format(i) for i in crange]),
         'ColorMode': 'BaseColor',
         } for inputfile in inputfiles]

    shutil.copy2(ref_path, outputfile)
    aggregate_file(outputfile, channels, ch_offset=0)


def ref_to_zeros(image_in):
    """Set BigDataViewer datasets to all-zeros."""

    if '.ims' in image_in:
        ims_to_zeros(image_in)
    elif '.bdv' in image_in:
        bdv_to_zeros(image_in)


def ims_to_zeros(image_in):
    """Set imaris datasets to all-zeros."""

    im = h5py.File(image_in, 'r+')
    n_reslev = len(im['/DataSet'])
    n_timepoints = len(im['/DataSet/ResolutionLevel 0'])
    n_channels = len(im['/DataSet/ResolutionLevel 0/TimePoint 0'])
    for rl in range(n_reslev):
        for tp in range(n_timepoints):
            for ch in range(n_channels):
                tploc = '/DataSet/ResolutionLevel {}/TimePoint {}'.format(rl, tp)
                chloc = '{}/Channel {}'.format(tploc, ch)
                dsloc = '{}/Data'.format(chloc)
                ds = im[dsloc]
                ds[:] = np.zeros(ds.shape, dtype=ds.dtype)

    im.close()


def bdv_to_zeros(image_in):
    """Set BigDataViewer datasets to all-zeros."""

    f = h5py.File(image_in, 'r+')
    timepoints = [v for k, v in f.items() if k.startswith('t')]
    for tp in timepoints:
        channels = [v for k, v in tp.items() if k.startswith('s')]
        for ch in channels:
            for k, rl in ch.items():
                ds = rl['cells']
                ds[:] = np.zeros(ds.shape, dtype=ds.dtype)

    f.close()


def find_resolution_level(image_in):
    """Find the smallest resolution level not downsampled in Z."""

    mo = Image(image_in, permission='r')
    mo.load(load_data=False)

    # FIXME: streamline this
    z_dim = mo.axlab.index('z')
    if '.ims' in image_in:
        Z = int(mo.dims[z_dim])
    elif '.bdv' in image_in:
        Z = int(mo.bdv_get_dims(reslev=0)[z_dim])
    Z_rl = Z
    rl_idx = 0
    while Z == Z_rl:
        rl_idx += 1
        if '.ims' in image_in:
            rl = mo.file['/DataSet/ResolutionLevel {}'.format(rl_idx)]
            im_info = rl['TimePoint 0/Channel 0']
            Z_rl = int(att2str(im_info.attrs['ImageSizeZ']))
        elif '.bdv' in image_in:
            Z_rl = int(mo.bdv_get_dims(reslev=rl_idx)[z_dim])

    mo.close()

    return rl_idx - 1


def att2str(att):
    return ''.join([t.decode('utf-8') for t in att])


def find_downsample_factors(image_in, rl0_idx, rl1_idx):
    """Find downsample factors."""

    def find_dims(im, idx):
        rl = im.file['/DataSet/ResolutionLevel {}'.format(idx)]
        im_info = rl['TimePoint 0/Channel 0']
        return [int(att2str(im_info.attrs['ImageSize{}'.format(dim)]))
                for dim in 'ZYX']

    im = Image(image_in, permission='r')
    im.load(load_data=False)

    if im.format == '.ims':
        dims0 = find_dims(im, rl0_idx)
        dims1 = find_dims(im, rl1_idx)
        dsfacs = np.around(np.array(dims0) / np.array(dims1)).astype('int')
    elif im.format == '.bdv':
        dsfacs = im.file['s00/resolutions'][rl1_idx, :][::-1]
    else:
        dsfacs = [1, 1, 1]

    im.close()

    return dsfacs



def aggregate_hdf5(outputfile, inputpat, vol='data', xml_ref=''):
    """Gather the inputfiles into an imarisfile by symbolic links."""

    inputfiles = glob(inputpat)
    inputfiles.sort()

    h5chs_to_virtual(outputfile, inputpat, ids='data')
    channels = [
        {
         'filepath': inputfile,
         'Name': vol,
         'xml_ref': xml_ref,
         } for i, inputfile in enumerate(inputfiles)]

    # h5chs_to_virtual(outputfile, inputpat, postfix='')
    aggregate_h5_channels(outputfile, channels, ch_offset=0)


def aggregate_h5(outputfile, inputstem, channel_pat='_ch??', postfix='', xml_ref=''):
    """Gather the inputfiles into an imarisfile by symbolic links."""

    inputfiles = glob('{}{}{}.h5'.format(inputstem, channel_pat, postfix))
    inputfiles.sort()
    print(inputfiles)

    channels = [
        {
         'filepath': inputfile,
         'Name': 'data',  # TODO flexibilolize in argument
         'xml_ref': xml_ref,
         } for i, inputfile in enumerate(inputfiles)]

    aggregate_h5_channels(outputfile, channels, ch_offset=0)


def aggregate_h5_channels(tgt_file, channels, ch_offset=0):
    """Create an aggregate file with links to individual channels."""

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

    f = h5py.File(tgt_file, 'w')
    for ch_dict in channels:

        linked_path = os.path.relpath(ch_dict['filepath'], os.path.dirname(tgt_file))
        ext_file = pathlib.Path(linked_path).as_posix()

        create_ext_link(f, ch_dict['Name'], ext_file, ch_dict['Name'])


def aggregate_bigstitcher(outputfile, inputpat='_ch??', postfix='', ext='.h5', xml_ref=''):
    """Gather the inputfiles into an imarisfile by symbolic links."""

    inputfiles = glob('{}{}{}'.format(inputpat, postfix, ext))
    inputfiles.sort()

    channels = [
        {
         'filepath': inputfile,
         'Name': 's{:02d}'.format(i),
         'xml_ref': xml_ref,
         } for i, inputfile in enumerate(inputfiles)]

    aggregate_bigstitcher_channels(outputfile, channels, ch_offset=0, ext=ext)


def aggregate_bigstitcher_channels(tgt_file, channels, ch_offset=0, ext='.h5'):
    """Create an aggregate file with links to individual channels."""

    def bdv_load_elsize(xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        item = root.find('./SequenceDescription/ViewSetups/ViewSetup/voxelSize/size')
        elsize_xyz = [float(e) for e in item.text.split()]

        return elsize_xyz

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

    f = h5py.File(tgt_file, 'w')

    linked_path = os.path.relpath(channels[0]['filepath'], os.path.dirname(tgt_file))
    ext_file = pathlib.Path(linked_path).as_posix()
    tgt_loc = ext_loc = '__DATA_TYPES__'
    create_ext_link(f, tgt_loc, ext_file, ext_loc)

    for ch_dict in channels:

        xml_path = ch_dict['xml_ref'] or ch_dict['filepath'].replace(ext, '_stacks.xml')
        # FIXME: remove this workaround for: 'VoxelSize unit upon fuse is in pixels, not um'
        # xml_path = ch_dict['xml_ref'] or ch_dict['filepath'].replace(ext, '.xml')
        elsize = bdv_load_elsize(xml_path)

        linked_path = os.path.relpath(ch_dict['filepath'], os.path.dirname(tgt_file))
        ext_file = pathlib.Path(linked_path).as_posix()

        tgt_loc = '/{}'.format(ch_dict['Name'])
        ext_loc = '/s00'
        #create_copy(f, ch_dict['Name'], ch_dict['filepath'], 's00')
        create_ext_link(f, ch_dict['Name'], ext_file, 's00')

        tgt_loc = '/t00000/{}'.format(ch_dict['Name'])
        ext_loc = '/t00000/s00'
        create_ext_link(f, tgt_loc, ext_file, ext_loc)

        for k, rl in f[tgt_loc].items():
            ds = rl['cells'.format(tgt_loc)]
            res = f[ch_dict['Name']]['resolutions'][int(k), :]
            elsize_xyz_rl = np.array(elsize) * res
            ds.attrs['element_size_um'] = elsize_xyz_rl[::-1]
            for i, label in enumerate('zyx'):
                ds.dims[i].label = label


def bdv_to_virtual(outputfile, inputfile, reslev=0):

    f = h5py.File(inputfile, 'r')
    shape = f['/t00000/s00/{}/cells'.format(reslev)].shape + (len(f['t00000']),)
    dtype = f['/t00000/s00/{}/cells'.format(reslev)].dtype
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    for ch in range(shape[3]):
        image_in = "{}/t00000/s{:02d}/{}".format(inputfile, ch, reslev)
        vsource = h5py.VirtualSource(image_in, name='cells', shape=shape[:3])
        layout[:, :, :, ch] = vsource

    # Add virtual dataset to output file
    if not outputfile:
        filestem, ext = os.path.splitext(inputfile)
        outputfile = '{}{}{}'.format(filestem, '_virt', ext)
    with h5py.File(outputfile, 'w', libver='latest') as f:
        f.create_virtual_dataset('data', layout, fillvalue=0)


def h5chs_to_virtual(outputfile, inputpat, ids='data'):

    inputfiles = glob(inputpat)
    inputfiles.sort()

    try:
        f = h5py.File(inputfiles[0], 'r')
    except IndexError:
        return

    f = h5py.File(inputfiles[0], 'r')
    shape = f[ids].shape + (len(inputfiles),)
    dtype = f[ids].dtype

    try:
        axlab = [dim.label for dim in f[ids].dims]
    except:
        axlab = None
    try:
        elsize = [es for es in f[ids].attrs['element_size_um']]
    except KeyError:
        elsize = None

    f.close()
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    for ch, inputfile in enumerate(inputfiles):
        image_in = "{}".format(inputfile)
        vsource = h5py.VirtualSource(image_in, name=ids, shape=shape[:3])
        layout[:, :, :, ch] = vsource

    # Add virtual dataset to output file
    if not outputfile:
        filestem, ext = os.path.splitext(inputfiles[0])
        outputfile = '{}{}{}'.format(filestem, '_virt', ext)
    with h5py.File(outputfile, 'a', libver='latest') as f:
        f.create_virtual_dataset(ids, layout, fillvalue=0)

        if axlab is not None:
            for i, label in enumerate(axlab + ['c']):
                f[ids].dims[i].label = label
        if elsize is not None:
            f[ids].attrs['element_size_um'] = elsize + [1]


if __name__ == "__main__":
    main(sys.argv[1:])
