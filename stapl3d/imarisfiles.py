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

from stapl3d import (
    get_n_workers,
    get_outputdir,
    get_imageprops,
    get_params,
    get_paths,
    Image,
    )

def split_channels(
    image_in,
    parameter_file,
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

    step_id = 'splitchannels'

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


def make_aggregate(outputfile, ref_path,
                   inputstem, channel_pat='_ch??', postfix='',
                   color=[1, 1, 1], crange=[0, 20000]):
    """Gather the inputfiles into an imarisfile by symbolic links."""

    inputfiles = glob('{}{}{}.ims'.format(inputstem, channel_pat, postfix))

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


def find_resolution_level(image_in):
    """Find the smallest resolution level not downsampled in Z."""

    mo = Image(image_in, permission='r')
    mo.load(load_data=False)

    Z = int(mo.dims[0])
    Z_rl = Z
    rl_idx = 0
    while Z == Z_rl:
        rl_idx += 1
        rl = mo.file['/DataSet/ResolutionLevel {}'.format(rl_idx)]
        im_info = rl['TimePoint 0/Channel 0']
        Z_rl = int(att2str(im_info.attrs['ImageSizeZ']))

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

    dims0 = find_dims(im, rl0_idx)
    dims1 = find_dims(im, rl1_idx)

    im.close()

    dsfacs = np.around(np.array(dims0) / np.array(dims1)).astype('int')

    return dsfacs
