import os
import pathlib
import h5py
import shutil
import numpy as np
from stapl3d import Image


def split_channels(ims_path, ref_path, channels=[], outputpat=''):
    """Split an imarisfile into channels."""

    f = h5py.File(ims_path, 'r')
    n_channels = len(f['/DataSet/ResolutionLevel 0/TimePoint 0'])
    n_reslev = len(f['/DataSet'])

    if not channels:
        channels = list(range(n_channels))
    if not outputpat:
        outputpat = '{}{}.ims'.format(os.path.splitext(ims_path)[0], '_ch{:02d}')

    for ch in channels:

        outpath = outputpat.format(ch)
        shutil.copy2(ref_path, outpath)
        g = h5py.File(outpath, 'r+')

        diloc = '/DataSetInfo'
        chloc = '{}/Channel {}'.format(diloc, ch)
        g.require_group(diloc)
        try:
            del g['{}/Channel {}'.format(diloc, 0)]
        except KeyError:
            pass
        f.copy(f[chloc], g[diloc], name='Channel 0')

        for rl in range(n_reslev):

            tploc = '/DataSet/ResolutionLevel {}/TimePoint 0'.format(rl)
            chloc = '{}/Channel {}'.format(tploc, ch)
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


def make_aggregate(inputfiles, outputfile, ref_path):
    """Gather the inputfiles into an imarisfile by symbolic links."""

    channels = [
        {
         'filepath': inputfile,
         'Name': 'chan',
         'Color': ' '.join(['{:.3f}'.format(i) for i in [1, 1, 1]]),
         'ColorRange': ' '.join(['{:.3f}'.format(i) for i in [0, 20000]]),
         'ColorMode': 'BaseColor',
         } for inputfile in inputfiles]

    shutil.copy2(ref_path, outputfile)
    aggregate_file(outputfile, channels, ch_offset=0)
