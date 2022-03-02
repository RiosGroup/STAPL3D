# -*- coding: utf-8 -*-

import logging

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Michiel Kleinnijenhuis"
__email__ = 'M.Kleinnijenhuis@prinsesmaximacentrum.nl'

import os
import re
import sys
import h5py
import glob
import pickle
import random
import argparse
import itertools
import multiprocessing

from xml import etree as et

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import yaml
# from ruamel import yaml

sys.stdout = open(os.devnull, 'w')

try:
    import nibabel as nib
except ImportError:
    print("nibabel could not be loaded")

try:
    from skimage.io import imread, imsave
    from skimage.transform import downscale_local_mean
    from skimage.measure import block_reduce
except ImportError:  # , AttributeError
    print("scikit-image io could not be loaded")

try:
    import javabridge as jv
    import bioformats as bf
except ImportError:
    print("bioformats could not be loaded")

import errno

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

try:
    import DM3lib as dm3
except ImportError:
    print("dm3lib could not be loaded")

sys.stdout = sys.__stdout__

VM_STARTED = False
VM_KILLED = False


class Image(object):
    """

    """

    def __init__(self, path,
                 elsize=None, axlab=None, dtype='float', reslev=0,
                 shape=None, dims=None, dataslices=None, slices=None,
                 chunks=None, compression='gzip', series=0,
                 protective=False, permission='r+'):

        self.path = path
        self.elsize = elsize  # TODO: translations / affine
        self.axlab = axlab
        self.dtype = dtype
        self.reslev = reslev
        if shape is None:
            shape = dims
        self.dims = shape
        self.shape = shape
        if slices is not None:
            self.slices = slices
        else:
            self.slices = self.get_slice_objects(dataslices)
        self.chunks = chunks
        self.compression = compression
        self.series = series
        self.protective = protective
        self.permission = permission

        self.file = None
        self.filereader = None
        self.ds = []

        self.set_format()

    def set_format(self):
        """Set the format of the image."""

        self.format = self.get_format(self.path)

    def get_format(self, path_in):
        """Set the format of the image."""

        if os.path.isdir(path_in):
            files = sorted(glob.glob(os.path.join(path_in, '*')))
            if files:
                path = files[0]
            else:
                path = path_in
            return '.tifs'
        else:
            path = path_in

        for ext in ['.h5', '.nii', '.dm3', '.tif', '.ims', '.bdv', '.czi', '.lif']:
            if ext in path:
                return ext

        return '.dat'

    def output_check(self, outpaths, save_steps=True, protective=False):
        """Check output paths for writing."""

        if not outpaths['out']:
            status = "WARNING"
            info = "not writing results to file"
            print("{}: {}".format(status, info))
            return status

        # validate any additional output formats
        if 'addext' in outpaths.keys():
            root = outpaths['addext'][0]
            for ext in outpaths['addext'][1]:
                status = self.output_check_all(root, ext, None,
                                               save_steps, protective)
                if status == "CANCELLED":
                    return status

        # validate the main output
        root, ext = os.path.splitext(outpaths['out'])
        status = self.output_check_all(root, ext, outpaths,
                                       save_steps, protective)

        return status

    def output_check_all(self, root, ext, outpaths=None, save_steps=False, protective=False):

        if '.h5' in ext:
            if outpaths is None:
                status = self.output_check_h5(['{}{}'.format(root, ext)], save_steps, protective)
            else:
                status = self.output_check_h5(outpaths, save_steps, protective)
        elif '.nii' in ext:
            status = self.output_check_dir(['{}{}'.format(root, ext)], protective)
        else:  # directory with images assumed
            status = self.output_check_dir([root], protective)

        return status

    def output_check_h5(self, outpaths, save_steps=True, protective=False):

        try:
            root, ds_main = outpaths['out'].split('.h5')
            h5file_out = h5py.File(root + '.h5', 'a')

        except ValueError:
            status = "CANCELLED"
            info = "main output is not a valid h5 dataset"
            print("{}: {}".format(status, info))
            return status

        else:
            # create a group and set outpath for any intermediate steps
            for dsname, outpath in outpaths.items():
                if ((dsname != 'out') and save_steps and (not outpath)):
                    grpname = ds_main + "_steps"
                    try:
                        h5file_out[grpname]
                    except KeyError:
                        h5file_out.create_group(grpname)
                    outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)

            h5file_out.close()

            # check the path for each h5 output
            for _, outpath in outpaths.items():
                if outpath:
                    status, info = self.h5_check(outpath, protective)
                    print("{}: {}".format(status, info))
                    if status == "CANCELLED":
                        return status

    def output_check_dir(self, outpaths, protective):
        """Check output paths for writing."""

        status = ''
        for outpath in outpaths:
            if os.path.exists(outpath):
                if protective:
                    status = 'CANCELLED'
                    info = "protecting {}".format(outpath)
                    print("{}: {}".format(status, info))
                    return status
                else:
                    status = "WARNING"
                    info = 'overwriting {}'.format(outpath)
                    print("{}: {}".format(status, info))
        if not status:
            outdir = os.path.dirname(outpaths[0])
            status = "INFO"
            info = "writing to {}".format(outdir)
            print("{}: {}".format(status, info))

        return status

    def h5_check(self):
        """Check if dataset exists in a h5 file."""

        h5path = self.h5_split()  # TODO: groups
        try:
            h5file = h5py.File(h5path['file'], 'r+')
        except IOError:  # FIXME: it's okay for the file not to be there
            status = "INFO"
            info = "could not open {}".format(self.path)
#             raise UserWarning("could not open {}".format(self.path))
        else:
            if h5path['dset'] in h5file:
                if self.protective:  # TODO: raise error
                    status = "CANCELLED"
                    info = "protecting {}".format(self.path)
                    raise Exception("protecting {}".format(self.path))
                else:
                    status = "WARNING"
                    info = "overwriting {}".format(self.path)
#                     raise UserWarning("overwriting {}".format(self.path))
            else:
                status = "INFO"
                info = "writing to {}".format(self.path)
#                 raise UserWarning("writing to {}".format(self.path))
            h5file.close()

    def h5_check_chunks(self):
        """Make sure chunksize does not exceed dimensions."""

        if self.chunks is None:
            return

        self.chunks = tuple([cs if cs < dim else dim
                             for cs, dim in zip(list(self.chunks), self.dims)])

    def split_path(self, filepath='', fileformat=''):
        """Split path into components."""

        filepath = filepath or self.path
        fileformat = fileformat or self.format

        comps = {}

        if os.path.isdir(filepath):
            comps['dir'] = filepath
            return comps

        if fileformat == '.dat':
            return {'dir': '', 'ext': '.dat', 'base': '', 'fname': '', 'file': ''}
        if fileformat == '.tifs':
            comps['ext'] = 'tif'
        elif fileformat == '.pbf':
            comps['ext'] = os.path.splitext(filepath)[1]
        elif fileformat == '.nii':
            comps['ext'] = '.nii.gz'    # TODO: non-zipped
        else:
            comps['ext'] = fileformat

        if comps['ext'] not in filepath:
            raise Exception('{} not in path'.format(comps['ext']))

        comps['base'] = filepath.split(comps['ext'])[0]
        comps['dir'], comps['fname'] = os.path.split(comps['base'])
        comps['file'] = comps['base'] + comps['ext']

        #if comps['ext'] in ['.h5', '.bdv']:
        if comps['ext'] in ['.h5']:
            comps_int = self.h5_split_int(filepath.split(comps['ext'])[1], ext=comps['ext'])
            comps['int'] = filepath.split(comps['ext'])[1]
            comps.update(comps_int)
        else:
            pass  # TODO: groups/dset from fname

        return comps

    def h5_split_int(self, path_int='', ext='.h5'):
        """Split components of a h5 path."""

        path_int = path_int or self.path.split(ext)[1]

        comps = {}

        if '/' not in path_int:
            raise Exception('no groups or dataset specified for hdf5 path')

        int_comp = path_int.split('/')
        comps['groups'] = int_comp[1:-1]
        comps['dset'] = int_comp[-1]

        return comps

    def h5_split(self, ext='.h5'):
        """Split components of a h5 path."""

        h5path = {}

        h5path['ext'] = ext

        if h5path['ext'] not in self.path:
            raise Exception('{} not in path'.format(h5path['ext']))

        h5path['base'], h5path['int'] = self.path.split(h5path['ext'])

        if self.format == '.ims':
            if not h5path['int']:
                if self.reslev == -1:
                    h5path['int'] = '/DataSet/ResolutionLevel 0'
                else:
                    h5path['int'] = '/DataSet/ResolutionLevel {}'.format(self.reslev)

        if self.format == '.bdv':
            if not h5path['int']:
                h5path['int'] = '/t00000'

        if '/' not in h5path['int']:
            raise Exception('no groups or dataset specified for hdf5 path')

        h5path_int_comp = h5path['int'].split('/')
        h5path['groups'] = h5path_int_comp[1:-1]
        h5path['dset'] = h5path_int_comp[-1]

        h5path['file'] = h5path['base'] + h5path['ext']
        h5path['dir'], h5path['fname'] = os.path.split(h5path['base'])

        return h5path

    def nii_split(self):
        """Split components of a nii path."""

        path_comps = {}

        if '.nii' not in self.path:
            raise Exception('.nii not in path')

        path_comps['base'] = self.path.split('.nii')[0]
        path_comps['ext'] = '.nii.gz'  # FIXME: non-zipped
#         path_comps['dir'], path_comps['fname'] = os.path.split(path_comps['base'])

#         h5path_int_comp = h5path['int'].split('/')
#         h5path['groups'] = h5path_int_comp[1:-1]
#         h5path['dset'] = h5path_int_comp[-1]
#
#         h5path['file'] = h5path['base'] + '.h5'

        return path_comps

    def h5_open(self, permission, comm=None):
        """Open a h5 file."""

        h5path = self.h5_split(ext=self.format)
#         if isinstance(self.file, h5py.File):
#             pass
#         else:
        if comm is None:
            self.file = h5py.File(h5path['file'], permission)
        else:
            self.file = h5py.File(h5path['file'], permission,
                                  driver='mpio', comm=comm)

    def load(self, comm=None, load_data=True):
        """Load a dataset."""

        if not self.path:
            pass

        formats = {'.h5': self.h5_load,
                   '.nii': self.nii_load,
                   '.dm3': self.dm3_load,
                   '.ims': self.ims_load,
                   '.bdv': self.bdv_load,
                   '.czi': self.czi_load,
                   '.lif': self.lif_load,
                   '.pbf': self.pbf_load,
                   '.tif': self.tif_load,  # NOTE: may need pbf for 3D tif
                   '.tifs': self.tifs_load,
                   '.dat': self.dat_load,
                   }

        formats[self.format](comm, load_data)

        self.set_slices()

        self.elsize = self.get_elsize()
        self.axlab = self.get_axlab()

    def dat_load(self, comm=None, load_data=True):
        """Load a h5 dataset."""

        pass

    def h5_load(self, comm=None, load_data=True):
        """Load a h5 dataset."""

        self.h5_open(self.permission, comm)
        h5path = self.h5_split(ext=self.format)
        self.ds = self.file[h5path['int']]
        self.dims = self.ds.shape
        self.dtype = self.ds.dtype
        self.chunks = self.ds.chunks

#         if load_data:
#             data, slices = self.load_dataset()

    def nii_load(self, comm=None, load_data=True):
        """Load a nifti dataset."""

        self.file = nib.load(self.path)
        self.ds = self.file.dataobj
        self.dims = self.ds.shape
        self.dtype = self.ds.dtype

#         if load_data:
#             data, slices = self.load_dataset()

    def tif_load(self, comm=None, load_data=True):
        """Load a 3D tif dataset."""

        self.file = None
        data = imread(self.path)
        self.dims = data.shape
        self.dtype = data.dtype

        if load_data:
            self.ds = data

    def tifs_load(self, comm=None, load_data=True):
        """Load a stack of tifs."""

        self.file = sorted(glob.glob(os.path.join(self.path, "*.tif*")))
        self.ds = []
        self.dims = np.array([len(self.file)] + self.tif_get_yxdims())
        self.dtype = self.tif_load_dtype()

        if load_data:  # TODO: slices (see dm3)
            for fpath in self.file:
                self.ds.append(self.get_image(fpath))
            self.ds = np.array(self.ds, dtype=self.dtype)

    def ims_load(self, comm=None, load_data=True):

        self.h5_open(self.permission, comm)
        h5path = self.h5_split(ext=self.format)
        self.ds = self.file[h5path['int']]
        self.dims = self.ims_get_dims()
        ch0 = self.ds['TimePoint 0/Channel 0/Data']
        self.dtype = ch0.dtype
        self.chunks = list(ch0.chunks) + [1, 1]

    def bdv_load(self, comm=None, load_data=True):

        self.h5_open(self.permission, comm)
        h5path = self.h5_split(ext=self.format)
        self.ds = self.file['/']
        self.dims = self.bdv_get_dims(self.reslev)
        ch0 = self.ds['t00000/s00/{}/cells'.format(self.reslev)]
        self.dtype = ch0.dtype
        self.chunks = list(ch0.chunks) + [1, 1]

    def czi_load(self, comm=None, load_data=True):

        import czifile
        self.file = czifile.CziFile(self.path)

        from tifffile import create_output
        from stapl3d.preprocessing import shading
        iminfo = shading.get_image_info(self.path)
        data = create_output(None, iminfo['zstack_shape'], iminfo['dtype'])
        data = np.squeeze(shading.read_zstack(self.path, 0, data))
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        if len(data.shape) == 4:
            data = np.transpose(data, axes=[1, 2, 3, 0])  # czyx to zyxc

        self.ds = data
        self.dims = data.shape
        self.dtype = data.dtype
        self.chunks = None

    def lif_load(self, comm=None, load_data=True):

        from readlif.reader import LifFile
        self.file = LifFile(self.path).get_image(0)  # TODO: choice of image / series

        from tifffile import create_output
        from stapl3d.preprocessing import shading
        iminfo = shading.get_image_info(self.path)
        data = create_output(None, iminfo['zstack_shape'], iminfo['dtype'])
        data = np.squeeze(shading.read_zstack(self.path, 0, data))
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        if len(data.shape) == 4:
            data = np.transpose(data, axes=[1, 2, 3, 0])  # czyx to zyxc

        self.ds = data
        self.dims = data.shape
        self.dtype = data.dtype
        self.chunks = None

    def bdv_get_dims(self, reslev=0):

        tps = [v for k, v in self.file.items() if k.startswith('t')]
        n_timepoints = len(tps)
        chs = [v for k, v in tps[0].items() if k.startswith('s')]
        n_channels = len(chs)
        rls = [v for k, v in chs[0].items()]
        n_reslevs = len(rls)

        dims_zyx = rls[reslev]['cells'].shape

        return list(dims_zyx) + [len(chs), len(tps)]

    def pbf_load(self, comm=None, load_data=True):
        """Load a dataset with python bioformats."""

        if not VM_STARTED:
            start()
        if VM_KILLED:
            raise RuntimeError("The Java Virtual Machine has already been "
                               "killed, and cannot be restarted. See the "
                               "python-javabridge documentation for more "
                               "information. You must restart your program "
                               "and try again.")

        md = bf.get_omexml_metadata(self.path)
        names, sizes, elsizes, axlabs, dtypes = parse_xml_metadata(md)
        if self.format == '.ims':
            self.h5_open('r', comm)
        else:
            self.file = bf.ImageReader(self.path)
        self.filereader = bf.ImageReader(self.path)
        self.ds = []
        self.dims = sizes[0]
        self.dtype = dtypes[0]

        if load_data:
            self.set_slices()
            self.elsize = list(elsizes[0])
            self.axlab = axlabs[0]
            self.ds = self.pbf_load_data()

    def pbf_load_data(self):

        # FIXME: same as h5 etc
        self.dims = [dim for dim in self.slices2shape()]
        data = np.empty(self.dims, self.dtype)

        i = {al: self.axlab.index(al) for al in self.axlab}
        slices = {al: self.slices[i[al]] for al in self.axlab}
        iterators = {al: range(slices[al].start, slices[al].stop, 1) for al in self.axlab}

        xywh = (slices['x'].start,
                slices['y'].start,
                slices['x'].stop - slices['x'].start,
                slices['y'].stop - slices['y'].start)

        dims_dict = {'x': xywh[2], 'y': xywh[3], 'z': 1, 'c': 1, 't': 1}
        shape_xy = [dims_dict[al] for al in self.axlab]

        for c_idx in iterators['c']:
            for z_idx in iterators['z']:
                for t_idx in iterators['t']:

                    c_idx_n = c_idx - slices['c'].start
                    z_idx_n = z_idx - slices['z'].start
                    t_idx_n = t_idx - slices['t'].start
                    slc_dict = {
                        'x': slice(0, xywh[2], 1),
                        'y': slice(0, xywh[3], 1),
                        'z': slice(z_idx_n, z_idx_n + 1, 1),
                        'c': slice(c_idx_n, c_idx_n + 1, 1),
                        't': slice(t_idx_n, t_idx_n + 1, 1),
                    }
                    slcs = [slc_dict[al] for al in self.axlab]

                    data_xy = self.filereader.read(c=c_idx, z=z_idx, t=t_idx, series=self.series,
                                                   wants_max_intensity=False, rescale=False, XYWH=xywh)
                    data[tuple(slcs)] = np.reshape(data_xy, shape_xy)

        return np.array(data)

    def pbf_get_dims(self):

        md = bf.get_omexml_metadata(self.path)
        names, dims, elsizes, axlabs, dtypes = parse_xml_metadata(md)

        return dims

    def dm3_load(self, comm=None, load_data=True):
        """Load a stack of dm3s."""

        self.file = sorted(glob.glob(os.path.join(self.path, "*.dm3")))
        self.ds = []
        self.dims = np.array([len(self.file)] + self.dm3_get_yxdims())
        self.dtype = self.dm3_load_dtype()

        if load_data:
            slcs_z = self.slices[self.axlab.index('z')]
            slcs_y = self.slices[self.axlab.index('y')]
            slcs_x = self.slices[self.axlab.index('x')]
            for fpath in self.file[slcs_z]:
                im = self.get_image(fpath)
                im = im[slcs_y, slcs_x]
                self.ds.append(im)
            self.ds = np.array(self.ds, dtype=self.dtype)

    def get_props(self, shape=[], elsize=[], axlab='',
                  chunks=False, slices=[], dtype='',
                  protective=False, squeeze=False):

        props = {'shape': shape or list(self.slices2shape()),
                 'elsize': elsize or list(self.elsize),
                 'axlab': axlab or str(self.axlab),
                 'chunks': chunks or self.chunks,  # FIXME: can be None
                 'slices': slices or list(self.slices),
                 'dtype': dtype or self.dtype,
                 'protective': protective or self.protective}

        if squeeze:
            props = self.squeeze_props(props)

        return props

    def squeeze_props(self, props=None, dim=None):

        # FIXME: list returned for axlab
        if props is None:
            props = self.get_props()

        if dim is None:
            if 'c' in self.axlab:
                dim = self.axlab.index('c')
            else:
                return props

        squeezable = ['shape', 'elsize', 'axlab', 'chunks', 'slices']
        for prop in squeezable:

            val = props[prop]
            if val is not None:
                props[prop] = list(val)
                del props[prop][dim]
            else:
                props[prop] = val

        return props

    def squeeze_channel(self, dim=None):

        props = {'shape': list(self.slices2shape()),
                 'elsize': self.elsize,
                 'axlab': self.axlab,
                 'chunks': self.chunks,
                 'slices': self.slices}

        if dim is None:
            if 'c' in self.axlab:
                dim = self.axlab.index('c')
            else:
                return props

        squeezed = {}
        for prop, val in props.items():
            if val is not None:
                squeezed[prop] = list(val)
                del squeezed[prop][dim]
            else:
                squeezed[prop] = val

        return squeezed

    def slice_dataset(self, squeeze=True):

        ndim = self.get_ndim()
        if self.slices is None:
            self.set_slices()
        slices = self.slices

        if self.format == '.pbf':
            data = self.pbf_load_data()
        elif self.format == '.ims':
            data = self.slice_dataset_ims(self.ds, slices)
        elif self.format == '.bdv':
            data = self.slice_dataset_bdv(self.ds, slices)
        else:
            if ndim == 1:
                data = self.ds[slices[0]]
            elif ndim == 2:
                data = self.ds[slices[0],
                               slices[1]]
            elif ndim == 3:
                data = self.ds[slices[0],
                               slices[1],
                               slices[2]]
            elif ndim == 4:
                data = self.ds[slices[0],
                               slices[1],
                               slices[2],
                               slices[3]]
            elif ndim == 5:
                data = self.ds[slices[0],
                               slices[1],
                               slices[2],
                               slices[3],
                               slices[4]]

#         self.ds = np.squeeze(data)  # FIXME?

        if squeeze:
            data = np.squeeze(data)

        return data

    def slice_dataset_ims(self, rs0_group, slices):
        """

        NOTE: this is a zero-padded version of the dataset.
        """

        slcs = [slices[self.axlab.index('z')],
                slices[self.axlab.index('y')],
                slices[self.axlab.index('x')],
                slices[self.axlab.index('c')],
                slices[self.axlab.index('t')]]

        dims = [len(range(*slc.indices(slc.stop))) for slc in slcs]
        data = np.empty(dims, dtype=self.dtype)

        if slcs[4].start is not None:
            t_start = int(slcs[4].start)
        else:
            t_start = None
        if slcs[4].start is not None:
            t_stop = int(slcs[4].stop)
        else:
            t_stop = None
        if slcs[4].step is not None:
            t_step = int(slcs[4].step)
        else:
            t_step = None
        if slcs[3].start is not None:
            c_start = int(slcs[3].start)
        else:
            c_start = None
        if slcs[3].stop is not None:
            c_stop = int(slcs[3].stop)
        else:
            c_stop = None
        if slcs[3].step is not None:
            c_step = int(slcs[3].step)
        else:
            c_step = None

        tp_names = ['TimePoint {}'.format(i) for i in range(len(rs0_group))]
        timepoints_sorted = [(tp_name, rs0_group[tp_name]) for tp_name in tp_names]
        t_iter_slc = itertools.islice(timepoints_sorted, t_start, t_stop, t_step)

        for tp_idx, (_, tp) in enumerate(t_iter_slc):

            ch_names = ['Channel {}'.format(i) for i in range(len(tp))]
            channels_sorted = [(ch_name, tp[ch_name]) for ch_name in ch_names]
            c_iter_slc = itertools.islice(channels_sorted, c_start, c_stop, c_step)

            for ch_idx, (_, ch) in enumerate(c_iter_slc):

                data_tmp = ch['Data'][slcs[0], slcs[1], slcs[2]]
                data[..., ch_idx, tp_idx] = data_tmp

        return data

    def slice_dataset_bdv(self, rs0_group, slices):
        """

        NOTE: this is a zero-padded version of the dataset.
        """

        slcs = [slices[self.axlab.index('z')],
                slices[self.axlab.index('y')],
                slices[self.axlab.index('x')],
                slices[self.axlab.index('c')],
                slices[self.axlab.index('t')]]

        dims = [len(range(*slc.indices(slc.stop))) for slc in slcs]
        data = np.empty(dims, dtype=self.dtype)

        if slcs[4].start is not None:
            t_start = int(slcs[4].start)
        else:
            t_start = None
        if slcs[4].start is not None:
            t_stop = int(slcs[4].stop)
        else:
            t_stop = None
        if slcs[4].step is not None:
            t_step = int(slcs[4].step)
        else:
            t_step = None
        if slcs[3].start is not None:
            c_start = int(slcs[3].start)
        else:
            c_start = None
        if slcs[3].stop is not None:
            c_stop = int(slcs[3].stop)
        else:
            c_stop = None
        if slcs[3].step is not None:
            c_step = int(slcs[3].step)
        else:
            c_step = None

        tp_names = [k for k, v in rs0_group.items() if k.startswith('t')]
        timepoints_sorted = [(tp_name, rs0_group[tp_name]) for tp_name in tp_names]
        t_iter_slc = itertools.islice(timepoints_sorted, t_start, t_stop, t_step)
        for tp_idx, (_, tp) in enumerate(t_iter_slc):
            ch_names = [k for k, v in tp.items() if k.startswith('s')]
            channels_sorted = [(ch_name, tp[ch_name]) for ch_name in ch_names]
            c_iter_slc = itertools.islice(channels_sorted, c_start, c_stop, c_step)
            for ch_idx, (_, ch) in enumerate(c_iter_slc):
                data_tmp = ch['{}/cells'.format(self.reslev)][slcs[0], slcs[1], slcs[2]]
                data[..., ch_idx, tp_idx] = data_tmp

        return data

    def load_dataset(self):

        self.slice_dataset()
        self.format = '.dat'
        self.file = None
        self.set_dims()

    def normalize_data(self, data):
        """Normalize data between 0 and 1."""

        data = data.astype('float64')
        datamin = np.amin(data)
        datamax = np.amax(data)
        data -= datamin
        data *= 1/(datamax-datamin)

        return data, [datamin, datamax]

    def remove_singleton_props(self, props, dim):

        del props['shape'][dim]
        del props['elsize'][dim]

        props['axlab'] = props['axlab'].replace(props['axlab'][dim], '')

        if props['chunks'] is not None:
            del props['chunks'][dim]

        if props['slices'] is not None:
            del props['slices'][dim]

        return props

    def remove_singleton(self, dim):

        del self.dims[dim]

        del self.elsize[dim]

        self.axlab = self.axlab.replace(self.axlab[dim], '')

        if self.chunks is not None:
            del self.chunks[dim]

        if self.slices is not None:
            del self.slices[dim]

        #FIXME
#         self.ds = np.squeeze(self.ds, dim)

    def get_ndim(self):
        """Return the cardinality of the dataset."""

        if self.format == '.nii':
            ndim = len(self.file.header.get_data_shape())
        else:
            ndim = len(self.dims)

        return ndim

    def set_dims(self):
        """Set the dims property to the shape of the dataset."""

        self.dims = self.ds.shape

    def set_dataslices(self):
        """Set the dataslices property to the full shape of the dataset."""

        if self.dataslices is None:
            self.dataslices = []
            for dim in self.dims:
                self.dataslices += [0, dim, 1]

    def get_offset_slices(self, offsets, slices=None):
        """."""

        if slices is None:
            slices = list(self.slices)

        slcs = [slice(slc.start + o, slc.stop + o, slc.step)
                for o, slc in zip(offsets, slices)]

        return slcs

    def set_slices(self):

        if self.slices is None:
            self.slices = [slice(0, dim, 1) for dim in self.dims]
        else:
            slices = []
            for slc, dim in zip(self.slices, self.dims):
                if slc.stop == 0:
                    slices.append(slice(slc.start, dim, slc.step))
                else:
                    slices.append(slc)
            self.slices = slices

    def get_slice_objects(self, dataslices, dims=None, offsets=None):
        """Get the full ranges for z, y, x if upper bound is undefined."""

        if dataslices is None:
            return

        if offsets is None:
            offsets = [0 for _ in dataslices[::3]]

        starts = dataslices[::3] + offsets
        stops = dataslices[1::3] + offsets
#         stops = [dim if stop == 0 else stop
#                  for stop, dim in zip(stops, dims)]
        steps = dataslices[2::3]
        slices = [slice(start, stop, step)
                  for start, stop, step in zip(starts, stops, steps)]

        return slices

    def slices2shape(self, slices=None):
        """Get the shape of the sliced dataset."""

        if slices is None:
            if self.slices is None:
                self.set_slices()
            slices = self.slices

        return (len(range(*slc.indices(slc.stop))) for slc in slices)

#         if len(slices) == 1:
#             shape = (len(range(*slices[0].indices(slices[0].stop))))
#         if len(slices) == 2:
#             shape = (len(range(*slices[0].indices(slices[0].stop))),
#                      len(range(*slices[1].indices(slices[1].stop))))
#         if len(slices) == 3:
#             shape = (len(range(*slices[0].indices(slices[0].stop))),
#                      len(range(*slices[1].indices(slices[1].stop))),
#                      len(range(*slices[2].indices(slices[2].stop))))
#         elif len(slices) == 4:
#             shape = (len(range(*slices[0].indices(slices[0].stop))),
#                      len(range(*slices[1].indices(slices[1].stop))),
#                      len(range(*slices[2].indices(slices[2].stop))),
#                      len(range(*slices[3].indices(slices[3].stop))))
#
#         return shape

    def get_elsize(self):
        """Get the element sizes."""

        if ((self.elsize is not None) and (len(self.elsize) > 0)):
            elsize = self.elsize
#             print("""WARNING:
#                   elsize already specified as {}""".format(self.elsize))
            return elsize

        formats = {'.h5': self.h5_load_elsize,
                   '.nii': self.nii_load_elsize,
                   '.dm3': self.dm3_load_elsize,
                   '.ims': self.ims_load_elsize,
                   '.bdv': self.bdv_load_elsize,
                   '.czi': self.czi_load_elsize,
                   '.lif': self.lif_load_elsize,
                   '.pbf': self.pbf_load_elsize,
                   '.tif': self.tif_load_elsize,
                   '.tifs': self.tifs_load_elsize}
        elsize = formats[self.format]()

        if elsize is None:
            elsize = np.array([1.] * self.get_ndim())
#             raise Exception("""WARNING: elsize is None;
#                                replaced by {}""".format(elsize))

        return elsize

    def h5_load_elsize(self):
        """Get the element sizes from a dataset."""

        if 'element_size_um' in self.ds.attrs.keys():
            elsize = self.ds.attrs['element_size_um']
        else:
            elsize = None

        return elsize

    def nii_load_elsize(self):
        """Get the element sizes from a dataset."""

        elsize = list(self.file.header.get_zooms())

        return elsize

    def dm3_load_elsize(self):
        """Get the element sizes from a dataset."""

        dm3f = dm3.DM3(self.file[0], debug=0)

        id_dm3 = 'root.ImageList.1.ImageData'
        tag = '{}.Calibrations.Dimension.{:d}.Scale'
        elsize_x = float(dm3f.tags.get(tag.format(id_dm3, 0)))
        elsize_y = float(dm3f.tags.get(tag.format(id_dm3, 1)))

        tag = 'root.ImageList.1.ImageTags.SBFSEM.Record.Slice thickness'
        slicethickness = float(dm3f.tags.get(tag))
        elsize_z = slicethickness / 1000

        return [elsize_z, elsize_y, elsize_x]

    def pbf_load_elsize(self):

        md = bf.get_omexml_metadata(self.path)
        elsizes = parse_xml_metadata(md)[2]

        return list(elsizes[0])

    def ims_load_elsize(self):

        def att2str(att):
            return ''.join([t.decode('utf-8') for t in att])

        im_info = self.file['/DataSetInfo/Image']

        extmin0 = float(att2str(im_info.attrs['ExtMin0']))
        extmin1 = float(att2str(im_info.attrs['ExtMin1']))
        extmin2 = float(att2str(im_info.attrs['ExtMin2']))
        extmax0 = float(att2str(im_info.attrs['ExtMax0']))
        extmax1 = float(att2str(im_info.attrs['ExtMax1']))
        extmax2 = float(att2str(im_info.attrs['ExtMax2']))

        extX = extmax0 - extmin0
        extY = extmax1 - extmin1
        extZ = extmax2 - extmin2

        dims = self.ims_get_dims()
        elsizeX = extX / dims[2]
        elsizeY = extY / dims[1]
        elsizeZ = extZ / dims[0]

        return [elsizeZ, elsizeY, elsizeX, 1, 1]

    def ims_get_dims(self):

        def att2str(att):
            return ''.join([t.decode('utf-8') for t in att])

        im_info = self.ds['TimePoint 0/Channel 0']
        dimZ = int(att2str(im_info.attrs['ImageSizeZ']))
        dimY = int(att2str(im_info.attrs['ImageSizeY']))
        dimX = int(att2str(im_info.attrs['ImageSizeX']))
        dimC = len(self.ds['TimePoint 0'])
        dimT = len(self.ds)

        return [dimZ, dimY, dimX, dimC, dimT]

    def bdv_load_elsize(self):
        """Get the element sizes from a dataset."""

        ds = self.ds['t00000/s00'][str(self.reslev)]['cells']
        elsize = ds.attrs['element_size_um']

        return list(elsize) + [1, 1]

    def czi_load_elsize(self):
        """Get the element sizes from a dataset."""

        from xml.etree import cElementTree as etree
        import czifile

        segment = czifile.Segment(self.file._fh, self.file.header.metadata_position)
        data = segment.data().data()
        md = etree.fromstring(data.encode('utf-8'))

        elsize_z = float(md.findall('.//ScalingZ')[0].text) * 1e6
        elsize_y = float(md.findall('.//ScalingY')[0].text) * 1e6
        elsize_x = float(md.findall('.//ScalingX')[0].text) * 1e6

        elsize = [elsize_y, elsize_x]
        if len(self.dims) > 2:
            elsize = [elsize_z] + elsize
        if len(self.dims) > 3:
            elsize += [1]
        if len(self.dims) > 4:
            elsize += [1]

        return elsize

    def lif_load_elsize(self):
        """Get the element sizes from a dataset."""

        idxs = [4, 5]
        if len(self.dims) == 3:
            idxs = [1] + idxs
        if len(self.dims) == 4:
            idxs += [0]

        elsize = [1./self.file.scale[idx] for idx in idxs]

        return elsize

    def tif_load_elsize(self):
        """Get the element sizes from a dataset."""

        elsize = None

        return elsize

    def tifs_load_elsize(self):
        """Get the element sizes from a dataset."""

        elsize = None

        return elsize

    def get_axlab(self):
        """Get the axis labels."""

        if ((self.axlab is not None) and (len(self.axlab) > 0)):
            axlab = ''.join(self.axlab)
#             print("""WARNING:
#                   axlab already specified as {}""".format(self.axlab))
            return axlab

        formats = {'.h5': self.h5_load_axlab,
                   '.nii': self.nii_load_axlab,
                   '.dm3': self.dm3_load_axlab,
                   '.ims': self.ims_load_axlab,
                   '.bdv': self.bdv_load_axlab,
                   '.czi': self.czi_load_axlab,
                   '.lif': self.lif_load_axlab,
                   '.pbf': self.pbf_load_axlab,
                   '.tif': self.tif_load_axlab,
                   '.tifs': self.tifs_load_axlab}
        axlab = formats[self.format]()

        if axlab is None:
            axlab = 'zyxct'[:self.get_ndim()]
#             raise Exception("""WARNING: axlab is None;
#                                replaced by {}""".format(axlab))

        return axlab

    def h5_load_axlab(self):
        """Get the dimension labels from a dataset."""

        if 'DIMENSION_LABELS' in self.ds.attrs.keys():
            try:
                axlab = b''.join(self.ds.attrs['DIMENSION_LABELS']).decode("utf-8")
            except TypeError:
                axlab = ''.join(self.ds.attrs['DIMENSION_LABELS'])
        else:
            axlab = None

        return axlab

    def nii_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'xyztc'[:self.get_ndim()]  # FIXME: get from header?

        return axlab

    def dm3_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'zyxct'[:self.get_ndim()]  # FIXME: get from header?

        return axlab

    def czi_load_axlab(self):
        """Get the element sizes from a dataset."""

        axlab = 'yx'
        if len(self.dims) > 2:
            axlab = 'z' + axlab
        if len(self.dims) > 3:
            axlab += 'c'
        if len(self.dims) > 4:
            axlab += 't'

        return axlab

    def lif_load_axlab(self):
        """Get the element sizes from a dataset."""

        axlab = 'yx'
        if len(self.dims) == 3:
            axlab = 'z' + axlab
        if len(self.dims) == 4:
            axlab += 'c'

        return axlab

    def pbf_load_axlab(self):

        md = bf.get_omexml_metadata(self.path)
        axlabs = parse_xml_metadata(md)[3]

        return axlabs[0]

    def ims_load_axlab(self):

        return 'zyxct'

    def bdv_load_axlab(self):

        return 'zyxct'

    def tif_load_axlab(self):
        """Get the dimension labels from a dataset."""

        # FIXME or describe: assumptions
        al = {2: 'yx', 3: 'zyx', 4: 'zyxc', 5:'zyxct'}

        return al[self.get_ndim()]

    def tifs_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'zyxct'[:self.get_ndim()]  # FIXME: get from header?

        return axlab

    def h5_load_attributes(self):
        """Load attributes from a dataset."""

        ndim = self.get_ndim()
        self.h5_load_elsize(np.array([1] * ndim))
        self.h5_load_axlab('zyxct'[:ndim])

    def h5_write_elsize(self):
        """Write the element sizes to a dataset."""

        if self.elsize is not None:
            self.ds.attrs['element_size_um'] = self.elsize

    def h5_write_axlab(self):
        """Write the dimension labels to a dataset."""

        if self.axlab is not None:
            for i, label in enumerate(self.axlab):
                self.ds.dims[i].label = label

    def h5_write_attributes(self):
        """Write attributes to a dataset."""

        self.h5_write_elsize()
        self.h5_write_axlab()

    def create(self, comm=None):
        """Create a dataset."""

#         self.output_check()  # TODO
        if not self.path:
            pass

        formats = {'.h5': self.h5_create,
                   '.ims': self.ims_create,
                   '.bdv': self.bdv_create,  # TODO
                   '.nii': self.nii_create,
                   '.tif': self.tif_create,
                   '.tifs': self.tifs_create,
                   '.dat': self.dat_create}

        formats[self.format](comm)

        self.set_slices()

    def h5_create(self, comm=None):
        """Create a h5 dataset."""

        if comm is not None:
            self.chunks = None
            self.compression = None

        self.h5_open('a', comm)

        h5path = self.h5_split()

        parent = self.file
        for grpname in h5path['groups']:
            try:
                parent = parent[grpname]
            except KeyError:
                parent = parent.create_group(grpname)

        if h5path['dset'] in parent:
            self.ds = parent[h5path['dset']]
        else:
            self.h5_check_chunks()
            self.h5_create_dset(parent, h5path['dset'], comm)
            self.h5_write_attributes()

    def h5_create_dset(self, parent, dset_name, comm=None):
        """Create a h5 dataset."""

        if comm is None:
            self.ds = parent.create_dataset(dset_name,
                                            shape=self.dims,
                                            dtype=self.dtype,
                                            chunks=self.chunks,
                                            compression=self.compression)
        else:
            self.ds = parent.create_dataset(dset_name,
                                            shape=self.dims,
                                            dtype=self.dtype)

    def ims_create(self, comm=None):

        datatype = self.dtype
        if datatype == 'float64':
            datatype = 'float32'
        self.load(comm, load_data=False)
        self.dims = self.ims_get_dims()

        # TODO: create new file, copy all but channel data, and use ref from other??
        ch0_idx = 0
        ch0_name = 'Channel {}'.format(ch0_idx)
        ch0_info = '/DataSetInfo/{}'.format(ch0_name)

        chn_idx = self.dims[3]
        chn_name = 'Channel {}'.format(chn_idx)
        chn_info = '/DataSetInfo/{}'.format(chn_name)

        self.file[ch0_info].copy(ch0_info, chn_info)
        nr = len(self.file['/DataSet'])

        for tp_idx in range(0, self.dims[4]):

            for rl_idx in range(0, nr):

                rl = self.file['/DataSet/ResolutionLevel {}'.format(rl_idx)]
                tp = rl['TimePoint {}'.format(tp_idx)]

                ch0 = tp['Channel {}'.format(ch0_idx)]
                ds0 = ch0['Data']
                hg0 = ch0['Histogram']
                #hg00 = ch0['Histogram1024']

                chn = tp.create_group(chn_name)
                dsn = chn.create_dataset('Data',
                                         shape=ds0.shape,
                                         dtype=datatype,
                                         chunks=ds0.chunks,
                                         compression=ds0.compression,
                                         )
                hgn = chn.create_dataset_like('Histogram', hg0)
                #hgn = chn.create_dataset_like('Histogram1024', hg00)

                for dim in 'XYZ':
                    isd = 'ImageSize{}'.format(dim)
                    chn.attrs[isd] = ch0.attrs[isd]

                # TODO?: self.ds =
        self.dims[3] += 1

    def bdv_create(self, comm=None):

        # FIXME
        pass

    def nii_create(self, comm=None):
        """Write a dataset to nifti format."""

        dtype = 'uint8' if self.dtype == 'bool' else self.dtype
        dtype = 'float' if self.dtype == 'float16' else dtype

#         try:
#             self.nii_load()  # FIXME
#         except:  # FileNotFoundError
        self.ds = np.empty(self.dims, dtype)
        self.file = nib.Nifti1Image(self.ds, self.get_transmat())

    def tif_create(self, comm=None):
        """Write a dataset to 3Dtif format."""

#         try:
#             self.tif_load()
#         except:
        self.file = None
        self.ds = np.empty(self.dims, self.dtype)

    def tifs_create(self, comm=None):
        """Write a dataset to a tif stack."""

#         try:
#             self.tifs_load()
#         except:
        self.file = None
        self.ds = []  # np.empty(self.dims, self.dtype)

    def dat_create(self, comm=None):
        """Write a dataset to dat format."""

        self.file = None
        self.ds = np.empty(self.dims, self.dtype)

    def get_transmat(self):
        """Return a transformation matrix with scaling of elsize."""

        mat = np.eye(4)
        if self.elsize is not None:
            mat[0][0] = self.elsize[0]
            mat[1][1] = self.elsize[1]
            mat[2][2] = self.elsize[2]

        return mat

    def write(self, data=None, slices=None):
        """Write a dataset."""

        if data is None:
            data = self.ds

        if slices is None:
            slices = self.slices

        formats = {'.h5': self.h5_write,
                   '.ims': self.ims_write,
                   '.bdv': self.bdv_write,
                   '.nii': self.nii_write,
                   '.tif': self.tif_write,
                   '.tifs': self.tifs_write,
                   '.dat': self.dat_write}

        formats[self.format](data, slices)

    def h5_write(self, data, slices):
        """Write data to a hdf5 dataset."""

        self.write_block(self.ds, data, slices)

    def bdv_write(self, data, slices):

        def slices2dsslices(start, step, shape):
            ds_step = step
            ds_start = int(start / step)
            ds_stop = ds_start + int(shape / ds_step)  # + ds_step
            ds_slice = slice(ds_start, ds_stop, 1)
            return ds_slice

        tp_names = ['t{:05d}'.format(tp_idx) for tp_idx in range(slices[self.axlab.index('t')].start, slices[self.axlab.index('t')].stop)]
        ch_names = ['s{:02d}'.format(ch_idx) for ch_idx in range(slices[self.axlab.index('c')].start, slices[self.axlab.index('c')].stop)]

        for tp_name in tp_names:
            for ch_name in ch_names:
                ch = self.file[tp_name][ch_name]
                for k, rl in ch.items():
                    rl_idx = int(k)
                    dsn = rl['cells']
                    ds_t = self.file[ch_name]['resolutions'][rl_idx, :][::-1].astype(int)
                    target_shape = list(self.slices2shape(slices))
                    slcs_out = [slices2dsslices(slc.start, step, shape) for slc, step, shape in zip(slices, ds_t, target_shape)]
                    ds_shape = list(self.slices2shape(slcs_out))
                    data_rl = data[::ds_t[0],::ds_t[1],::ds_t[2]]
                    data_rl = data_rl[:ds_shape[0], :ds_shape[1], :ds_shape[2]]
                    self.write_block(dsn, data_rl, slcs_out)


    def ims_write(self, data, slices):
        # data: rl0 unpadded block size (margins removed)
        # slices: rl0 block slices into full unpadded dataset

        def slices2dsslices(start, step, shape):
            ds_step = step
            ds_start = int(start / step)
            ds_stop = ds_start + int(shape / ds_step)  # + ds_step
            ds_slice = slice(ds_start, ds_stop, 1)
            return ds_slice

        def write_attribute(ch, name, val, formatstring='{:.3f}'):
            arr = [c for c in formatstring.format(val)]
            ch.attrs[name] = np.array(arr, dtype='|S1')

        chn_idx = slices[3].start
        chn_name = 'Channel {}'.format(chn_idx)

        ds = self.ds.parent
        nr = len(ds)

        # FIXME: timepoints not implemented
        for tp_idx in range(0, self.dims[4]):

            for rl_idx in range(0, nr):

                rl = ds['ResolutionLevel {}'.format(rl_idx)]
                tp = rl['TimePoint {}'.format(tp_idx)]

                chn = tp[chn_name]
                dsn = chn['Data']  # padded dataset

                # unpadded shape
                tags = ['ImageSizeZ', 'ImageSizeY', 'ImageSizeX']
                if rl_idx == 0:
                    ZYXt =[int(''.join([c.decode('utf-8') for c in chn.attrs[tag]])) for tag in tags]
                ZYX =[int(''.join([c.decode('utf-8') for c in chn.attrs[tag]])) for tag in tags]

                # downsample factors (wrt to full resolution)
                ds_t = [int(t / c) for t, c in zip(ZYXt, ZYX)]
                target_shape = list(self.slices2shape(slices))

                # downsample data
                  # TODO: downsample_blockwise?
                data_rl = data[::ds_t[0],::ds_t[1],::ds_t[2]]
                data_rl = data_rl[:ZYX[0], :ZYX[1], :ZYX[2]]

                # define output slices
                slcs_out = [slices2dsslices(slc.start, step, shape)
                            for slc, step, shape in zip(slices, ds_t, target_shape)]
                slcs_out = [slice(slc_out.start, slc_out.start + dr, 1)
                            for slc_out, dr in zip(slcs_out, data_rl.shape)]

                # write the block
                self.write_block(dsn, data_rl, slcs_out)

                # write histogram
                # FIXME!: histogram of full dataset on close only
                hgn = chn['Histogram']
                hist = np.histogram(data_rl, bins=hgn.shape[0])
                hgn[:] = hist[0]
                # write histogram attributes
                attributes = {
                    'HistogramMin': (np.amin(hist[1]), '{:.3f}'),
                    'HistogramMax': (np.amax(hist[1]), '{:.3f}'),
                }
                for k, v in attributes.items():
                    write_attribute(chn, k, v[0], v[1])

                #hgn1024 = chn['Histogram1024']
                #hist1024 = np.histogram(data_rl, bins=hgn1024.shape[0])
                #hgn1024[:] = hist1024[0]
                # write histogram attributes
                #attributes = {
                #    'HistogramMin1024': (np.amin(hist1024[1]), '{:.3f}'),
                #    'HistogramMax1024': (np.amax(hist1024[1]), '{:.3f}'),
                #}
                #for k, v in attributes.items():
                #    write_attribute(chn, k, v[0], v[1])

    def nii_write(self, data, slices):
        """Write data to a nifti dataset."""

        if data.dtype == 'bool':
            data = data.astype('uint8')

#         self.ds = self.file.get_fdata()  # loads as floats
#         self.ds = np.asanyarray(self.file.dataobj).astype(self.dtype)  # unsafe cast
        self.ds = np.asanyarray(self.file.dataobj)
        self.write_block(self.ds, data, slices)
        nib.Nifti1Image(self.ds, self.get_transmat()).to_filename(self.path)

    def nii_write_mat(self, data, slices, mat):
        """Write data to a nifti dataset."""

        if data.dtype == 'bool':
            data = data.astype('uint8')

#         self.ds = self.file.get_fdata()  # loads as floats
#         self.ds = np.asanyarray(self.file.dataobj).astype(self.dtype)  # unsafe cast
        self.ds = np.asanyarray(self.file.dataobj)
        self.write_block(self.ds, data, slices)
        nib.Nifti1Image(self.ds, mat).to_filename(self.path)

    def tif_write(self, data, slices):
        """Write data to a 3Dtif dataset."""

        if data is not None:
            self.write_block(self.ds, data, slices)

        if self.ds.dtype == 'bool':
            dtype = 'uint8'
        else:
            dtype = self.dtype

        imsave(self.path, self.ds.astype(dtype, copy=False))

    def tifs_write(self, data, slices):
        """Write data to a tif stack."""

        # TODO: 4D data?

#         nzfills = 5
#         ext = '.tif'
        nzfills = 4
        ext = '.png'
        slicedim = self.axlab.index('z')
        slcoffset = slices[slicedim].start

        self.mkdir_p()
#         ch = slices[3].start
#         tile=0
#         chstring = '_ch{:02d}_tile{:04d}'.format(ch, tile)
        fstring = '{{:0{0}d}}'.format(nzfills)

        if data.dtype == 'bool':
            data = data.astype('uint8')

        if ext != '.tif':
            data = self.normalize_data(data)[0]

        if data.ndim == 2:
            slcno = slcoffset
            filepath = os.path.join(self.path, fstring.format(slcno) + ext)
            imsave(filepath, data)
            return

        for slc in range(0, data.shape[slicedim]):  # TODO: slices
            slcno = slc + slcoffset
            if slicedim == 0:
                slcdata = data[slc, :, :]
            elif slicedim == 1:
                slcdata = data[:, slc, :]
            elif slicedim == 2:
                slcdata = data[:, :, slc]

            filepath = os.path.join(self.path, fstring.format(slcno) + ext)
#             filepath = os.path.join(self.path, fstring.format(slcno) + chstring + ext)
            imsave(filepath, slcdata)

    def dat_write(self, data, slices):
        """Write data to a dataset."""

        self.write_block(self.ds, data, slices)

    def write_block(self, ds, data, slices):
        """Write a block of data into a dataset."""

        if slices is None:
            slices = self.slices

        ndim = self.get_ndim()
        if self.format in ['.ims', '.bdv']:
            ndim = 3

        # ds[tuple(slices)] = data
        if ndim == 1:
            ds[slices[0]] = data
        elif ndim == 2:
            ds[slices[0], slices[1]] = data
        elif ndim == 3:
            ds[slices[0], slices[1], slices[2]] = data
        elif ndim == 4:
            ds[slices[0], slices[1], slices[2], slices[3]] = data
        elif ndim == 5:
            ds[slices[0], slices[1], slices[2], slices[3], slices[4]] = data

        return ds

    def close(self):
        """Close a file."""

        try:
            if isinstance(self.file, h5py.File):
                self.file.close()
            if isinstance(self.file, bf.ImageReader):
                self.file.close()
        except:
            pass

    def get_metadata(self, files, datatype, outlayout, elsize):
        """Get metadata from a dm3 file."""

        # derive the stop-values from the image data if not specified
        if files[0].endswith('.dm3'):
            yxdims, alt_dtype, elsize = self.dm3_get_metadata(files, outlayout, elsize)
        else:
            yxdims = imread(files[0]).shape
            alt_dtype = imread(files[0]).dtype

        zyxdims = [len(files)] + list(yxdims)

        datatype = datatype or alt_dtype

        element_size_um = [0 if el is None else el for el in elsize]

        return zyxdims, datatype, element_size_um

    def dm3_get_metadata(self, files, outlayout, elsize):
        """Get metadata from a dm3 file."""

        try:
            import DM3lib as dm3
        except ImportError:
            raise

        dm3f = dm3.DM3(files[0], debug=0)

#         yxdims = dm3f.imagedata.shape
        alt_dtype = dm3f.imagedata.dtype
#         yxelsize = dm3f.pxsize[0]

        id = 'root.ImageList.1.ImageData'
        tag = '{}.Dimensions.{:d}'
        yxdims = []
        for dim in [0, 1]:
            yxdims += [int(dm3f.tags.get(tag.format(id, dim)))]

        tag = '{}.Calibrations.Dimension.{:d}.Scale'
        for lab, dim in zip('xy', [0, 1]):
            if elsize[outlayout.index(lab)] == -1:
                pxsize = float(dm3f.tags.get(tag.format(id, dim)))
                elsize[outlayout.index(lab)] = pxsize
        tag = 'root.ImageList.1.ImageTags.SBFSEM.Record.Slice thickness'
        slicethickness = float(dm3f.tags.get(tag.format(id, dim)))
        elsize[outlayout.index('z')] = slicethickness / 1000

        return yxdims, alt_dtype, elsize

    def dm3_get_yxdims(self):
        """Get the element sizes from a dm3 dataset."""

        dm3f = dm3.DM3(self.file[0], debug=0)

        id = 'root.ImageList.1.ImageData'
        tag = '{}.Dimensions.{:d}'
        yxdims = []
        for dim in [0, 1]:
            yxdims += [int(dm3f.tags.get(tag.format(id, dim)))]

        return yxdims

    def tif_get_yxdims(self):
        """Get the dimensions of a tif image."""

        if self.file:
            im = imread(self.file[0])
        else:
            im = imread(self.path)

        return list(im.shape)

    def tif_load_dtype(self):
        """Get the datatype of a tif image."""

        if self.file:
            im = imread(self.file[0])
        else:
            im = imread(self.path)

        return im.dtype

    def dm3_load_dtype(self):
        """Get the datatype from a dm3 file."""

        dataTypes = {
            0:  'NULL_DATA',
            1:  'SIGNED_INT16_DATA',
            2:  'REAL4_DATA',
            3:  'COMPLEX8_DATA',
            4:  'OBSELETE_DATA',
            5:  'PACKED_DATA',
            6:  'UNSIGNED_INT8_DATA',
            7:  'SIGNED_INT32_DATA',
            8:  'RGB_DATA',
            9:  'SIGNED_INT8_DATA',
            10: 'UNSIGNED_INT16_DATA',
            11: 'UNSIGNED_INT32_DATA',
            12: 'REAL8_DATA',
            13: 'COMPLEX16_DATA',
            14: 'BINARY_DATA',
            15: 'RGB_UINT8_0_DATA',
            16: 'RGB_UINT8_1_DATA',
            17: 'RGB_UINT16_DATA',
            18: 'RGB_FLOAT32_DATA',
            19: 'RGB_FLOAT64_DATA',
            20: 'RGBA_UINT8_0_DATA',
            21: 'RGBA_UINT8_1_DATA',
            22: 'RGBA_UINT8_2_DATA',
            23: 'RGBA_UINT8_3_DATA',
            24: 'RGBA_UINT16_DATA',
            25: 'RGBA_FLOAT32_DATA',
            26: 'RGBA_FLOAT64_DATA',
            27: 'POINT2_SINT16_0_DATA',
            28: 'POINT2_SINT16_1_DATA',
            29: 'POINT2_SINT32_0_DATA',
            30: 'POINT2_FLOAT32_0_DATA',
            31: 'RECT_SINT16_1_DATA',
            32: 'RECT_SINT32_1_DATA',
            33: 'RECT_FLOAT32_1_DATA',
            34: 'RECT_FLOAT32_0_DATA',
            35: 'SIGNED_INT64_DATA',
            36: 'UNSIGNED_INT64_DATA',
            37: 'LAST_DATA',
            }

        # PIL "raw" decoder modes for the various image dataTypes
        dataTypesDec = {
            1:  'F;16S',     #16-bit LE signed integer
            2:  'F;32F',     #32-bit LE floating point
            6:  'F;8',       #8-bit unsigned integer
            7:  'F;32S',     #32-bit LE signed integer
            9:  'F;8S',      #8-bit signed integer
            10: 'F;16',      #16-bit LE unsigned integer
            11: 'F;32',      #32-bit LE unsigned integer
            14: 'F;8',       #binary
            }

        dtypes = {
            1:  'int16',     #16-bit LE signed integer
            2:  'float32',   #32-bit LE floating point
            6:  'uint8',     #8-bit unsigned integer
            7:  'int32',     #32-bit LE signed integer
            9:  'int8',      #8-bit signed integer
            10: 'uint16',    #16-bit LE unsigned integer
            11: 'uint32',    #32-bit LE unsigned integer
            }

        dm3f = dm3.DM3(self.file[0], debug=0)

        id = 'root.ImageList.1.ImageData'
        tag = '{}.DataType'
        datatype = dtypes[int(dm3f.tags.get(tag.format(id)))]

        return datatype

    def pbf_load_dtype(self):
        """Get the datatype from a czi file."""

        md = bf.OMEXML(bf.get_omexml_metadata(self.path))
        pixels = md.image(self.series).Pixels
        datatype = pixels.PixelType

        return datatype

#     def get_image(self, index):
#
#         fpath = self.file[index]
#
#         if fpath.endswith('.dm3'):
#             dm3f = dm3.DM3(fpath, debug=0)
#             im = dm3f.imagedata
#         else:
#             im = imread(fpath)
#
#         return im

    def get_image(self, fpath):
        """Load dm3 or tif image data."""

        if fpath.endswith('.dm3'):
            dm3f = dm3.DM3(fpath, debug=0)
            im = dm3f.imagedata
        else:
            im = imread(fpath)

        return im

    def transpose(self, outlayout):
        """Transpose the image."""

        in2out = [self.axlab.index(l) for l in outlayout]

        self.axlab = outlayout
        self.elsize = np.array(self.elsize)[in2out]
        self.ds[:] = np.transpose(self.ds[:], in2out)  # FIXME: fail for proxies
        self.slices = [self.slices[i] for i in in2out]

    def squeeze(self, dims):

        # TODO: squeeze data?
        squeezable = ['shape', 'dims', 'elsize', 'chunks', 'slices', 'axlab']
        # print(self.axlab, dims)
        for dim in dims:
            if dim not in self.axlab:
                continue
            for attr in squeezable:
                attr_val = getattr(self, attr)
                if attr_val is not None:
                    attr_list = list(attr_val)
                    try:
                        del attr_list[self.axlab.index(dim)]
                    except:
                        pass
                    setattr(self, attr, attr_list)
                    # setattr(self, attr, tuple(attr_list))

    def mkdir_p(self):
        try:
            os.makedirs(self.path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(self.path):
                pass
            else:
                raise

    def get_props2(self):
        # TODO: replace get_props throughout
        props = {k:v for k, v in vars(self).items()}
        props['path'] = ''
        props.pop('file')
        props.pop('filereader')
        props.pop('ds')
        props.pop('format')
        return props

    def downsampled(self, downsample_factors, ismask=False, outputpath=''):

        props = self.get_props2()
        props['path'] = outputpath
        props['permission'] = 'r+'
        mo = Image(**props)

        # downsample
        dsfac = tuple([downsample_factors[dim] for dim in self.axlab])
        ds = self.slice_dataset()
        if ismask:
            data = block_reduce(ds, dsfac, np.max)
        else:
            data = downscale_local_mean(ds, dsfac).astype('float32')

        mo.shape = data.shape
        mo.dims = data.shape
        mo.elsize = [es * ds for es, ds in zip(self.elsize, dsfac)]
        mo.slices = None
        mo.dtype = data.dtype

        mo.create()
        mo.write(data)

        return mo

    def extract_channel(self, ch=0, tp=0, outputpath=''):

        slcs = [slc for slc in self.slices]

        props = self.get_props2()
        props['path'] = outputpath
        props['permission'] = 'r+'
        mo = Image(**props)

        if 't' in self.axlab:
            self.slices[self.axlab.index('t')] = slice(tp, tp + 1, 1)
            mo.squeeze('t')

        if 'c' in self.axlab:
            self.slices[self.axlab.index('c')] = slice(ch, ch + 1, 1)
            mo.squeeze('c')

        mo.create()
        mo.write(self.slice_dataset())

        self.slices = slcs

        return mo

    def find_downsample_factors(self, rl0_idx=0, rl1_idx=-1):
        """Find downsample factors."""

        def att2str(att):
            return ''.join([t.decode('utf-8') for t in att])

        def find_dims(im, idx):
            rl = self.file['/DataSet/ResolutionLevel {}'.format(idx)]
            im_info = rl['TimePoint 0/Channel 0']
            return [int(att2str(im_info.attrs['ImageSize{}'.format(dim)]))
                    for dim in 'ZYX']

        if rl1_idx == -1:
            rl1_idx = self.reslev

        if self.format == '.ims':
            dims0 = find_dims(self, rl0_idx)
            dims1 = find_dims(self, rl1_idx)
            dsfacs = np.around(np.array(dims0) / np.array(dims1)).astype('int')
        elif self.format == '.bdv':
            dsfacs = self.file['s00/resolutions'][rl1_idx, :][::-1]
        else:  # FIXME
            dsfacs = [1] * len(self.dims)

        return dsfacs


class MaskImage(Image):

    def __init__(self, path,
                 **kwargs):

        super(MaskImage, self).__init__(path, **kwargs)

    def invert(self):  # __invert__(self)

        self.ds[:] = ~self.ds[:]


class LabelImage(Image):

    def __init__(self, path,
                 maxlabel=0, ulabels=[],
                 **kwargs):

        super(LabelImage, self).__init__(path, **kwargs)
        self.maxlabel = maxlabel
        self.ulabels = np.array(ulabels)

    def load(self, comm=None, load_data=True):
        """Load a dataset."""

        super(LabelImage, self).load(comm, load_data)
        if load_data:  # TODO: look at downstream
            self.set_maxlabel()


    def read_labelsets(self, lsfile):
        """Read labelsets from file."""

        e = os.path.splitext(lsfile)[1]
        if e == '.pickle':
            with open(lsfile, 'rb') as f:
                labelsets = pickle.load(f)
        else:
            labelsets = self.read_labelsets_from_txt(lsfile)

        return labelsets

    def read_labelsets_from_txt(self, lsfile):
        """Read labelsets from a textfile."""
        labelsets = {}

        with open(lsfile) as f:
            lines = f.readlines()
            for line in lines:
                splitline = line.split(':', 2)
                lsk = int(splitline[0])
                lsv = set(np.fromstring(splitline[1], dtype=int, sep=' '))
                labelsets[lsk] = lsv

        return labelsets

    def write_labelsets(self, labelsets, filetypes=['pickle']):
        """Write labelsets to file."""

        filestem = self.split_path()['base']

        if 'txt' in filetypes:
            filepath = filestem + '.txt'
            self.write_labelsets_to_txt(labelsets, filepath)
        if 'pickle' in filetypes:
            filepath = filestem + '.pickle'
            with open(filepath, "wb") as f:
                pickle.dump(labelsets, f)

    def write_labelsets_to_txt(self, labelsets, filepath):
        """Write labelsets to a textfile."""

        with open(filepath, "wb") as f:
            for lsk, lsv in labelsets.items():
                f.write("%8d: " % lsk)
                ls = sorted(list(lsv))
                for l in ls:
                    f.write("%8d " % l)
                f.write('\n')

    def set_ulabels(self):

        self.ulabels = np.unique(self.ds)

    def set_maxlabel(self):

        if not self.ulabels.any():
            self.set_ulabels()
        self.maxlabel = int(np.amax(self.ulabels))
#         print("{} labels (max: {}) in volume {}".format(len(self.ulabels) - 1,
#                                                         self.maxlabel,
#                                                         self.path))

    def get_fwmap(self, empty=False):

        if empty:
            fw = np.zeros(self.maxlabel + 1, dtype='i')
        else:
            fw = [l if l in self.ulabels else 0
                  for l in range(0, self.maxlabel + 1)]

        return fw

    def forward_map(self, fw=[], labelsets={}, delete_labelsets=False,
                    from_empty=False, ds=[]):
        """Map all labels in value to key."""

        fw = fw or self.get_fwmap(empty=from_empty)

        for lsk, lsv in labelsets.items():
            lsv = sorted(list(lsv))
            for l in lsv:
                if delete_labelsets:
                    fw[l] = 0
                else:
                    fw[l] = lsk

        fw[0] = 0
        if isinstance(ds, np.ndarray):
            fwmapped = np.array(fw)[ds]
        else:
            fwmapped = np.array(fw)[self.ds]

        return fwmapped


class wmeMPI(object):

    def __init__(self, usempi, mpi_dtype=''):

        if usempi:
            if not mpi_dtype:
                self.mpi_dtype = MPI.SIGNED_LONG_LONG
            else:
                self.mpi_dtype = mpi_dtype
            self.comm = MPI.COMM_WORLD
            self.enabled = True
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.mpi_dtype = None
            self.comm = None
            self.enabled = False
            self.rank = 0
            self.size = 1

        self.nblocks = 1
        self.blocks = []
        self.series = []
        self.n_local = ()
        self.displacements = []

    def scatter_series(self, randomize=False):
        """Scatter a series of jobnrs over processes."""

        series = np.array(range(0, self.nblocks), dtype=int)
        if randomize:
            random.shuffle(series)

        if not self.enabled:
            self.series = series
            self.n_local = None
            self.diplacements = None
            return

        n_all = len(series)
        n_local = np.ones(self.size, dtype=int) * n_all / self.size
        n_local[0:n_all % self.size] += 1
        self.n_local = tuple(n_local)

        series = np.array(series, dtype=int)
        self.series = np.zeros(n_local[self.rank], dtype=int)

        self.displacements = tuple(sum(n_local[0:r])
                                   for r in range(0, self.size))

        self.comm.Scatterv([series,
                            self.n_local,
                            self.displacements,
                            self.mpi_dtype],
                           self.series, root=0)

    def set_blocks(self, im, blocksize, margin=[], blockrange=[], path_tpl='', imslices=[]):
        """Create a list of dictionaries with data block info.

        TODO: step?
        """

        imslices = imslices or im.slices

        shape = list((len(range(*slc.indices(slc.stop))) for slc in imslices))

        if not blocksize:
            blocksize = [dim for dim in shape]
        if not margin:
            margin = [0 for dim in shape]

        blocksize = [dim if bs == 0 else bs
                     for bs, dim in zip(blocksize, shape)]

        starts, stops, = {}, {}
        for i, dim in enumerate(im.axlab):
            starts[dim], stops[dim] = self.get_blockbounds(imslices[i].start,
                                                           shape[i],
                                                           blocksize[i],
                                                           margin[i])

        ndim = len(im.axlab)
        starts = tuple(starts[dim] for dim in im.axlab)
        stops = tuple(stops[dim] for dim in im.axlab)
        startsgrid = np.array(np.meshgrid(*starts))
        stopsgrid = np.array(np.meshgrid(*stops))
        starts = np.transpose(np.reshape(startsgrid, [ndim, -1]))
        stops = np.transpose(np.reshape(stopsgrid, [ndim, -1]))

        idstring = '{:05d}-{:05d}_{:05d}-{:05d}_{:05d}-{:05d}'
        for start, stop in zip(starts, stops):

            block = {}
            block['slices'] = [slice(sta, sto) for sta, sto in zip(start, stop)]

            x = block['slices'][im.axlab.index('x')]
            y = block['slices'][im.axlab.index('y')]
            z = block['slices'][im.axlab.index('z')]
            block['id'] = idstring.format(x.start, x.stop,
                                          y.start, y.stop,
                                          z.start, z.stop)
            block['path'] = path_tpl.format(block['id'])

            self.blocks.append(block)

        if blockrange:
            self.blocks = self.blocks[blockrange[0]:blockrange[1]]

        self.nblocks = len(self.blocks)

    def get_blockbounds(self, offset, shape, blocksize, margin):
        """Get the block range for a dimension."""

        # blocks
        starts = range(offset, shape + offset, blocksize)
        stops = np.array(starts) + blocksize

        # blocks with margin
        starts = np.array(starts) - margin
        stops = np.array(stops) + margin

        # blocks with margin reduced on boundary blocks
        starts[starts < offset] = offset
        stops[stops > shape + offset] = shape + offset

        return starts, stops


def get_image(image_in, imtype='', **kwargs):

    comm = kwargs.pop('comm', None)
    load_data = kwargs.pop('load_data', True)

    if isinstance(image_in, Image):
        im = image_in
        if 'slices' in kwargs.keys():
            im.slices = kwargs['slices']
        if im.format == '.h5':
            im.h5_load(comm, load_data)
    else:
        if imtype == 'Label':
            im = LabelImage(image_in, **kwargs)
        elif imtype == 'Mask':
            im = MaskImage(image_in, **kwargs)
        else:
            im = Image(image_in, **kwargs)

        im.load(comm, load_data)

    return im


def parse_xml_metadata(xml_string, array_order='tzyxc'):
    """Get interesting metadata from the LIF file XML string.
    Parameters
    ----------
    xml_string : string
        The string containing the XML data.
    array_order : string
        The order of the dimensions in the multidimensional array.
        Valid orders are a permutation of "tzyxc" for time, the three
        spatial dimensions, and channels.
    Returns
    -------
    names : list of string
        The name of each image series.
    sizes : list of tuple of int
        The pixel size in the specified order of each series.
    resolutions : list of tuple of float
        The resolution of each series in the order given by
        `array_order`. Time and channel dimensions are ignored.
    """

    names, sizes, resolutions, dimorders, datatypes = [], [], [], [], []

    datatype_tag = 'Type'
    axlab_tag = 'DimensionOrder'

    metadata_root = et.ElementTree.fromstring(xml_string)

    for child in metadata_root:
        if child.tag.endswith('Image'):
            names.append(child.attrib['Name'])
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    axlab = att[axlab_tag].lower()
                    dimorders.append(axlab)
                    sizes.append(tuple([int(att['Size' + c]) for c in axlab.upper()]))
                    resolutions.append(tuple([float(att['PhysicalSize' + c]) if c in 'XYZ' else 1 for c in axlab.upper()]))
                    datatypes.append(att[datatype_tag])

    return names, sizes, resolutions, dimorders, datatypes


def start(max_heap_size='8G'):
    """Start the Java Virtual Machine, enabling bioformats IO.
    Parameters
    ----------
    max_heap_size : string, optional
        The maximum memory usage by the virtual machine. Valid strings
        include '256M', '64k', and '2G'. Expect to need a lot.
    """
    jv.start_vm(class_path=bf.JARS, max_heap_size=max_heap_size)
    global VM_STARTED
    VM_STARTED = True


def done():
    """Kill the JVM. Once killed, it cannot be restarted.
    Notes
    -----
    See the python-javabridge documentation for more information.
    """
    jv.kill_vm()
    global VM_KILLED
    VM_KILLED = True


def split_filename(filename, blockoffset=[0, 0, 0]):
    """Extract the data indices from the filename."""

    import re

    datadir, tail = os.path.split(filename)
    fname = os.path.splitext(tail)[0]
    parts = re.findall('([0-9]{5}-[0-9]{5})', fname)
    id_string = '_'.join(parts)
    dset_name = fname.split(id_string)[0][:-1]

    x = int(parts[-3].split("-")[0]) - blockoffset[0]
    X = int(parts[-3].split("-")[1]) - blockoffset[0]
    y = int(parts[-2].split("-")[0]) - blockoffset[1]
    Y = int(parts[-2].split("-")[1]) - blockoffset[1]
    z = int(parts[-1].split("-")[0]) - blockoffset[2]
    Z = int(parts[-1].split("-")[1]) - blockoffset[2]

    dset_info = {'datadir': datadir, 'base': dset_name,
                 'nzfills': len(parts[1].split("-")[0]),
                 'postfix': id_string,
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}

    return dset_info, x, X, y, Y, z, Z


def get_n_workers(n_workers=0, params={}):
    """Determine the number of workers."""

    cpu_count = multiprocessing.cpu_count()

    n_workers = n_workers or cpu_count

    try:
        if params['n_workers'] > 0:
            n_workers = params['n_workers']
    except (KeyError, TypeError):
        pass

    n_workers = min(n_workers, cpu_count)

    return n_workers


def get_blockfiles(image_in, block_dir, block_selection=[], block_postfix='.h5'):
    """Return a list of filepaths (with indices) in the blockdirectory."""

    paths = get_paths(image_in)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)
    filepat = '{}_*{}'.format(dataset, block_postfix)

    filepaths = glob.glob(os.path.join(block_dir, filepat))
    filepaths.sort()

    if block_selection:
        filepaths = [filepaths[i] for i in block_selection]

    return filepaths, list(range(len(filepaths)))


def get_blocksize(image_in, bs=640):
    """Load the matrix size from file, replacing xy-blocksize."""

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    blocksize = list(im.dims)
    blocksize[im.axlab.index('x')] = bs
    blocksize[im.axlab.index('y')] = bs

    return blocksize


def get_blockmargin(image_in, bm=64):
    """Return a 0-list (len=ndim) with blockmargins inserted for xy."""

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    blockmargin = [0] * len(im.dims)
    blockmargin[im.axlab.index('x')] = bm
    blockmargin[im.axlab.index('y')] = bm

    return blockmargin


def get_blockinfo(image_in, parameter_file,
                  params=dict(blocksize=[], blockmargin=[], blockrange=[])):
    """Find the blocksize, blockmargin and block indices."""

    if not params['blocksize']:
        ds_par = get_params(dict(), parameter_file, 'dataset')
        try:
            bs = ds_par['blocksize_xy']
        except:
            bs = 640
        params['blocksize'] = get_blocksize(image_in, bs)

    if not params['blockmargin']:
        ds_par = get_params(dict(), parameter_file, 'dataset')
        try:
            bm = ds_par['blockmargin_xy']
        except:
            bm = 64
        params['blockmargin'] = get_blockmargin(image_in, bm)

    n_blocks = get_n_blocks(image_in, params['blocksize'], params['blockmargin'])

    if 'blockrange' in params.keys():
        if params['blockrange']:
            params['blocks'] = list(range(params['blockrange'][0], params['blockrange'][1]))
            return params['blocksize'], params['blockmargin'], params['blocks']

    if 'blocks' in params.keys():
        if params['blocks']:
            return params['blocksize'], params['blockmargin'], params['blocks']

    params['blocks'] = list(range(n_blocks))

    return params['blocksize'], params['blockmargin'], params['blocks']


def get_n_blocks(image_in, blocksize, blockmargin):
    """Return the number of blocks in the dataset."""

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    mpi = wmeMPI(usempi=False)
    mpi.set_blocks(im, blocksize, blockmargin)
    im.close()

    return len(mpi.blocks)


def prep_pars(**kwargs):
    """Retunr general configuration, specific parameters and outputdirectory."""

    cfg = get_config(kwargs['parameter_file'])
    kwargs['directory'] = get_outdir(kwargs['image_in'], kwargs['step_id'], kwargs['outputdir'])
    pars = get_pars(kwargs, cfg, kwargs['step_id'], kwargs['step'])

    return cfg, pars, kwargs['directory']


def get_pars(params, cfg, pfile_entry, sub_entry='estimate'):
    """Merge parameters from arguments and parameterfile(=leading)."""

    file_params = {}

    try:
        cfg_mod = cfg[pfile_entry]
    except KeyError:
        print('Parameters for {} not found in parameterfile'.format(pfile_entry))
        print('Continuing with all default values')
        return params
    else:

        if sub_entry in cfg_mod.keys():

            cfg_step = cfg_mod[sub_entry]

            if cfg_step is not None:

                for k, v in cfg_step.items():

                    if v is not None:

                        file_params.update(v)

        else:

            file_params = cfg_mod

    if file_params:
        params.update(file_params)

    if 'postfix' in cfg_mod.keys():
        params['postfix'] = cfg_mod['postfix']

    return params


def format_(elements, delimiter='_'):
    return delimiter.join([x for x in elements if x])


def get_prevpath(input, image_in, dataset, step_id, step, formatstring, fallback=''):

    if isinstance(input, bool):
        if not input:
            return fallback
        inputstem = get_inputstem(image_in, dataset, step_id, step)
        return formatstring.format(inputstem)
    else:
        if os.path.exists(get_paths(input)['file']):
            return input


def get_inputstem(image_in, dataset, step_id, step):
    """Derive the inputstem from the usual previous step."""

    cfg_step = get_config_step(image_in, dataset, step_id, step)
    basename = format_([cfg_step['files']['dataset'], cfg_step['files']['suffix']])
    inputstem = os.path.join(cfg_step['files']['directory'], basename)

    return inputstem


def dump_pars(step_id, step, outputdir, pars, parsets):
    """Write parameters to yaml file."""

    d = {step_id: {step: {
        'files':  {s: pars[s] for s in parsets['fpar']},
        'params': {s: pars[s] for s in parsets['ppar']},
        'submit': {s: pars[s] for s in parsets['spar']},
        }}}

    # TODO: convert any numpy dtypes here.
    # basename = format_([pars['dataset'], pars['suffix'], step_id, step])
    # basename = format_([step_id, step])
    basename = format_([pars['dataset'], step_id, step])
    outstem = os.path.join(pars['directory'], basename)
    with open('{}.yml'.format(outstem), 'w') as f:
        yaml.dump(d, f, default_flow_style=False)


def get_params(params, parameter_file, pfile_entry, sub_entry='params'):
    """Merge parameters from arguments and parameterfile(=leading)."""

    file_params = {}
    if parameter_file:
        with open(parameter_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            if sub_entry in cfg[pfile_entry].keys():
                file_params = cfg[pfile_entry][sub_entry]
            else:
                file_params = cfg[pfile_entry]

    if file_params:
        params.update(file_params)

    return params


def get_config(parameter_file):
    """Work out the output (sub)directory."""

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


def get_config_step(image_in, dataset, step_id, step):
    """Get saved configuration of a processing step."""

    inputdir = get_outdir(image_in, step_id)
    basename = format_([dataset, step_id, step])
    ymlfile = os.path.join(inputdir, '{}.yml'.format(basename))
    cfg = get_config(ymlfile)

    return cfg[step_id][step]


def get_outdir(image_in, step_id, outputdir=''):
    """Work out the output (sub)directory.

    1. Use supplied outputdir.
    2. Use <imagedir>/<parfile_key>.
    """

    if not outputdir:
        paths = get_paths(image_in)
        datadir, filename = os.path.split(paths['base'])
        outputdir = os.path.join(datadir, step_id)

    os.makedirs(outputdir, exist_ok=True)

    return outputdir


def get_outputdir(image_in, parameter_file, outputdir, step_id='', fallback=''):
    """Work out the output (sub)directory.

    1. Use supplied outputdir.
    2. Use <imagedir>/<step_id> (as parameterfile 'dirtree:datadir:<step_id>').
    3. Use <imagedir>/<fallback>.
    4. Use <imagedir>/<parfile_key>.
    """

    dirs = get_params(dict(), parameter_file, 'dirtree')
    try:
        subdir = dirs['datadir'][step_id] or ''
    except KeyError:
        subdir = fallback or step_id  # TODO: check backward compat

    if not outputdir:
        paths = get_paths(image_in)
        datadir, filename = os.path.split(paths['base'])
        outputdir = os.path.join(datadir, subdir)

    os.makedirs(outputdir, exist_ok=True)

    return outputdir


def get_paths(image_in, resolution_level=-1):
    """Get split path from inputfile."""

    if resolution_level != -1:  # we should have an Imaris pyramid
        image_in = '{}/DataSet/ResolutionLevel {}'.format(image_in, resolution_level)
    im = Image(image_in, permission='r')
    paths = im.split_path()
    im.close()

    return paths


def get_ims_ref_path(image_in, parameter_file, ims_ref_path=''):
    """Get split path from inputfile."""

    paths = get_paths(image_in)

    if not ims_ref_path:

        with open(parameter_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        filename = '{}{}{}{}.ims'.format(
            cfg['dataset']['name'],
            cfg['shading']['params']['postfix'],
            cfg['stitching']['params']['postfix'],
            cfg['dataset']['ims_ref_postfix'],
            )

        ims_ref_path = os.path.join(paths['dir'], filename)

    return ims_ref_path


def get_imageprops(image_in):
    """Return a dict with image attributes."""

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    props = im.get_props()
    im.close()

    return props


def transpose_props(props, outlayout=''):
    """Transpose the attributes of an image."""

    if not outlayout:
        outlayout = props['axlab'][::-1]
    in2out = [props['axlab'].index(l) for l in outlayout]
    props['elsize'] = np.array(props['elsize'])[in2out]
    props['shape'] = np.array(props['shape'])[in2out]
    props['axlab'] = ''.join([props['axlab'][i] for i in in2out])
    if 'chunks' in props.keys():
        if props['chunks'] is not None:
            props['chunks'] = np.array(props['chunks'])[in2out]
    if 'slices' in props.keys():
        if props['slices'] is not None:
            props['slices'] = [props['slices'][i] for i in in2out]

    return props


def h5_nii_convert(image_in, image_out, datatype='', minmax=False):
    """Convert between h5 (zyx) and nii (xyz) file formats."""

    im_in = Image(image_in)
    im_in.load(load_data=False)
    data = im_in.slice_dataset()

    props = transpose_props(im_in.get_props())

    if minmax:
        from skimage import exposure
        data = exposure.rescale_intensity(data)

    if datatype == 'uint8':  # for ACME
        from skimage import img_as_ubyte
        data = img_as_ubyte(data)
        props['dtype'] = datatype

    elif datatype:
        # FIXME: skimage.util.dtype.convert will be deprecated
        from skimage.util.dtype import convert
        data = convert(data, np.dtype(datatype), force_copy=False)
        props['dtype'] = datatype

    im_out = Image(image_out, **props)
    im_out.create()
    im_out.write(data.transpose().astype(props['dtype']))
    im_in.close()
    im_out.close()


def parse_args(step_id, fun_selector, *argv):
    """Parse arguments common to all modules."""

    if isinstance(fun_selector, dict):
        steps = list(fun_selector.keys())
    elif isinstance(fun_selector, list):
        steps = fun_selector

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-i', '--image_in',
        required=True,
        help='path to raw image file',
        )
    parser.add_argument(
        '-p', '--parameter_file',
        # required=True,
        help='path to yaml parameter file',
        )
    parser.add_argument(
        '-s', '--steps',
        default='all',
        nargs='+',
        choices=steps,
        required=False,
        help='processing steps to execute',
        )
    parser.add_argument(
        '-S', '--step_id',
        default=step_id,
        required=False,
        help='top level name of the module parameters in parameter file',
        )
    parser.add_argument(
        '-o', '--outputdir',
        required=False,
        help='path to output directory',
        )
    parser.add_argument(
        '-x', '--prefix',
        required=False,
        help='name of the dataset prepended to each file',
        )
    # parser.add_argument(
    #     '-x', '--suffix',
    #     required=False,
    #     help='path to output directory',
    #     )
    parser.add_argument(
        '-n', '--max_workers',
        type=int,
        default=0,
        required=False,
        help='maximal number of concurrent workers (0: automatic)',
        )

    args = parser.parse_args()

    if args.steps is None or args.steps == 'all':
        args.steps = steps

    return args


# def parse_args_common(step_ids, fun_selector, *argv):
#     """Parse arguments common to all modules."""
#
#     steps = list(fun_selector.keys())
#
#     parser = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#         )
#     parser.add_argument(
#         '-i', '--image_in',
#         required=True,
#         help='path to image file',
#         )
#     parser.add_argument(
#         '-p', '--parameter_file',
#         required=True,
#         help='path to yaml parameter file',
#         )
#     parser.add_argument(
#         '-s', '--steps',
#         default='all',
#         nargs='+',
#         choices=steps,
#         required=False,
#         help='processing steps to execute',
#         )
#     parser.add_argument(
#         '-S', '--step_ids',
#         default=step_ids,
#         nargs='+',
#         required=False,
#         help='top level name(s) of the processing step(s) in parameter file',
#         )
#     parser.add_argument(
#         '-o', '--outputdir',
#         required=False,
#         help='path to output directory',
#         )
#     parser.add_argument(
#         '-n', '--n_workers',
#         type=int,
#         default=0,
#         required=False,
#         help='number of workers (0: automatic)',
#         )
#
#     args = parser.parse_args()
#
#     if args.steps is None or args.steps == 'all':
#         mapper = dict(zip(steps, args.step_ids))
#     else:
#         # FIXME: this is dodgy for when the wrong number of args is supplied
#         if len(args.step_ids) != len(args.steps):
#             idxs = [steps.index(step) for step in args.steps]
#             step_ids = [step_ids[i] for i in idxs]
#         mapper = dict(zip(args.steps, step_ids))
#
#     return args, mapper
#

class Stapl3r(object):
    """Base class for STAPL3D framework.

    Parameters
    ----------
    image_in : string
        Path to dataset.
    parameter_file : string
        Path to yaml parameter file.
    module_id : string
        Name of the STAPL3D module.
    step_id: string
        Identifier of the yaml parameterfile entry.
    directory : string
        Name of output subdirectory.
    prefix : string
        Output prefix.
    datadir : string
        Override for datadir [when image_in used to define alternate input].
    max_workers : int
        Maximal number of cores to use for processing.
    verbosity : int
        Verbosity level.

    Attributes
    ----------


    Examples
    --------

    """

    def __init__(self,
        image_in='',
        parameter_file='',
        module_id='',
        step_id='',
        directory='',
        prefix='',
        datadir='',
        max_workers=0,
        verbosity=1,
        ):

        self.verbosity = verbosity

        self._module_id = module_id  # module name
        self.step_id = step_id or self._module_id # yml-file entry

        self.image_in = image_in
        if '.h5' in image_in:
            image_in = image_in.split('.h5')[0]

        if os.path.isdir(datadir):
            self.datadir = datadir
        elif os.path.isdir(image_in):
            self.datadir = os.path.abspath(image_in)
        else:
            self.datadir = os.path.abspath(os.path.dirname(image_in))

        self.directory = ''
        self.set_directory(directory)
        self.prefix = prefix

        self._logdir = 'logs'
        os.makedirs(self._logdir, exist_ok=True)

        self.parameter_file = parameter_file
        self._cfg = {}
        self.set_config()

        self.max_workers = max_workers
        self._n_workers = 0
        self._n_jobs = 0

        self.inputpaths = {}
        self.outputpaths = {}
        self.inputs = {}
        self.outputs = {}

        self._compute_env = os.environ.get('compute_env')

        self._delimiter = '_'

        self._set_suffix_formats()

        self._FPAR_NAMES = ('image_in', 'parameter_file', 'directory', 'datadir', 'prefix', 'inputs', 'outputs')

        self._fdict = {
            'fontsize': 7,
            'fontweight' : matplotlib.rcParams['axes.titleweight'],
            'verticalalignment': 'baseline',
            }

        self._images = []
        self._labels = []

    def __str__(self):
        """Print attributes in yml structure."""
        return self.dump_parameters()

    def run(self, steps=[]):
        """Run all steps in the module."""

        steps = steps or self._fun_selector.keys()
        for step in steps:
            print(f'Running {self._module_id}:{step}')
            self._fun_selector[step]()

    def _set_suffix_formats(self):
        """Set format strings for dimension suffixes."""

        d_suffixes = {
            'z': 'Z{z:03d}', 'y': 'Y{y:05d}', 'x': 'X{x:05d}',
            'c': 'C{c:03d}', 't': 'T{t:03d}', 's': 'S{s:03d}',
            'b': 'B{b:05d}',  # 'b': '{:05d}-{:05d}_{:05d}-{:05d}_{:05d}-{:05d}',
            'f': '{f}',
            'a': '{a}',
            }

        try:
            p_suffixes = self._cfg['suffix_formats']
        except KeyError:
            p_suffixes = {}

        self._suffix_formats = {**d_suffixes, **p_suffixes}

    def _prep_step(self, step, kwargs={}):
        """Run through common operations before step execution.

        - Attributes are set.
        - Paths are set.
        - Parallelization is generated.
        - Number of workers is set.
        - Parameters are dumped to yml and logs.

        Returns the argument list for parallelization.
        """

        if self._parallelization[step]:
            self.viewer = None

        # self.set_parameters(step, kwargs)
        kwargs.update({'step': step})
        self.__dict__.update(kwargs)

        self._set_paths_step()

        arglist = self._get_arglist()
        self._set_n_workers(len(arglist))

        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        return arglist

    def set_parameters(self, step, step_id=''):  # , kwargs={}
        """Set parameters for a processing step."""

        # if kwargs:
        #     kwargs.update({'step': step})
        #     self.__dict__.update(kwargs)
        # else:
            # DONE FIXME: this may overwrite things set on objects with parfile pars if no kwargs are given on method call;
            # NB: this should only be done from __init__ methods, not from _prep_step
        step_id = step_id or self.step_id
        pars = {'step': step}
        try:
            pars.update(self._cfg[step_id][step])
        except TypeError:
            pass
        except KeyError:
            pass
        self.__dict__.update(pars)

    def _merge_paths(self, paths, step, key='inputs', step_id=''):
        """Merge default paths with paths from parameterfile[=leading]."""

        step_id = step_id or self.step_id
        try:
            self._cfg[step_id][step][key]
        except (KeyError, TypeError):
            par_paths = {}
            # print(f'No {key} specified for step {step}:')
            # print(f'   ... using default file structures.')
        else:
            par_paths = {}
            for k, v in self._cfg[step_id][step][key].items():
                if v == 'image_in':
                    par_paths[k] = self.image_in
                elif v is None:
                    par_paths[k] = ''
                else:
                    par_paths[k] = self._cfg[step_id][step][key][k]
            print(f'Parameter file specified {key} for step "{step}":')
            print(f'   ... using {par_paths}.')

        def l2p(p):
            if isinstance(p, list):
                p = os.path.join(*p)
            return p

        paths = {**paths[key], **par_paths}
        paths = {ids: l2p(p) for ids, p in paths.items()}

        return paths

    def _get_arglist(self, parallelized_pars=[]):
        """Generate a argument list for multiprocessing."""

        parallelized_pars = parallelized_pars or self._parallelization[self.step]

        def getset(parname, alt_val):
            par = getattr(self, parname) or alt_val
            setattr(self, parname, par)
            return par

        imdims = ['stacks', 'channels', 'planes']
        if any(pp in imdims for pp in parallelized_pars):
            # if 'data' in self.inputs:
            #     # shading:estimate,
            #     image_in = self.inputs['data']  # shading, biasfield
            # elif 'metrics' in self.inputs:
            #     image_in = self.inputs['estimate']['data']
            # else:
            #     image_in = self.inputpaths['prep']['data']  # stitching
            first_step = list(self.inputpaths.keys())[0]
            image_in = self.inputpaths[first_step]['data']
            from stapl3d.preprocessing import shading  # TODO: without import
            iminfo = shading.get_image_info(image_in)
            pars = [getset(pp, iminfo[pp]) for pp in parallelized_pars]

        elif parallelized_pars == ['filepaths']:  # may generalize this?
            filepaths = self.filepaths
            pars = [getset(pp, filepaths) for pp in parallelized_pars]

        elif parallelized_pars == ['blocks']:
            blocks = list(range(len(self._blocks)))
            pars = [getset(pp, blocks) for pp in parallelized_pars]

        # elif parallelized_pars == ['blockfiles']:
        #     filepaths = self.blockfiles
        #     if self.blocks:
        #         filepaths = [filepaths[i] for i in self.blocks]
        #     pars = [getset(pp, filepaths) for pp in parallelized_pars]

        # elif parallelized_pars == ['_blocks']:
        #     if self.blocks:
        #         block_idxs = self.blocks
        #     else:
        #         block_idxs = [block.idx for block in self._blocks]
        #     block_obs = [block for block in self._blocks if block.idx in block_idxs]
        #     pars = [getset(pp, block_obs) for pp in parallelized_pars]

        elif parallelized_pars == ['volumes']:
            paths = get_paths(self._blocks[0].path)
            volumes = self.volumes
            def extract(name, node):
                if isinstance(node, h5py.Dataset):
                    volumes.append({name: {}})
                return None
            if not volumes:
                with h5py.File(paths['file'], 'r') as f:
                    f.visititems(extract)
            pars = [getset(pp, volumes) for pp in parallelized_pars]

        elif parallelized_pars == ['features']:
            feats = list(range(len(self.features.keys())))
            pars = [getset(pp, feats) for pp in parallelized_pars]

        elif parallelized_pars == []:
            pars = [(0,)]

        arglist = list(itertools.product(*pars))
        self._n_jobs = len(arglist)

        return arglist

    def set_config(self):
        """Load parameters from the yml parameter file."""

        if not self.parameter_file:
            return

        with open(self.parameter_file, 'r') as ymlfile:
            self._cfg = yaml.safe_load(ymlfile)

    def get_config(self):
        with open(self.parameter_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        return cfg

    def set_directory(self, directory='', subdirectory=''):
        """Work out the output (sub)directory.

        # 1) may be set in arg if called from main
        # 2) may be supplied in parameterfile
        # 3) may be supplied as argument to estimate
        # 4) may be supplied as argument to estimate_channel

        1. Use supplied outputdir.
        2. Use <imagedir>/<parfile_key>.
        """

        if directory:
            self.directory = os.path.abspath(directory)
        elif subdirectory:
            self.directory = os.path.join(self.datadir, subdirectory)

        if not self.directory:
            self.directory = os.path.join(self.datadir, self.step_id)

        os.makedirs(self.directory, exist_ok=True)

        self.directory = os.path.relpath(self.directory, self.datadir)

    def _step_pars(self, parsets, pdict):
        return {
            'files':  {s: pdict[s] for s in parsets['fpar']},
            'params': {s: pdict[s] for s in parsets['ppar']},
            'submit': {s: pdict[s] for s in parsets['spar']},
            }

    def dump_parameters(self, step='', filestem='', write=True):
        """Write parameters as yml."""

        # FIXME: not always serialized correctly (if they are references)

        steps = [step] if step else self._parameter_sets.keys()
        step_dict = {step: self._step_pars(self._parameter_sets[step], vars(self))
                     for step in steps}
        module_dict = {self._module_id: step_dict}

        if write:
            if not filestem:
                prefixes = [self.prefix, self._module_id, step]
                filestem = self._build_path(moduledir=self._logdir, prefixes=prefixes)
            filepath = self._abs(f'{filestem}.yml')
            with open(filepath, 'w') as f:
                yaml.dump(module_dict, f, default_flow_style=False)

        return yaml.dump(module_dict, default_flow_style=False)

    def _merge_parameters(self, ymls, outputpath, **kwargs):
        """Merge specific parameters from a set of yml files."""

        trees = kwargs['trees']

        aggregate = kwargs['aggregate'] if 'aggregate' in kwargs.keys() else True

        def replace(d, path, replacement):
            cur = d
            for k in path[:-1]:
                cur = cur[k]
            cur[path[-1]] = replacement

        with open(ymls[0], 'r') as ymlfile:
            cfg_out = yaml.safe_load(ymlfile)

        for tree in trees:
            aggr = {}
            for ymlpath in ymls:
                with open(ymlpath, 'r') as ymlfile:
                    cfg = yaml.safe_load(ymlfile)
                for item in tree:
                    cfg = cfg[item]
                # FIXME: abusing instance for aggregationVSreplace switch
                if isinstance(cfg, dict):
                    aggr.update(cfg)
                else:
                    aggr = cfg
            replace(cfg_out, tree, aggr)

        if outputpath:
            with open(outputpath, 'w') as f:
                yaml.dump(cfg_out, f, default_flow_style=False)

        return yaml.dump(cfg_out, default_flow_style=False)

    def _set_n_workers(self, n_workers=0):
        """Determine the number of workers."""

        cpu_count = multiprocessing.cpu_count()
        n_workers = self.max_workers or n_workers or cpu_count
        self._n_workers = min(n_workers, cpu_count)

    def format_(self, elements=[], delimiter=''):
        """String together elements of the filename."""

        # TODO: private
        # TODO: use alias instead of dataset?
        elements = elements or [self.prefix, self._module_id]
        delimiter = delimiter or self._delimiter
        name = delimiter.join([x for x in elements if x])
        # if use_fallback:, use_fallback=True
        #     name = name or self._module_id

        return name

    def get_filestem(self, sufdict={}, elements=[], delimiter=''):
        """Generate (partial) filename."""

        suf = []
        for k, v in sufdict.items():
            if v == 'p':
                suf.append(self._suffix_formats[k])
            elif v == '*' or v == '?':
                suf.append(self._suffix_formats[k].format(0).replace('0', v))
            else:
                suf.append(self._suffix_formats[k].format(v))

        elements = elements or [self.prefix, self._module_id]

        name = self.format_(elements + suf, delimiter)

        return name

    def _get_inputstem(self, step_id, step, idx=0):
        """Derive the inputstem from the usual previous step."""

        cfg_step = get_config_step(self.image_in, self.prefix, step_id, step)
        files = cfg_step['files']

        if files['_outputs']:
            instem = files['_outputs'][idx]
        else:
            filestem = self.get_filestem()
            instem = os.path.join(files['datadir'], files['directory'], filestem)

        return instem

    def _get_input(self, input, step_id, step, formatstring='{}', fallback=''):
        """

        input=False => use fallback value
        input=True  => derive from step_id+step
        input=<string> => use input
        """

        if isinstance(input, bool):
            if input:
                # try:
                instem = self._get_inputstem(step_id, step)
                # except:
                #     instem = self.get_filestem()
                #     instem = os.path.join(self.datadir, self.directory, filestem)
                return formatstring.format(instem)
            else:
                return fallback
        else:
            if os.path.exists(get_paths(input)['file']):
                return input

    def _set_inputpath(self, step_id, step, formatstring, fallback=''):

        self.inputpath = self._get_input(self.inputpath, step_id, step, formatstring, fallback)

    def _get_outputpath(self, filestem, ext='', delimiter='.', rel=False):

        filename = self.format_([filestem, ext], delimiter)
        p = os.path.join(self.directory, filename)
        if not rel:
            p = os.path.join(self.datadir, p)
        return p

    def _set_outputpath(self, filestem, ext=''):

        self.outputpath = self._get_outputpath(filestem, ext)

    # def set_paths(self, path_struture):
    #
    #     for step in self._fun_selector.keys():
    #         self.inputpaths[step] = [self._get_path(pspec) for pspec in path_struture[step]['inputs']]
    #         self.outputpaths[step] = [self._get_path(pspec) for pspec in path_struture[step]['outputs']]

    def _get_path(self, pspec={'suf': {}, 'ext': '', 'delimiter': '.'}):
        """"""

        if not pspec:
            p = self._get_outputpath(self.get_filestem(), rel=True)
        elif isinstance(pspec, dict):
            filestem = self.get_filestem(pspec['suf'])
            p = self._get_outputpath(filestem, pspec['ext'], rel=True)
        else:  # or maybe Pathlike...
            p = pspec

        return p
        # return os.path.relpath(p, self.datadir)

    def _abs(self, filepath):
        if filepath and not os.path.isabs(filepath):
            filepath = os.path.join(self.datadir, filepath)
        return filepath

    def _abspaths(self, paths, ):
        return {ids: self._abs(paths[ids]) for ids in paths.keys()}

    def _l2p(self, p):
        return os.path.join(*p) if isinstance(p, list) else p

    def _prep_paths(self, paths, reps={}, abs=True):
        """Format-in reps by keywords into (absolute) in- and outputpaths."""

        class FormatDict(dict):
            def __missing__(self, key):
                return '{' + key + '}'

        reps = FormatDict(**reps)

        paths = {ids: self._l2p(p) for ids, p in paths.items()}

        for ids, p in paths.items():
            paths[ids] = p.format_map(reps)

        if abs:
            paths = {ids: self._abs(p) for ids, p in paths.items()}

        return paths

    def _find_reps(self, filepat, filestem=None, block_idx=-1):

        reps = {}
        if '{b' in filepat:
            reps['b'] = block_idx
        if '{f}' in filepat:
            reps['f'] = filestem

        return reps

    def fill_paths(self, step_id, reps={}, abs=True):
        """Format-in reps by keywords into (absolute) in- and outputpaths."""

        inpaths = self._prep_paths(self.inputpaths[step_id], reps, abs)
        outpaths = self._prep_paths(self.outputpaths[step_id], reps, abs)

        return inpaths, outpaths

    # def _get_prevpath(self, suf, dir, ext):
    #     stem = self.get_filestem(elements=[self.prefix, suf])
    #     return os.path.join(self.datadir, dir, f'{stem}.{ext}')

    def _get_inpath(self, prev_path):
        """

        # -1) from inputpaths attribute directly
        # 0) from 'inputs' of current step in config file
        # 1) from 'outputs' of previous step in config dump
        # 2) from 'outputs' of previous step in config file
        # 3) from expected default?
        """

        try:
            pars = self._load_dumped_step(
                prev_path['moduledir'],
                prev_path['module_id'],
                prev_path['step'],
                )
            inpath = pars[prev_path['ioitem']][prev_path['output']]
        except (KeyError, TypeError):
            try:
                inpath = self._cfg[prev_path['step_id']][prev_path['step']][prev_path['ioitem']][prev_path['output']]
            except (KeyError, TypeError):
                print(f"Could not determine inputpath from {prev_path['module_id']}:{prev_path['step']}")
                # print(f"Trying default path")
                inpath = 'default'

        return inpath

    def _build_path(self, datadir='', moduledir='', prefixes=[], suffixes=[], ext='', rel=True):
        """Generate filepath.

        <datadir>/<directory>/<prefix>_<dim-suffixes>.<ext>
        """

        basename = self._build_basename(prefixes, suffixes, ext)

        moduledir = moduledir or self.directory
        filepath = os.path.join(moduledir, basename)

        if not rel:
            datadir = datadir or self.datadir
            filepath = os.path.join(datadir, filepath)

        return filepath

    def _build_basename(self, prefixes=[], suffixes=[], ext=''):
        """Generate filename with extension."""

        filestem = self._build_filestem(prefixes, suffixes)

        basename = self.format_([filestem, ext], delimiter='.')

        return basename

    def _build_filestem(self, prefixes=[], suffixes=[], use_fallback=True):
        """Generate filename without extension."""

        prefixes = prefixes or [self.prefix, self._module_id]
        prefix = self.format_(prefixes)

        suffix = self._build_suffix(suffixes)

        filestem = self.format_([prefix, suffix])

        if use_fallback:
            filestem = filestem or self._module_id

        return filestem

    def _build_suffix(self, suffixes=[]):
        """Join suffixes into a single string."""

        suf = []
        for s in suffixes:
            if isinstance(s, dict):
                suf += [self._unpack_suffix(k, v) for k, v in s.items()]
            elif isinstance(s, list):
                suf += s
            else:
                suf.append(s)

        return self._delimiter.join(suf)

    def _unpack_suffix(self, dim, val):
        """Turn a suffix {k: v} spec into a formatter, matcher or formatted suffix."""

        if val == 'p':
            s = self._suffix_formats[dim]
        elif val == '?':
            s = self._suffix_formats[dim].format(**{dim: 0}).replace('0', val)
        elif val == '*':
            s = self._pat2mat(self._suffix_formats[dim])
        else:
            s = self._suffix_formats[dim].format(**{dim: val})

        return s

    def _pat2mat(self, s, mat='*', pat=r"{[^{}]+}"):
        """Replace all sets of curly brackets with wildcard."""

        return re.sub(pat, mat, self._l2p(s))

    def _set_paths_step(self):
        self.inputs = self.inputpaths[self.step]
        self.outputs = self.outputpaths[self.step]
        self._verify_paths()

    def _verify_paths(self):
        for _, filepath in self.outputs.items():
            if '.h5' in filepath:
                filepath = filepath.split('.h5')[0]
            directory, filename = os.path.split(filepath)
            if directory and directory!='{d}':
                os.makedirs(directory, exist_ok=True)

    def _get_h5_dset(self, filepath, ids, slices={}):
        im = Image('{}/{}'.format(filepath, ids), permission='r')
        im.load(load_data=False)
        if slices:
            for d, slc in slices.items():
                idx = im.axlab.index(d)
                if slc == 'ctr':
                    slc = int(im.dims[idx] / 2)
                im.slices[idx] = slc
        data = im.slice_dataset(squeeze=False)
        im.close()
        return data

    def view(self, input='', images=[], labels=[], settings={}):

        import napari

        if images is None:
            images = []
        else:
            images = images or self._images
        if labels is None:
            labels = []
        else:
            labels = labels or self._labels

        slices = settings['slices'] if 'slices' in settings.keys() else {}

        viewer = napari.Viewer()
        self.viewer = viewer

        if isinstance(input, str):
            self.view_single(input, images, labels, slices)
        if isinstance(input, list):
            self.view_blocks(input, images, labels, slices)

        self.set_view(settings)

    def view_single(self, filepath='', images=[], labels=[], slices={}):

        for ids in images:
            try:
                data = self._get_h5_dset(filepath, ids, slices)
            except:
                pass
            else:
                self.viewer.add_image(data, name=ids)
                self.view_set_axes(filepath, ids)
        for ids in labels:
            try:
                data = self._get_h5_dset(filepath, ids, slices)
            except:
                pass
            else:
                self.viewer.add_labels(data, name=ids)
                self.view_set_axes(filepath, ids)

    def view_blocks(self, block_idxs=[], images=[], labels=[], slices={}):

        block_id0 = self._blocks[block_idxs[0]].id

        for block_idx in block_idxs:

            block = self._blocks[block_idx]
            filepath = block.path.replace('/{ods}', '')

            for ids in images + labels:

                im = Image('{}/{}'.format(filepath, ids), permission='r')
                im.load(load_data=False)
                im.close()
                affine = self._view_trans_affine(block.affine, im.elsize[:3])

                name = f'{block.id}_{ids}'
                data = self._get_h5_dset(filepath, ids, slices)

                if ids in images:
                    self.viewer.add_image(data, name=name, affine=affine)
                    clim = self.viewer.layers[f'{block_id0}_{ids}'].contrast_limits
                    self.viewer.layers[f'{block.id}_{ids}'].contrast_limits = clim
                if ids in labels:
                    self.viewer.add_labels(data, name=name, affine=affine)

        self.viewer.dims.axis_labels = [al for al in im.axlab]

    def _view_trans_affine(self, affine, elsize):
        Tt = np.copy(affine)
        for i, es in enumerate(elsize):
            Tt[0, i] *= es
        Ts = np.diag(list(elsize) + [1])
        return  Ts @ Tt

    def view_set_axes(self, filepath, ids):

        im = Image('{}/{}'.format(filepath, ids), permission='r')
        im.load(load_data=False)
        im.close()

        elsize = [es for i, es in enumerate(im.elsize) if im.dims[i] > 1]
        for lay in self.viewer.layers:
            lay.scale = elsize

        axlab = [al for i, al in enumerate(im.axlab) if im.dims[i] > 1]
        self.viewer.dims.axis_labels = axlab

    def set_view(self, settings={}):
        """Viewer settings functions."""

        # Set the window title bar.
        if 'title' in settings.keys():
            self.viewer.title = settings['title']
        # Move scrollbars to centreslices.
        if 'crosshairs' in settings.keys():
            self.viewer.dims.current_step = settings['crosshairs']
        # Show/hide axes.
        if 'axes_visible' in settings.keys():
            self.viewer.axes.visible = settings['axes_visible']
        # Set equal contrast limits.
        if 'clim' in settings.keys():
            for lay in self.viewer.layers:
                lay.contrast_limits = settings['clim']
        # Set the layer opacity.
        if 'opacity' in settings.keys():
            for lay in self.viewer.layers:
                lay.opacity = settings['opacity']

    def _init_log(self):

        logfile = self._build_path(moduledir=self._logdir, ext='log', rel=False)

        if self.verbosity == 0:
            handlers = [logging.FileHandler(logfile)]
        else:
            handlers = [logging.FileHandler(logfile), logging.StreamHandler()]
            # FIXME: this goes to stderr

        logging.basicConfig(
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s',
            handlers=handlers,
            )

    def _logstep(self, parameters=[]):

        if not parameters:
            parameters = self.dump_parameters(step=self.step)

        logging.info('Running step "{}" over {} jobs using {} workers'.format(self.step, self._n_jobs, self._n_workers))
        logging.info("Parameters: \n{}".format(parameters))

    def _logmp(self, filestem):

        # logging.info('Smoothing with sigma={}'.format(self.sigma))
        logging.info('foo{}'.format(self.sigma))
        logging.info("{} in PID {}".format(filestem, os.getpid()))


# class Reporter(Stapl3r):
#     """Correct z-stack shading."""
#     def __init__(self, image_in, parameter_file, **kwargs):
#         pass

    def report(self, outputpath='', name='', ioff=True, **kwargs):

        figsize = (8.27, 11.69)  # A4 portrait
        f = plt.figure(figsize=figsize, constrained_layout=False)
        gs = gridspec.GridSpec(1, 1, figure=f)

        figtitle = 'STAPL-3D {} report'.format(self._module_id)
        name = name or self.prefix  # TODO: we would always like the full name here?
        figtitle = '{} \n {}'.format(figtitle, name)

        # TODO generalize (to eg blocks) or remove and supply name from function call
        if 'channel' in kwargs.keys():
            channel = kwargs['channel']
            figtitle += ' channel ' + self._suffix_formats['c'].format(c=channel)
            suf = {'c': channel}
        else:
            channel = None
            suf = {}

        info_dict = self._get_info_dict(**kwargs)
        axdict = self._gen_subgrid(f, gs[0], channel=channel)

        self._plot_params(f, axdict, info_dict)
        self._plot_images(f, axdict, info_dict)
        self._plot_profiles(f, axdict, info_dict)

        f.suptitle(figtitle, fontsize=14, fontweight='bold')
        if outputpath is not None:
            outputpath = outputpath or self._build_path(suffixes=[suf])
            f.savefig(outputpath, format='pdf')
        if ioff:
            plt.close(f)

    def summary_report(self, name='', channel=None, ioff=True, outputpath=''):

        figsize = (8.27, 11.69)  # A4 portrait
        f = plt.figure(figsize=figsize, constrained_layout=False)
        gs = gridspec.GridSpec(1, 1, figure=f)

        figtitle = 'STAPL-3D {} report'.format(self._module_id)
        name = name or self.prefix  # TODO: we would like the full name here?
        figtitle = '{} \n {}'.format(figtitle, name)

        filestem = self._build_path()
        outputpath = outputpath or '{}_summary.pdf'.format(filestem)

        info_dict = self._get_info_dict_summary(filestem, channel=channel)

        if self._module_id == 'equalization':
            axdict = {'gs': gs}
        else:
            axdict = {'graph': f.add_subplot(gs[0].subgridspec(1, 3)[1:])}

        self._summary_report(f, axdict, info_dict)

        f.suptitle(figtitle, fontsize=14, fontweight='bold')
        f.savefig(outputpath)
        if ioff:
            plt.close(f)

    def _report_axes_pars(self, f, gs):

        gs00 = gs.subgridspec(10, 1)
        ax = f.add_subplot(gs00[0, 0])
        ax.set_title('parameters', self._fdict, fontweight='bold')
        ax.tick_params(axis='both', direction='in')

        return ax

    def _load_dumped_step(self, moduledir, module_id, step, pars={}):

        ymlpath = self._build_path(
            moduledir=moduledir,
            prefixes=[self.prefix, module_id, step],
            ext='yml',
            rel=False,
            )
        try:
            with open(ymlpath, 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
        except FileNotFoundError:
            pass
        else:
            for k in ['params', 'files', 'submit']:
                pars.update(cfg[module_id][step][k])

        return pars

    # def load_dumped_step(self, step, pars={}):
    #
    #     elements = [self.prefix, self._module_id, step]
    #     filestem = self.get_filestem(elements=elements)
    #     ymlpath = self._get_outputpath(filestem, 'yml')
    #     try:
    #         with open(ymlpath, 'r') as ymlfile:
    #             cfg = yaml.safe_load(ymlfile)
    #     except FileNotFoundError:
    #         pass
    #     else:
    #         for k in ['params', 'files', 'submit']:
    #             pars.update(cfg[self._module_id][step][k])
    #
    #     return pars

    def load_dumped_pars(self, pars={}):
        """TODO: private, or make intuitive action"""

        for step in self._fun_selector.keys():
            pars = self._load_dumped_step(self.directory, self._module_id, step, pars)

        return pars

    def _add_titles(self, axdict, titles):

        c = [0.5, 0.5, 0.5]
        titles_kwargs_sets = {
            'tl': {'fontweight': 'bold', 'c': c,
                   'loc': 'left'},
            'tr': {'fontweight': 'bold', 'c': c,
                   'loc': 'right'},
            'lc': {'fontweight': 'bold', 'c': c,
                   'rotation': 90, 'x': -0.1, 'y': 0.5, 'va': 'center'},
            'rc': {'fontweight': 'bold', 'c': c,
                   'rotation': -90, 'x': 1.1, 'y': 0.5, 'va': 'center'},
            'lcm': {'fontweight': 'bold', 'c': c,
                   'rotation': 90, 'x': -0.15, 'y': 0.3, 'va': 'center'},
            'rcm': {'fontweight': 'bold', 'c': c,
                   'rotation': -90, 'x': 1.6, 'y': 0.3, 'va': 'center'},
        }

        for axname, (title, kw_key, idx) in titles.items():
            ax = axdict[axname]
            if isinstance(ax, list):
                ax = ax[idx]
            ax.set_title(title, **titles_kwargs_sets[kw_key])

    def _plot_params(self, f, axdict, info_dict={}):
        """Show parameter table in report."""

        cellText = []
        for par, name in self._parameter_table.items():
            v = info_dict['parameters'][par]
            if not isinstance(v, list):
                v = [v]
            v = [np.round(x, 4) if isinstance(x, float) else x for x in v]
            cellText.append([name, ', '.join(str(x) for x in v)])

        if len(cellText):
            axdict['p'].table(cellText, loc='bottom')
        axdict['p'].axis('off')

    def _draw_thresholds(self, ax, thresholds, colors, linestyles, labels, legend='upper right'):

        # r = ax.get_ylim()
        # cs = ax.vlines(thresholds, r[0], r[1], colors, linestyles, label=['foo', 'bar', 'baz'])
        # if legend:
        #     ax.legend(loc=legend)

        def add_label(ax, val, lab, c):
            x_bounds = ax.get_xlim()
            x = (val - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
            ax.annotate(
                text=lab, xy=(x, 1.01), c=c,
                xycoords='axes fraction', va='top', ha='center',
                rotation=270, fontsize=5,
                )

        for t, c, l, lab in zip(thresholds, colors, linestyles, labels):
            if t is not None:
                ax.axvline(t, ymax=0.75, color=c, linestyle=l)
                add_label(ax, t, lab, c)

    def _gen_axes(self, f, gs, sg_width=9, row=0, col=0, xlab='', slc=None,
                  spines=[], ticks=[], ticks_position='left'):
        gs_sp = gs[row, col].subgridspec(1, sg_width)
        ax = f.add_subplot(gs_sp[slc])
        for l in spines:
            ax.spines[l].set_visible(False)
        if ticks:
            ax.yaxis.set_ticks(ticks)
        ax.yaxis.set_ticks_position(ticks_position)
        ax.tick_params(axis='both', direction='in')
        ax.set_xlabel('{} [px]'.format(xlab), loc='center', labelpad=-7)
        return ax

    def _get_extent(self, img, elsize, x_idx=2, y_idx=1):
        w = elsize[x_idx] * img.shape[1]
        h = elsize[y_idx] * img.shape[0]
        extent = [0, w, 0, h]
        return extent

    def _get_clim(self, cslc, q=[0.05, 0.95], roundfuns=[np.round, np.round]):
        c_min = np.amin([np.quantile(cslc[d], q[0]) for d in 'xyz'])
        c_min = max(0, c_min)  # FIXME: make negatives possible
        c_max = np.amax([np.quantile(cslc[d], q[1]) for d in 'xyz'])
        c_min = self._power_rounder(c_min, roundfuns[0])
        c_max = self._power_rounder(c_max, roundfuns[1])
        # TODO: c_min should never be c_max
        return [c_min, c_max]

    def _power_rounder(self, s, roundfun=np.round):
        if not s: return s
        d = np.power(10, np.floor(np.log10(s)))
        s = roundfun(s / d) * d
        return s

    def _add_scalebar(self, ax, w, fstring=r'{} $\mu$m', color='white'):
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm
        fontprops = fm.FontProperties(size=12)
        size = self._power_rounder(w / 5)
        scalebar = AnchoredSizeBar(ax.transData,
                                   size, fstring.format(size),
                                   'lower right',
                                   pad=0.1,
                                   color=color,
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)

    def _plot_colorbar(self, f, ax, im, cax=None, orientation='vertical', loc='right', clip_threshold=0.0):
        cbar = f.colorbar(im, cax=cax, extend='both', shrink=0.9, ax=ax, orientation=orientation)
        cbar.ax.tick_params(labelsize=7)
        if cax is not None:
            cax.yaxis.set_ticks_position(loc)
        if clip_threshold:
            cbar.set_ticks([0, clip_threshold, 1])
        return cbar

    def _clipped_colormap(self, threshold, clip_color, n_colors=100):
        """Create clipping colormap."""

        colors = matplotlib.cm.viridis(np.linspace(0, 1, n_colors))
        n = int(threshold * n_colors)
        for i in range(n):
            colors[i, :] = clip_color

        return matplotlib.colors.ListedColormap(colors)

    def _plot_profiles(self, f, axdict, info_dict):
        """Plot graphs with profiles."""
        pass

    def _plot_images(self, f, axdict, info_dict):
        """Plot graphs with profiles."""
        pass

    def _merge_reports(self, pdfs, outputpath):
        """Merge pages of a report."""

        if not pdfs:
            return

        try:
            from PyPDF2 import PdfFileMerger
            merger = PdfFileMerger()
            for pdf in pdfs:
                merger.append(pdf)
            merger.write(outputpath)
            merger.close()
        except:
            print('NOTICE: could not merge report pdfs')
        else:
            for pdf in pdfs:
                os.remove(pdf)

    def _merge(self, filetype, fun, **kwargs):
        """Merge files."""

        mpaths = []
        outputs = self._prep_paths(self.outputs)
        for filepath in self.filepaths:
            _, inputs, _ = self._get_filepaths_inout(filepath)
            #filestem = os.path.splitext(os.path.basename(filepath))[0]
            #inputs = self._prep_paths(self.inputs, reps={'f': filestem})
            mpaths.append(inputs[filetype])

        mpaths.sort()

        try:
            fun(mpaths, outputs[filetype], **kwargs)
            for mpath in mpaths:
                os.remove(mpath)
        except FileNotFoundError:
            print(f"WARNING: {filetype}s could not be merged")
