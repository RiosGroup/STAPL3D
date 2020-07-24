# -*- coding: utf-8 -*-

import logging

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Michiel Kleinnijenhuis"
__email__ = 'M.Kleinnijenhuis@prinsesmaximacentrum.nl'

import os
import sys
import h5py
import glob
import pickle
import random
import multiprocessing
import numpy as np
from xml import etree as et
from itertools import islice

import yaml

sys.stdout = open(os.devnull, 'w')

try:
    import nibabel as nib
except ImportError:
    print("nibabel could not be loaded")

try:
    from skimage.io import imread, imsave
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
                 elsize=None, axlab=None, dtype='float',
                 shape=None, dataslices=None, slices=None,
                 chunks=None, compression='gzip', series=0,
                 protective=False, permission='r+'):

        self.path = path
        self.elsize = elsize  # TODO: translations / affine
        self.axlab = axlab
        self.dtype = dtype
        self.dims = shape
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

        for ext in ['.h5', '.nii', '.dm3', '.tif', '.ims']:
            if ext in path:
                return ext

        for ext in ['.czi', '.lif']:  # ETC
            if ext in path:
                return '.pbf'

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
            print(info)

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

        if comps['ext'] == '.h5':
            comps_int = self.h5_split_int(filepath.split('.h5')[1])
            comps['int'] = filepath.split('.h5')[1]
            comps.update(comps_int)
        else:
            pass  # TODO: groups/dset from fname

        return comps

    def h5_split_int(self, path_int=''):
        """Split components of a h5 path."""

        path_int = path_int or self.path.split('.h5')[1]

        comps = {}

        if '/' not in path_int:
            raise Exception('no groups or dataset specified for .h5 path')

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
                h5path['int'] = '/DataSet/ResolutionLevel 0'

        if '/' not in h5path['int']:
            raise Exception('no groups or dataset specified for .h5 path')

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
                   '.pbf': self.pbf_load,
                   '.tif': self.pbf_load,
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
        self.ds = imread(self.path)
        self.dims = self.ds.shape
        self.dtype = self.ds.dtype

        if load_data:
            self.ds = imread(self.path)

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
        t_iter_slc = islice(timepoints_sorted, t_start, t_stop, t_step)
        for tp_idx, (_, tp) in enumerate(t_iter_slc):
            ch_names = ['Channel {}'.format(i) for i in range(len(tp))]
            channels_sorted = [(ch_name, tp[ch_name]) for ch_name in ch_names]
            c_iter_slc = islice(channels_sorted, c_start, c_stop, c_step)
            for ch_idx, (_, ch) in enumerate(c_iter_slc):

                data_tmp = ch['Data'][slcs[0], slcs[1], slcs[2]]
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

        if self.format == '.h5':
            ndim = len(self.dims)
        elif self.format == '.nii':
            ndim = len(self.file.header.get_data_shape())
        elif self.format == '.tif':
            ndim = len(self.dims)
        elif self.format == '.tifs':
            ndim = len(self.dims)
        elif self.format == '.dm3':
            ndim = len(self.dims)
        elif self.format == '.pbf':
            ndim = len(self.dims)
        elif self.format == '.ims':
            ndim = len(self.dims)
        elif self.format == '.dat':
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
            axlab = b''.join(self.ds.attrs['DIMENSION_LABELS']).decode("utf-8")
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

    def pbf_load_axlab(self):

        md = bf.get_omexml_metadata(self.path)
        axlabs = parse_xml_metadata(md)[3]

        return axlabs[0]

    def ims_load_axlab(self):

        return 'zyxct'

    def tif_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'zyxct'[:self.get_ndim()]  # FIXME: get from header?

        return axlab

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
                   '.nii': self.nii_write,
                   '.tif': self.tif_write,
                   '.tifs': self.tifs_write,
                   '.dat': self.dat_write}

        formats[self.format](data, slices)

    def h5_write(self, data, slices):
        """Write data to a hdf5 dataset."""

        self.write_block(self.ds, data, slices)

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
        if self.format == '.ims':
            ndim = 3

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

    def squeeze(self):
        pass

    def mkdir_p(self):
        try:
            os.makedirs(self.path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(self.path):
                pass
            else:
                raise


class MaskImage(Image):

    def __init__(self, path,
                 **kwargs):

        super(MaskImage, self).__init__(path, **kwargs)

    def invert(self):

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
                    from_empty=False):
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


def get_n_workers(n_workers, params):
    """Determine the number of workers."""

    cpu_count = multiprocessing.cpu_count()

    n_workers = cpu_count

    try:
        if params['n_workers'] > 0:
            n_workers = params['n_workers']
    except (KeyError, TypeError):
        pass

    n_workers = min(n_workers, cpu_count)

    return n_workers


def get_blockfiles(image_in, block_dir, block_selection=[], block_postfix='.h5'):

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

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    blocksize = im.dims
    blocksize[im.axlab.index('x')] = bs
    blocksize[im.axlab.index('y')] = bs

    return blocksize


def get_blockmargin(image_in, bm=64):

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    blockmargin = [0] * len(im.dims)
    blockmargin[im.axlab.index('x')] = bm
    blockmargin[im.axlab.index('y')] = bm

    return blockmargin


def get_blockinfo(image_in, parameter_file, params=dict(blocksize=[], blockmargin=[], blockrange=[])):

    if not params['blocksize']:
        ds_par = get_params(dict(), parameter_file, 'dataset')
        bs = ds_par['blocksize_xy'] or 640
        params['blocksize'] = get_blocksize(image_in, bs)

    if not params['blockmargin']:
        ds_par = get_params(dict(), parameter_file, 'dataset')
        bm = ds_par['blockmargin_xy'] or 64
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

    im = Image(image_in, permission='r')
    im.load(load_data=False)
    mpi = wmeMPI(usempi=False)
    mpi.set_blocks(im, blocksize, blockmargin)
    im.close()

    return len(mpi.blocks)


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


def get_outputdir(image_in, parameter_file, outputdir, step_id, fallback=''):

    dirs = get_params(dict(), parameter_file, 'dirtree')
    try:
        subdir = dirs['datadir'][step_id] or ''
    except KeyError:
        subdir = fallback

    if not outputdir:
        paths = get_paths(image_in)
        datadir, filename = os.path.split(paths['base'])
        outputdir = os.path.join(datadir, subdir)

    os.makedirs(outputdir, exist_ok=True)

    return outputdir


def prep_outputdir(outputdir, image_in='', subdir=''):
    """"""

    if not outputdir:
        paths = get_paths(image_in)
        datadir, filename = os.path.split(paths['base'])
        outputdir = os.path.join(datadir, subdir)

    os.makedirs(outputdir, exist_ok=True)

    return outputdir


def get_paths(image_in, resolution_level=-1, channel=0, outputstem='', step='', save_steps=False):
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
    if datatype:
        from skimage.util.dtype import convert
        data = convert(data, np.dtype(datatype), force_copy=False)
        props['dtype'] = datatype

    im_out = Image(image_out, **props)
    im_out.create()
    im_out.write(data.transpose().astype(props['dtype']))
    im_in.close()
    im_out.close()
