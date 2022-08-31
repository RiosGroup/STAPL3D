#!/usr/bin/env python

"""Block operations.

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

from stapl3d import parse_args, Stapl3r, Image, LabelImage, imarisfiles
from stapl3d.preprocessing.registration import get_affine

import glob
import h5py
import pathlib

logger = logging.getLogger(__name__)


def main(argv):
    """Block operations."""

    steps = ['split', 'merge']  # TODO: make this useful
    args = parse_args('blocks', steps, *argv)

    subclasses = {'split': Splitt3r, 'merge': Merg3r}

    for step in args.steps:
        block3r = subclasses[step](
            args.image_in,
            args.parameter_file,
            step_id=args.step_id,
            directory=args.outputdir,
            prefix=args.prefix,
            max_workers=args.max_workers,
        )
        block3r._fun_selector[step]()


class Block(object):
    """Block."""

    def __init__(self, id='', idx=0, path='',
                 axlab='', elsize={}, slices={},
                 blocker_info={}, affine=[]):

        self.id = id  # block ID: default f'B{idx:05d}'
        self.idx = idx  # block index

        self.path = path  # path to blockfile: f'{prefix}{block.id}.h5'

        self.axlab = axlab
        self.elsize = {al: float(es) for al, es in elsize.items()}
        self.blocker_info = blocker_info
        self._set_block_slices(slices, blocker_info)
        self.shape = {al: slc.stop - slc.start for al, slc in slices.items()}

        # TODO: integrate with affine matrix on Image class, (and registration module)
        #if affine:
        #    self.affine = affine
        #else:  # default: only block translation in voxel space
        spacing = [es for al, es in elsize.items() if al in 'xyz']

        starts = [slc.start for al, slc in self.slices.items() if al in 'xyz']
        origin = [sp * st for sp, st in zip(spacing, starts)]
        spacing = spacing or [1, 1, 1]
        origin = origin or [0, 0, 0]
        self.affine = get_affine([0, 0, 0], spacing, origin)

        starts = [slc.start for al, slc in self.slices_region.items() if al in 'xyz']
        origin = [sp * st for sp, st in zip(spacing, starts)]
        spacing = spacing or [1, 1, 1]
        origin = origin or [0, 0, 0]
        self.affine_region = get_affine([0, 0, 0], spacing, origin)

        self.datasets = {}

        # self.create_dataset(self.ids)
        # self.write_blockinfo(self.path)

    def _set_block_slices(self, slices, blocker_info):
        """Set the slicers of the block.

        slices:                     with margin into the full dataset
        slices_region:              without margin into the full dataset
        slices_blockfile:           with margin into the block
        slices_region_blockfile:    without margin into the block
        """

        self.slices = slices
        self.slices_region = {}
        self.slices_blockfile = {al: slice(0, slc.stop - slc.start)
                                 for al, slc in slices.items()}
        self.slices_region_blockfile = {}

        for al, slc in slices.items():
            (fc, fC), (bc, bC) = self._margins(
                slc.start, slc.stop,
                blocker_info['fullsize'][al],
                blocker_info['blocksize'][al],
                blocker_info['blockmargin'][al],
                blocker_info['shift_final_block_inward'],
                )
            self.slices_region[al] = slice(fc, fC)
            self.slices_region_blockfile[al] = slice(bc, bC)

    def _margins(self, fc, fC,
                 fullsize, blocksize, margin,
                 shift_final_block_inward):
        """Return coordinates (fullstack and block) corrected for margin.

        bc and bC are start and stop of the slice in the block voxel space
        fc and fC are start and stop of the slice in the dataset voxel space
        """

        #fullsize = self.blocker_info['fullsize'][al]
        #blocksize = self.blocker_info['blocksize'][al]
        #margin = self.blocker_info['blockmargin'][al]
        #shift_final_block_inward = self.blocker_info['shift_final_block_inward']

        final_block = fC >= fullsize

        if shift_final_block_inward and final_block:
            bc = margin
            bC = blocksize - margin  # always full blocksize
            fc = fullsize - blocksize
            fC = fullsize
            return (fc, fC), (bc, bC)

        # lower coordinates
        if fc == -margin:  # lower-end boundary block with pad_before = margin
            fc = 0
            bc = margin
        elif fc == 0:  # lower-end boundary block without padding
            bc = 0
        else:
            bc = margin
            fc += margin

        # upper coordinates
        if fC > fullsize:  # upper-end boundary block with pad_after = fC - fullsize
            bC = bc + (fullsize % blocksize)
            fC = fullsize
        elif fC == fullsize:  # upper-end boundary block without padding
            bC = bc + blocksize + (fullsize % blocksize)  # ???
        else:
            bC = bc + blocksize
            fC -= margin

        return (fc, fC), (bc, bC)

    def write_blockinfo(self, filepath=''):
        """Write the block attributes to file."""

        filepath = filepath or self.path
        f = h5py.File(filepath, 'a')

        grp = f.require_group('block_info')
        self.write_block_attributes(grp)

        grp = grp.require_group('blocker_info')
        self.write_blocker_attributes(grp)

        f.close()

    def load_blockinfo(self, filepath=''):
        """Load the block attributes from file."""

        filepath = filepath or self.path

        f = h5py.File(filepath, 'r')

        self.load_block_attributes(f['block_info'])

        self.load_blocker_attributes(f['block_info']['blocker_info'])

        f.close()

    def _get_block_attributes(self):
        """Return lists of attributes associated with the block."""

        attr_simple = ['axlab']  # NB: essential for loading in order
        attr_simple += ['id', 'idx', 'path', 'affine']
        attr_dict = ['elsize', 'shape']
        attr_slices = [
            'slices',
            'slices_region',
            'slices_blockfile',
            'slices_region_blockfile',
            ]

        attr_todo = ['chunks']  # TODO

        return attr_simple, attr_dict, attr_slices

    def write_block_attributes(self, grp):
        """Write the block attributes to file."""

        attrs = grp.attrs

        attr_simple, attr_dict, attr_slices = self._get_block_attributes()

        for attr_name, attr_value in vars(self).items():

            if attr_name in attr_simple:
                attrs[attr_name] = attr_value

            elif attr_name in attr_dict:
                for al, v in attr_value.items():
                    attrs[f'{attr_name}_{al}'] = v

            elif attr_name in attr_slices:
                for al, v in attr_value.items():
                    attrs[f'{attr_name}_{al}_start'] = v.start
                    attrs[f'{attr_name}_{al}_stop'] = v.stop

    def load_block_attributes(self, grp, axlab=''):
        """Load the block attributes from file."""

        attrs = grp.attrs

        attr_simple, attr_dict, attr_slices = self._get_block_attributes()

        for attr_name in attr_simple:
            setattr(self, attr_name, attrs[attr_name])

        axlab = ''.join(axlab) or ''.join(self.axlab)  # for loading axlab subset
        for attr_name in attr_dict:
            attr_value = {al: attrs[f'{attr_name}_{al}'] for al in axlab}
            setattr(self, attr_name, attr_value)

        for attr_name in attr_slices:
            attr_value = {al: slice(
                attrs[f'{attr_name}_{al}_start'],
                attrs[f'{attr_name}_{al}_stop'],
                ) for al in axlab}
            setattr(self, attr_name, attr_value)

        self.axlab = ''.join(axlab)

    def _get_blocker_attributes(self):
        """Return lists of attributes associated with the blocker."""

        attr_simple = ['_axlab']  # NB: essential for loading in order
        attr_simple += ['inputpath']
        attr_simple += [
            'boundary_truncation',
            'shift_final_block_inward',
            ]
        attr_dict = ['fullsize', 'blocksize', 'blockmargin']

        attr_todo = ['pad_kwargs']  # TODO

        return attr_simple, attr_dict

    def write_blocker_attributes(self, grp):
        """Write the blocker attributes to file."""

        attrs = grp.attrs

        attr_simple, attr_dict = self._get_blocker_attributes()

        for attr_name, attr_value in self.blocker_info.items():

            if attr_name in attr_simple:
                attrs[attr_name] = attr_value

            elif attr_name in attr_dict:
                for al, v in attr_value.items():
                    attrs[f'{attr_name}_{al}'] = v

    def load_blocker_attributes(self, grp):
        """Load the blocker attributes from file."""

        attrs = grp.attrs

        attr_simple, attr_dict = self._get_blocker_attributes()

        blocker_info = {}
        for attr_name in attr_simple:
            blocker_info[attr_name] = attrs[attr_name]

        for attr_name in attr_dict:
            blocker_info[attr_name] = {al: attrs[f'{attr_name}_{al}']
                                       for al in attrs['_axlab']}

        blocker_info['pad_kwargs'] = {}  # TODO
        blocker_info['_axlab'] = ''.join(blocker_info['_axlab'])

        self.blocker_info = blocker_info

    def create_dataset(self, ids='', blockfile='', axlab='', elsize={},
                       dtype='', slices={}, blocker_info={}, create_image=False):
        """Create an ND dataset in the Block."""

        blockfile = blockfile or self.path

        axlab = axlab or self.axlab  # selector of axes subsets
        elsize = {al: {**elsize, **self.elsize}[al] for al in axlab}
        slices = {al: {**slices, **self.slices}[al] for al in axlab}

        binfo = {**self.blocker_info, **blocker_info}  # insert axes found in blocker_info argument
        _, attr_dict = self._get_blocker_attributes()
        for attr_name in attr_dict:  # remove axes not in axlab
            binfo[attr_name] = {al: binfo[attr_name][al] for al in axlab}

        self.datasets[ids] = Block_dataset(
            ids, blockfile, axlab, elsize,
            dtype, slices, binfo,
            )

        if create_image:
            self.datasets[ids].create_image()

    def __str__(self):
        return yaml.dump(vars(self), default_flow_style=False)


class Block_dataset(Block):
    """Block dataset."""

    def __init__(self, ids, blockfile, axlab, elsize, dtype, slices, blocker_info):

        self.ids = ids
        self.blockfile = blockfile
        self.path = f'{blockfile}/{ids}'

        self.axlab = axlab
        self.elsize = {al: float(es) for al, es in elsize.items()}
        self.blocker_info = blocker_info
        self._set_block_slices(slices, blocker_info)
        self.shape = {al: slc.stop - slc.start for al, slc in slices.items()}

        self.dtype = dtype
        self.chunks = None
        self.image = None

    def create_image(self, path='', data=None, from_source=False, from_block=False):
        """Create the block Image object."""

        if from_block:
            im = Image(path, permission='r')
            im.load()
            props = im.get_props2()
            props['path'] = ''
            im.close()
        else:
            props = {
                'path': path,
                'elsize': self.elsize,
                'axlab': self.axlab,
                'dtype': self.dtype,
                'shape': self.shape,
                'chunks': self.chunks,
            }

        self.image = Image(**props)
        self.image.create()  # FIXME: dat_create creates empty np array on self.image.ds

        if data is not None:
            self.image.ds[:] = data
        elif from_source:
            self.read(from_source=True, from_block=False, padded=True)
        elif from_block:
            self.read(from_source=False, from_block=True, padded=True)

        self.image.close()

    def read(self, from_source=False, from_block=False, padded=True):
        """Read block data."""

        if from_source:
            data = self.read_data_from_source(padded=True)
            #data = self.read_data_from_source(padded)
            self.dtype = self.dtype or data.dtype
            if self.image is None:
                self.create_image(data=data)  # self.path?

        elif from_block:
            if self.image is None:
                self.create_image(self.path, from_block=True)
            data = self.read_data_from_blockfile(padded=True)

        self.image.ds[:] = data

    def read_data_from_source(self, padded=True):
        """Read block data from full dataset."""

        path = self.blocker_info['inputpath']
#        if self.ids_in:
#            path = path + f'/{self.ids_in}'  # FIXME: read from h5
        src_im = Image(path, permission='r')
        src_im.load()

        # TODO: padding into slice_dataset method?
        if padded:
            pad_width, src_im.slices = self.get_padding()
        else:
            pad_width = None
            src_im.slices = self.slices_region

        data = src_im.slice_dataset(squeeze=False)
        src_im.close()

        if pad_width is not None:
            data = np.pad(data, pad_width, **self.blocker_info['pad_kwargs'])

        return data

    def read_data_from_blockfile(self, padded=True):
        """Read block data from blockfile.
        """

        path = self.path
        src_im = Image(path, permission='r')
        src_im.load()

        if not padded:
            src_im.slices = [self.slices_region_blockfile[al] for al in src_im.axlab]

        data = src_im.slice_dataset(squeeze=False)

        # grp = src_im.ds
        axlab = src_im.ds.attrs['DIMENSION_LABELS']
        grp = src_im.file['block_info']
        self.load_block_attributes(grp, axlab)
        self.load_blocker_attributes(src_im.file['block_info']['blocker_info'])

        src_im.close()

        return data

    def get_padding(self):
        """Return pad_width from slices overshoot and truncated slices."""

        padding, slices = [], []

        for al, bslc in self.slices.items():

            m = self.blocker_info['fullsize'][al]

            if bslc.start < 0:
                pad_before = -bslc.start
                start = 0
            else:
                pad_before = 0
                start = bslc.start

            if bslc.stop > m:
                pad_after = bslc.stop - m
                stop = m
            else:
                pad_after = 0
                stop = bslc.stop

            slices.append(slice(start, stop))
            padding.append((pad_before, pad_after))

        return tuple(padding), slices

    def write(self, data):
        """Write the block data to file."""

        if self.image.format == '.dat':  # we have an image in memory
            self.image.path = self.path
            self.image.set_format()
            self.image.create()
            #self.chunks = self.image.chunks
        else:
            self.image.load()

        self.image.write(data)

        self.write_block_attributes(self.image.ds)
        self.write_blockinfo(filepath=self.blockfile)  # TODO: conditional (write-protection?)

        self.image.close()

    def _get_block_attributes(self):
        """Return lists of attributes associated with the block."""

        attr_simple = ['axlab']  # NB: essential for loading in order
        # attr_simple += ['path']  # FIXME: path is saved as {ods} if getting from block_info group
        attr_dict = ['elsize', 'shape']
        attr_slices = [
            'slices',
            'slices_region',
            'slices_blockfile',
            'slices_region_blockfile',
            ]

        attr_todo = ['chunks']  # TODO

        return attr_simple, attr_dict, attr_slices


class Block3r(Stapl3r):
    _doc_main = """Divide a dataset into blocks.

    A dataset can be either
    1. a large ND dataset
    2. a set of stacks

    Input:
    Provide the path to the dataset as the 'image_in' argument to blocks.Block3r
    or specify it directly via the parameter file <step_id>:blockinfo:inputs:data
    or load from a previous run:
    Input format:
    1. single file:     path:       e.g. '<dataset>.ims'
    2. stacks:          pattern:    e.g. '{f}.czi'
    3. blocks:          pattern:    e.g. 'blocks/blocks_B{b:05d}.h5'

    Output:
    Provide the path to the blockfiles
    via the parameter file <step_id>:blockinfo:outputs:blockfiles
    Output format:
    1. single file:     pattern:    e.g. 'blocks/blocks_B{b:05d}.h5'
    2. stacks:          pattern:    e.g. 'blocks/{f}.h5'
    3. blocks:          pattern:    e.g. 'blocks/blocks_B{b:05d}.h5'

    Parameter file example:
    blocks:
        blockinfo:
            blocksize:
                x: 1280
                y: 1280
            blockmargin:
                x: 64
                y: 64

    """
    _doc_attr = """

    Block3r Attributes
    ----------
    blocks : list, default [0, 1, ..., N]
        List of block indices to process.

    fullsize : dict, default {al: <axis-shape> for al in <axislabels>}
        Axislabel-Size key-value pairs for the full dataset size.
    blocksize : dict, default {al: <axis-shape> for al in <axislabels>}
        Axislabel-Size key-value pairs for the block.
    blockmargin : dict, default {al: 0 for al in <axislabels>}
        Axislabel-Margin key-value pairs for the block margin / padding.

    pad_kwargs : dict, default {}
        Keyword arguments passed to np.pad.

    boundary_truncation : str, default 'dataset'
        Truncate the boundary blocks.
        (i.e. remove the block padding on boundary blocks).
        '': no truncation
        'dataset': truncate to the dataset area
        'margin': truncate to the dataset area + blockmargin
    shift_final_block_inward : bool, default False
        Create blocks of equal size by shifting the final block inward.

    """
    _doc_exam = """

    Examples
    --------
    # 1. Creating a block3r that divides HFK16w.ims into a grid of blocks.
    from stapl3d import blocks
    block3r = blocks.Block3r('HFK16w.ims')
    # Print the block information for each block.
    block3r.print_blockinfo()

    # 2. Creating a block3r with all the czi-images in the current directory.
    from stapl3d import blocks
    block3r = blocks.Block3r('{f}.czi')
    block3r.write_blockinfo()

    # 3. Creating a simple 3D block3r from scratch.

    from stapl3d import blocks
    block3r = blocks.Block3r()

    fullsize = {'z': 100, 'y': 750, 'x': 750}
    blocksize = {'y': 500, 'x': 500}
    blockmargin = {'y': 50, 'x': 50}
    block3r.set_fullsize(fullsize)
    block3r.set_blocksize(blocksize)
    block3r.set_blockmargin(blockmargin)

    block3r.boundary_truncation = 'margin'

    block3r.generate_blocks()

    # Print the blocker and block information for each block.
    print(block3r)
    block3r.print_blockinfo()

    # View the block layout in napari
    block3r.view_block_layout = ['fullsize', 'margins', 'blocks']
    block3r.view()

    """
    __doc__ = f"{_doc_main}{Stapl3r.__doc__}{_doc_attr}{_doc_exam}"

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'blocks'

        super(Block3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector = {
            'blockinfo': self.write_blockinfo,
            }

        self._parallelization = {
            'blockinfo': ['blocks'],
            }

        self._parameter_sets = {
            'blockinfo': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('fullsize', 'blocksize', 'blockmargin',
                         'boundary_truncation',
                         'shift_final_block_inward', 'pad_kwargs'),
                'spar': ('_n_workers', 'blocks'),
                },
            }

        self._parameter_table = {
            'fullsize': 'Size of the full dataset',
            'blocksize': 'Block size',
            'blockmargin': 'Block margin',
            'boundary_truncation': 'Truncate boundary blocks',
            'shift_final_block_inward': 'Shift the upper-bound blocks inward',
            'pad_kwargs': 'Keyword arguments passed to numpy.pad',
        }

        default_attr = {
            'fullsize': {},
            'blocksize': {},
            'blockmargin': {},
            'boundary_truncation': 'dataset',  # 'dataset', 'margin', ''
            'shift_final_block_inward': False,
            'pad_kwargs': {},
            'blocks': [],
            '_axlab': '',
            '_elsize': {},
            '_blocks': [],
            '_seamgrid': [],
            '_inputmode': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._prep_blocks()

        self.view_block_layout = []  # ['fullsize', 'margins', 'blocks']
        self._images = []
        self._labels = []

    def _init_paths(self):
        """Set input and output filepaths.

        REQUIRED:
        blockinfo:inputs:data           input image OR {f}.<ext>
        blockinfo:outputs:blockfiles    blockpattern ...{b}... OR ...{f}...

        # TODO: flexible outputdirectory throughout
        """

        if '{f' in self.image_in:
            prefixes, suffix = [''], 'f'
        else:
            prefixes, suffix = [self.prefix, 'blocks'], 'b'

        bpat = self._build_path(
            moduledir='blocks',
            prefixes=prefixes,
            suffixes=[{suffix: 'p'}],
            )

        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        self._paths = {
            'blockinfo': {
                'inputs': {
                    'data': self.image_in,
                    },
                'outputs': {
                    'blockfiles': f"{bpat}.h5",
                },
            },
        }

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def write_blockinfo(self, **kwargs):
        """Write the block attributes to file for all blocks."""

        arglist = self._prep_step('blockinfo', kwargs)

        for i, block in enumerate(self._blocks):
            if i in self.blocks:
                block.write_blockinfo()

    def print_blockinfo(self, **kwargs):
        """Print the region's slices into the full dataset for all blocks."""

        arglist = self._prep_step('blockinfo', kwargs)

        for i, block in enumerate(self._blocks):
            if i in self.blocks:

                slc = {al: f'slice({slc.start}, {slc.stop})'  # , {slc.step}
                       for al, slc in block.slices_region.items()}
                print(f'Block {i:05d} with id {block.id} contains region (unpadded)')
                print(yaml.dump(slc, default_flow_style=False))

                slc = {al: f'slice({slc.start}, {slc.stop})'  # , {slc.step}
                       for al, slc in block.slices.items()}
                print(f'Block {i:05d} with id {block.id} contains region (padded)')
                print(yaml.dump(slc, default_flow_style=False))

    def generate_blocks(self):
        """Generate blocks from attributes."""

        inputpath = self.inputpaths['blockinfo']['data']
        print(f'Generating blocks from "{inputpath}"')
        self._set_inputmode(inputpath)
        assert self._inputmode in ['grid', 'stacks']
        self._generate_blocks()

    def load_blocks(self, filepaths=''):
        """Load blocks from STAPL3D blockfiles."""

        inputpath = self.inputpaths['blockinfo']['data']
        print(f'Loading blocks from "{inputpath}"')
        self._set_inputmode(inputpath)
        assert self._inputmode == 'blockfiles'
        self._load_blocks(filepaths)

    def _prep_blocks(self):
        """Set blocker characteristics and blocks."""

        inputpath = self.inputpaths['blockinfo']['data']

        self.filepaths = self._glob_h5(self._pat2mat(self._abs(inputpath)))

        self._set_inputmode(inputpath)

        self.set_fullsize()  # FIXME: only correct for firstfile of stacks
        self.set_blocksize()  # FIXME: only correct for firstfile of stacks
        self.set_blockmargin()

        if (set(self.fullsize.keys()) ==
            set(self.blocksize.keys()) ==
            set(self.blockmargin.keys())) and self.fullsize:

            if self._inputmode == 'blockfiles':
                self._load_blocks()
            else:
                self._generate_blocks()

            self._set_seamgrid()  # TODO: for load_blocks_from_blockfiles

    def _set_inputmode(self, inputpath=''):
        """Set the inputmode according to inputpath (grid, stacks or blocks)."""

        inputpath = inputpath or self.inputpaths['blockinfo']['data']

        if '{b' in inputpath or '{f' in inputpath:
            if '.h5' in inputpath:
                try:
                    block = Block()
                    block.load_blockinfo(self.filepaths[0])
                except FileNotFoundError:
                    print('Could not find blockfile.')
                    self._inputmode = 'stacks'  # '{f}.czi'
                except IndexError:
                    print('Could not find blockinfo group in blockfile.')
                    self._inputmode = 'stacks'  # '{f}.czi'
                except KeyError:
                    self._inputmode = 'stacks'  # '{f}.czi'
                else:
                    self._inputmode = 'blockfiles'  # 'blocks\\blocks_B{b:05d}.h5'
            else:
                self._inputmode = 'stacks'  # '{f}.czi'
        else:
            self._inputmode = 'grid'  # '<dataset>.ims'

        print(f'Input mode set to "{self._inputmode}"')

    def set_fullsize(self, fullsize={}):
        """Set the shape of the full dataset."""

        try:
            dims = None
            for i, filepath in enumerate(self.filepaths):
                im = Image(filepath, permission='r')
                im.load(load_data=False)
                im.close()
                if dims is None:
                    dims = np.array([0] * len(im.dims))
                dims = [d.item() for d in np.maximum(dims, np.array(im.dims))]
                self._elsize = dict(zip(im.axlab, im.elsize))
            imsize = dict(zip(im.axlab, dims))
        except:
            imsize = {}

        # argument overrides attribute overrides image shape
        self.fullsize = {**imsize, **self.fullsize, **fullsize}

        print(f'Full dataset size set to "{self.fullsize}"')

    def set_blocksize(self, blocksize={}):
        """Set the size of the block."""

        self.blocksize = {**self.fullsize, **self.blocksize, **blocksize}

        print(f'Block size set to "{self.blocksize}"')

    def set_blockmargin(self, blockmargin={}):
        """Set the margins of the block."""

        bm_im = {d: 0 for d in self.blocksize.keys()}
        self.blockmargin = {**bm_im, **self.blockmargin, **blockmargin}

        logging.info(f'Block margin set to "{self.blockmargin}"')

    def _generate_blocks(self):

        axlab = self._axlab or [al for al in self.fullsize.keys()]
        elsize = self._elsize if self._elsize else {al: 1 for al in axlab}

        if self._inputmode == 'stacks':
            starts, stops = self._get_bounds_for_files(self.blockmargin)
        else:
            starts, stops = self._get_bounds_for_grid(self.blockmargin)

        block_template = self.outputpaths['blockinfo']['blockfiles']

        blocks = []
        for b_idx, (start, stop) in enumerate(zip(starts, stops)):

            slices = {al: slice(int(sta), int(sto))
                      for al, sta, sto in zip(axlab, start, stop)}

            blocker_info = {k:v for k, v in vars(self).items()
                            if k in self._parameter_table.keys()}
            blocker_info['_axlab'] = axlab
            if self._inputmode == 'stacks':
                blocker_info['inputpath'] = self.filepaths[b_idx]
            else:
                blocker_info['inputpath'] = self.inputpaths['blockinfo']['data']

            block = Block(
                id=self._suffix_formats['b'].format(b=b_idx),
                idx=b_idx,
                path=self._get_blockpath(block_template, b_idx),
                axlab=axlab,
                elsize=elsize,
                slices=slices,
                blocker_info=blocker_info,
                )
            blocks.append(block)

        self._blocks = blocks

    def _get_bounds_for_grid(self, margin):
        """Generate slices for a grid of blocks."""

        axlab = ''.join(margin.keys())

        imslices = {al: slice(0, self.fullsize[al]) for al in axlab}
        shape = {al: len(range(*slc.indices(slc.stop))) for al, slc in imslices.items()}
        blocksize = self.blocksize or shape
        blocksize = {al: bs if bs else shape[al] for al, bs in blocksize.items()}

        starts, stops, = {}, {}
        for al in axlab:
            starts[al], stops[al] = self._get_blockbounds(
                imslices[al].start,
                imslices[al].stop,
                blocksize[al],
                margin[al],
                )

        ndim = len(axlab)
        starts = tuple(starts[dim] for dim in axlab)
        stops = tuple(stops[dim] for dim in axlab)
        startsgrid = np.array(np.meshgrid(*starts))
        stopsgrid = np.array(np.meshgrid(*stops))
        starts = np.transpose(np.reshape(startsgrid, [ndim, -1]))
        stops = np.transpose(np.reshape(stopsgrid, [ndim, -1]))

        return starts, stops

    def _get_bounds_for_files(self, margin):
        """Generate slices for a list of files."""

        stops = []
        for filepath in self.filepaths:
            try:
#                if self.ids:
#                    filepath = f'{filepath}/{self.ids}'  # FIXME
                im = Image(filepath)
                im.load(load_data=False)
                stops.append(im.dims)
                im.close()
            except:  # test for featur3r: TODO: generalize
                stops.append([0, 0, 0])

        starts = [[0] * len(stop) for stop in stops]

        return starts, stops

    def _get_blockbounds(self, ds_start, ds_stop, blocksize, margin):
        """Get the block range (with margin applied) for a dimension."""

        # blocks
        starts = np.array((range(ds_start, ds_stop, blocksize)))
        stops = np.array(starts) + blocksize

        if self.shift_final_block_inward:
            stops[-1] = ds_stop
            starts[-1] = ds_stop - blocksize

        # blocks with margin
        starts = np.array(starts) - margin
        stops = np.array(stops) + margin

        # boundary block truncations
        if self.boundary_truncation in ['dataset', 'margin']:
            bounds = {
                'dataset': [ds_start, ds_stop],
                'margin': [ds_start - margin, ds_stop + margin]
            }
            bounds = bounds[self.boundary_truncation]
            starts = np.clip(starts, bounds[0], np.inf).astype(int)
            stops = np.clip(stops, -np.inf, bounds[1]).astype(int)

        return starts, stops

    def _get_blockpath(self, block_template, b_idx):
        """Build path to blockfile."""

        if '{b' in block_template:

            bfile = block_template.format(b=b_idx)

        elif '{f' in block_template:

            fp = self.filepaths[b_idx]
            if '.h5' in fp:
                fstem = os.path.basename(fp.split('.h5')[0])
            else:
                fstem = os.path.basename(os.path.splitext(fp)[0])

            bfile = block_template.format(f=fstem)

        return bfile

    def _load_blocks(self, filepaths=''):

        if not filepaths:
            # self.filepaths?
            blockdir = os.path.join(self.datadir, 'blocks')
            filepaths = glob.glob(os.path.join(blockdir, '*.h5'))

        filepaths.sort()

        self._blocks = []
        for fp in filepaths:  # NB/FIXME: requires all blocks to be globbed
            block = Block()
            block.load_blockinfo(fp)
            self._blocks.append(block)

        # Load blocker_info from last block.
        attrs = block.blocker_info
        attr_simple, attr_dict = block._get_blocker_attributes()
        for attr_name in attr_simple + attr_dict:
            setattr(self, attr_name, attrs[attr_name])

    def _prep_paths_blockfiles(self, paths, block, key='blockfiles', reps={}):
        """Format path to blockfile."""

        blockbase = self._blocks[block.idx].path
        filepath = os.path.join(self.datadir, blockbase)
        filestem = os.path.basename(filepath)

        reps['b'] = block.idx
        reps['f'] = filestem

        return self._prep_paths(paths, reps=reps)

    def _set_seamgrid(self, fullsize={}, blocksize={}):
        """Determine the number of seams in the blocked dataset."""
        # TODO?: write to blockfiles?
        # TODO?: generalize and change to blockgrid

        fullsize = fullsize or self.fullsize
        blocksize = blocksize or self.blocksize

        nx = int( np.ceil( fullsize['x'] / blocksize['x'] ) )
        ny = int( np.ceil( fullsize['y'] / blocksize['y'] ) )
        n_seams_yx = [ny - 1, nx - 1]
        seams = list(range(np.prod(n_seams_yx)))
        self._seamgrid = np.reshape(seams, n_seams_yx)

    def view(self, input=[], images=[], labels=[], settings={}):
        """View blocks with napari."""

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, str):
            input = input or self._blocks[0].path
        elif isinstance(input, (int, float)):
            input = self._blocks[input].path
        elif isinstance(input, list):
            input = input or list(range(len(self._blocks)))

        super().view(input, images, labels, settings)


class Splitt3r(Block3r):
    """Block splitting."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'splitter'

        super(Splitt3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'split': self.split,
            })

        self._parallelization.update({
            'split': ['blocks'],
            })

        self._parameter_sets.update({
            'split': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('volumes',),
                'spar': ('_n_workers', 'blocksize', 'blockmargin', 'blocks'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'volumes': {},

            'ods': 'data',
            'squeeze': '',
            'outlayout': '',
            'remove_margins': False,
            'merged_output': False,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_splitter()

        self._init_log()

        self._prep_blocks()

        self._images = ['mean', 'memb/mean', 'nucl/mean']
        self._labels = []

    def _init_paths_splitter(self):

        vols = list(self.volumes.keys())

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        self._paths.update({
            'split': {
                'inputs': {
                    'data': self.inputpaths['blockinfo']['data'],
                    },
                'outputs': {
                    **{'blockfiles': blockfiles},
                    **{ods: f"{blockfiles}/{ods}" for ods in vols},
                    },
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def split(self, **kwargs):
        """Average channels and write as blocks."""

        arglist = self._prep_step('split', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._split_with_combinechannels, arglist)

    def _split_with_combinechannels(self, block_idx):
        """Average channels and write as blocks."""

        block = self._blocks[block_idx]

        inputs = self._prep_paths_blockfiles(self.inputs, block, key='data')
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        block.create_dataset('data')

        self.process_block(block)

    def process_block(self, block):
        """Process datablock."""

        block_ds_in = block.datasets['data']
        block_ds_in.read(from_source=True)
        im = block_ds_in.image

        voldicts = self._get_voldicts(im.get_props2())

        # aggrgegate weighted sums in output['data']
        ch_idxs = set([l for k, v in voldicts.items() for l in v['idxs']])
        for ch_idx in ch_idxs:

            # slice channel as float
            im.slices[im.axlab.index('c')] = slice(ch_idx, ch_idx + 1, None)
            data = im.slice_dataset(squeeze=False).astype('float')

            # add to weighted sum for each volume listing the channel
            for ods, output in voldicts.items():
                if output['output_ND']:
                    continue
                if ch_idx in output['idxs']:
                    data *= output['weights'][output['idxs'].index(ch_idx)]
                    output['data'] += data

        for ods, output in voldicts.items():

            if output['output_ND']:
                mo = im
            else:
                props = im.get_props2()
                props['shape'][props['axlab'].index('c')] = 1
                output['data'] /= len(output['idxs'])
                data = output['data'].astype(output['dtype'])

                props['path'] = ''
                props['shape'] = props['dims'] = data.shape
                props['dtype'] = data.dtype
                props['slices'] = None
                mo = Image(**props)
                mo.create()
                mo.ds[:] = data

            axlab = [al for al in mo.axlab if al not in output['squeeze']]
            axlab = str(''.join(axlab))
            axes = tuple(mo.axlab.index(al) for al in output['squeeze'] if al in mo.axlab)
            block.create_dataset(
                ods,
                dtype=output['dtype'],
                axlab=axlab,
                create_image=True,
                )
            block.datasets[ods].write(np.squeeze(mo.ds, axis=axes))

    def _get_voldicts(self, props):

        vols = {k: v for k, v in self.volumes.items()}  # copy

        for k, ov in vols.items():

            if 'c' in props['axlab']:
                c_axis = props['axlab'].index('c')
                idxs = [i for i in range(props['shape'][c_axis])]
            else:
                print('channel axis not found; not performing averaging')
                c_axis, idxs = None, []

            default = {
                'ods': k,
                'idxs': idxs,
                'weights': [1] * len(idxs),
                'dtype': props['dtype'],
#                'chunksize': props['chunksize'],  # TODO
                'squeeze': '',
                'output_ND': False,
                }
            vols[k] = {**default, **ov}

            shape = [d for d in props['shape']]

            if vols[k]['output_ND']:
                vols[k]['idxs'], vols[k]['weights'] = [], []
            else:
                #props['axlab'].index('c')
                shape[c_axis] = 1
                vols[k]['squeeze'] += 'c'

            vols[k]['data'] = np.zeros(shape, dtype='float')

        return vols

    def view(self, input=[], images=[], labels=[], settings={}):

        images = images or self._images
        labels = labels or self._labels

        if isinstance(input, str):
            input = input or self._blocks[0].path
        elif isinstance(input, (int, float)):
            input = self._blocks[input].path
        elif isinstance(input, list):
            input = input or list(range(len(self._blocks)))

        super().view(input, images, labels, settings)


class Merg3r(Block3r):
    """Block merging."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'merger'

        super(Merg3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'merge': self.merge,
            'postprocess': self.postprocess,
            })

        self._parallelization.update({
            'merge': ['vols'],
            'postprocess': [],
            })

        self._parameter_sets.update({
            'merge': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('fullsize', 'datatype', 'elsize', 'inlayout', 'squeeze'),
                'spar': ('_n_workers', 'blocksize', 'blockmargin', 'blocks', 'vols'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        self._parameter_table.update({})

        default_attr = {
            'volumes': {},
            'datatype': '',
            'elsize': [],
            'inlayout': '',
            'squeeze': '',
            'vols': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_merger()

        self._init_log()

        self._prep_blocks()

        self._images = []
        self._labels = []

    def _init_paths_merger(self):

        vols = list(self.volumes.keys())

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        self._paths.update({
            'merge': {
                'inputs': {
                    **{'blockfiles': blockfiles},
                    },
                'outputs': {
                    'volume': self._build_basename(prefixes=[self.prefix, '{A}'], ext='h5/{a}'),
                    'ulabels': self._build_basename(prefixes=[self.prefix, '{A}'], ext='npy'),
                    },
                },
            'postprocess': {
                'inputs': {
                    'volume': self._build_basename(prefixes=[self.prefix, '{A}'], ext='h5'),
                    },
                'outputs': {
                    'aggregate': self._build_basename(ext='h5'),
                    },
                },
            })

        for step in ['merge', 'postprocess']:  # self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def merge(self, **kwargs):
        """Merge blocks of data into a single hdf5 file.

        Volumes are processed in parallel.
        """

        # from biasfield.py, TODO: deduplicate
        if not 'ims_ref' in self.inputpaths['merge'].keys():
            filepath_ims = self.image_in
            filepath_ref = filepath_ims.replace('.ims', '_ref.ims')
            # TODO: protect existing files
            imarisfiles.create_ref(filepath_ims)
            self.inputpaths['merge']['ims_ref'] = filepath_ref
            self.inputpaths['postprocess']['ims_ref'] = filepath_ref

        arglist = self._prep_step('merge', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._mergeblocks, arglist)

    def _mergeblocks(self, vol_idx):
        """Merge blocks of data into a single hdf5 file."""

        volname = list(self.volumes)[vol_idx]
        volume = list(self.volumes.values())[vol_idx]
        is_labelimage = ('is_labelimage' in volume.keys()) and volume['is_labelimage']

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs, reps={'a': volname, 'A': volname.replace('/', '-')})

        # Select a subset of blocks
        self.blocks = self.blocks or list(range(len(self._blocks)))


        # CASES:
        # - blocklayout to identical layout
        # - new axes
        # - squeezed axes in block-data
        # - squeezed axes in output data


        """
        # FIXME: squeezed blocks with dim=3 (eg split over T) have 4d elsize but 3d axlab

        merge_axes = [al for al, fs in self.fullsize.items() if self.blocksize[al] <= fs]

        dims = self.fullsize

        # FIXME: im can be squeezed
        # dims = [self.fullsize[al] if al in merge_axes else props['dims'][im.axlab.index(al)]
        #         for al in props['axlab']]
        # dims = [self.fullsize[al] if al in merge_axes else props['dims'][block0.axlab.index(al)]
        #         for al in props['axlab']]

        # TESTING 5D to 3D
        merge_axes = ['z', 'y', 'x']  # FIXME

        #props['chunks'] = [1, 37, 64, 64]
        #props['compression'] = None

        print(props)
        self.squeeze = 'ct'
        for ax in self.squeeze:
            props = im.squeeze_props(props, dim=props['axlab'].index(ax))
        print(props)

        #im = Image(inputs['data'], permission='r')
        #im.load()
        #_, squeezed_idxs = self._find_squeezed_dims(im)
        #im.close()

        block0 = self._blocks[self.blocks[0]]
        im = Image(block0.path.format(ods=volname), permission='r')
        im.load()
        props = im.get_props2()
        im.close()

        props['path'] = outputs[volname]  #f'{outstem}.h5/{volname}'
        props['permission'] = 'r+'
        props['slices'] = None
        props['chunks'] = None

        props['axlab'] = 'zyx'
        props['dims'] = props['shape'] = [self.fullsize[al] for al in props['axlab']]

        self.squeeze = 'ct'
        for ax in self.squeeze:
            props = im.squeeze_props(props, dim=props['axlab'].index(ax))
        print(props)
        """

        props = self._get_outputprops(volname, outputs['volume'])

        merge_axes = ['z', 'y', 'x']

        if '.h5' in props['path']:
            mo = LabelImage(**props)
            mo.create()
        elif props['path'].endswith('.ims'):
            shutil.copy2(inputs['ims_ref'], outputs['volume'])
            mo = Image(outputs['volume'])
            mo.load()

        # Merge the blocks sequentially (TODO: reintroduce MPI with h5-para, see also backproject.py).
        ulabels = set([])
        for block_idx in self.blocks:
            try:

                # Read block
                block = self._blocks[block_idx]
                block.create_dataset(volname)
                block_ds_in = block.datasets[volname]
                block_ds_in.read(from_block=True)
                im = block_ds_in.image

                for al in im.axlab:
                    im.slices[im.axlab.index(al)] = block_ds_in.slices_region_blockfile[al]
                data = im.slice_dataset(squeeze=False)

                # Set slices on output volume
                for al in mo.axlab:
                    mo.slices[mo.axlab.index(al)] = block_ds_in.slices_region[al]

#                if props['path'].endswith('.ims'):
#                    data *= 65535  # TODO: integrate with Backproject3r

                #data = np.expand_dims(data, axis=squeezed_idxs)
                mo.write(data.astype(mo.dtype))
                im.close()

            except FileNotFoundError:
                print(f'{blockpath} not found')

            if is_labelimage:
                ulabels |= set(np.unique(data))

        if is_labelimage:
            np.save(outputs['ulabels'], np.array(ulabels))
            mo.ds.attrs['maxlabel'] = max(ulabels)

        im.close()
        mo.close()

    def set_slices(self, im, mo, block, merge_axes=''):
        """Set the block's slices on the input block and output volume.

        for the axes which need to be merged
        # TODO: handle shift_final_block_inward
        """

        merge_axes = merge_axes or block.axlab
#        for al in merge_axes:
        for al in block.axlab:
            # TODO: need to set ALL axes of mo, also potential inserts
            if al in merge_axes:
                l = block.slices[al].start
                u = block.slices[al].stop
                (ol, ou), (il, iu) = self._margins(
                    l, u,
                    self.blocksize[al],
                    self.blockmargin[al],
                    self.fullsize[al],
                )
                im.slices[im.axlab.index(al)] = slice(il, iu)
                mo.slices[mo.axlab.index(al)] = slice(ol, ou)
            else:
                slc = block.slices[al]
                if al in im.axlab:
                    im.slices[im.axlab.index(al)] = slc
                #mo.slices[mo.axlab.index(al)] = slc

    def _get_merge_axes(self, im, block, merge_axes=''):
         for al in im.axlab:
            # 1. check if the volume was split over dimension K
            ax_idx = block.axlab.index(al)
            axis_blocked = block.slices[ax_idx].stop != self.fullsize[al]
            # 2. check if merging is desired (default: True if axis_blocked)
            if axis_blocked:
                merge_axes += al
         return merge_axes

    def _get_outputprops(self, volname, outputpath):

        # Properties of the input blocks
        block0 = self._blocks[self.blocks[0]]
        path = f'{block0.path}/{volname}'
        im = Image(path, permission='r')
        im.load()
        props = im.get_props2()
        im.close()

        props['path'] = outputpath
        props['permission'] = 'r+'
        props['slices'] = None
        props['chunks'] = None

        # TODO: add potential inserts, remove squeezes
        #props['axlab'] = 'zyx'  # FIXME:
        props['dims'] = props['shape'] = [self.fullsize[al] for al in props['axlab']]

        return props

    def postprocess(self, **kwargs):
        """Merge bias field estimation files.

        TODO: deduplicate from biasfield.py
        """

        # from biasfield.py, TODO: deduplicate
        if not 'ims_ref' in self.inputpaths['postprocess'].keys():
            filepath_ims = self.image_in
            filepath_ref = filepath_ims.replace('.ims', '_ref.ims')
            # TODO: protect existing files
            imarisfiles.create_ref(filepath_ims)
            self.inputpaths['postprocess']['ims_ref'] = filepath_ref

        self._prep_step('postprocess', kwargs)

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        volumes = {}
        vols = self.vols or list(range(len(self.volumes)))
        for vol_idx in vols:
            volname = list(self.volumes)[vol_idx]
            inputs = self._prep_paths(self.inputs, reps={'a': volname, 'A': volname.replace('/', '-')})
            volumes[volname] = inputs['volume']

        outputfile = outputs['aggregate']
        if outputfile.endswith('.ims'):
            inputpaths = list(volumes.values())
            inputpaths.sort()
            imarisfiles.ims_linked_channels(outputfile, inputpaths, inputs['ims_ref'])
        elif '.h5' in outputfile:
            channels = [{'Name': volname, 'filepath': inputpath}
                        for volname, inputpath in volumes.items()]
            imarisfiles.aggregate_h5_channels(outputfile, channels)

    def get_outputpath(self, ids, volume, ext='', fullpath=True):

        try:
            suf = volume['suffix'].replace('/', '-')
        except:
            suf = ids.replace('/', '-')

        outstem = self._build_filestem(prefixes=[self.prefix, suf], use_fallback=False)

        if not ext:
            ext = volume['format'] if 'format' in volume.keys() else 'h5'

        if ext == 'h5' and fullpath:
            outputpath = f"{outstem}.{ext}/{ids}"
        else:
            outputpath = f"{outstem}.{ext}"

        return outputpath

    def view(self, input=[], images=[], labels=[], settings={}):

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, str):
            input = input
        elif isinstance(input, (int, float)):
            volname = list(self._volumes[input].keys())[0]
            input = self.outputpaths['merge'][volname]
        # elif isinstance(input, list):
        #     input = input or list(range(len(self._blocks)))
        # TODO: postprocess
        # TODO: multiple volumes overlaid

        super().view(input, images, labels, settings)


if __name__ == "__main__":
    main(sys.argv[1:])
