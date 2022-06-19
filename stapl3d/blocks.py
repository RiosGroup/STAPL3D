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

from stapl3d import parse_args, Stapl3r, Image, LabelImage

import glob
import h5py
import pathlib

logger = logging.getLogger(__name__)


def main(argv):
    """Block operations."""

    steps = ['split', 'merge']
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
    """Block.

    TODO: save blockgrid and slice into blockgrid (to easily identify boundary blocks)
    TODO: include chunks?, original path, Image?
    """

    def __init__(self, id='', idx=0, path='', slices={}, axlab='', elsize=[], blocker_info={}, affine=[]):

        self.id = id
        self.idx = idx

        self.path = path
        self.slices = slices

        self.axlab = axlab
        self.elsize = elsize

        self.blocker_info = blocker_info

        # TODO: integrate with affine matrix on Image class
        if affine:
            self.affine = affine
        else:  # default: only block translation in voxel space
            self.affine = np.eye(4)
            slices_sel = [slc for slc, al in zip(slices, axlab) if al in 'xyz']
            for i, slc in enumerate(slices_sel):
                self.affine[i, 3] = slc.start

    def __str__(self):
        return yaml.dump(vars(self), default_flow_style=False)


class Block3r(Stapl3r):
    """Block operations."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'blocks'

        super(Block3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector = {
            'blockinfo': self.print_blockinfo,
            }

        self._parallelization = {
            'blockinfo': ['blocks'],
            }

        self._parameter_sets = {
            'blockinfo': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('fullsize', 'blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'blocks'),
                },
            }

        self._parameter_table = {
            'fullsize': 'Size of the full dataset',
            'blocksize': 'Block size',
            'blockmargin': 'Block margin',
            'truncate_to_dataset': 'Truncate boundary blocks to dataset',
            'truncate_to_margin': 'Truncate boundary blocks to include margin',
            'shift_final_block_inward': 'Shift the upper-bound blocks inward',
        }

        default_attr = {
            'fullsize': {},
            'blocksize': {},
            'blockmargin': {},
            'truncate_to_dataset': True,
            'truncate_to_margin': False,
            'shift_final_block_inward': False,
            'pad_mode': 'constant',
            'pad_kwargs': {},
            'blocks': [],
            'squeeze': '',
            '_elsize': {},
            '_blocks': [],
            '_blocks0': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._prep_blocks()

        self._images = []
        self._labels = []

    def _init_paths(self):
        """Set input and output filepaths.

        REQUIRED:
        blockinfo:inputs:data           input image OR {f}.<ext>
        blockinfo:outputs:blockfiles    blockpattern ...{b}... OR ...{f}...
        """

        if '{f' in self.image_in:
            prefixes, suffix = [''], 'f'
        else:
            prefixes, suffix = [self.prefix, 'blocks'], 'b'
        bpat = self._build_path(moduledir='blocks', prefixes=prefixes, suffixes=[{suffix: 'p'}])

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

    def print_blockinfo(self, **kwargs):
        """Print the slices into the full dataset for all blocks."""

        arglist = self._prep_step('blockinfo', kwargs)
        for i, block in enumerate(self._blocks):
            if i in self.blocks:
                print(f'Block {i:05d} with id {block.id} refers to region {block.slices}')

    def _prep_blocks(self):
        """Set blocker characteristics and blocks.

        _blocks0 contains a blocker version without the margins
        """

        step = 'blockinfo'
        datafiles = self.inputpaths[step]['data']
        blockfiles = self.outputpaths[step]['blockfiles']

        if '{b' in datafiles or '{f' in datafiles:  # files as blocks
            self.filepaths = self.get_filepaths(datafiles)
            self.set_fullsize(self.filepaths[0])  # FIXME: only correct for firstfile
        else:  # blocks from grid
            self.filepaths = []
            self.set_fullsize(datafiles)

        self.set_blocksize()
        self.set_blockmargin()

        self._blocks = self.generate_blocks(blockfiles)
        self._blocks0 = self.generate_blocks(blockfiles, blocks0=True)

    def set_fullsize(self, image_in, fullsize={}):
        """Set the shape of the full dataset."""

        try:
            im = Image(image_in, permission='r')
            im.load(load_data=False)
            im.close()
            imsize = dict(zip(im.axlab, im.dims))
            self._elsize = dict(zip(im.axlab, im.elsize))
        except:
            imsize = {}

        self.fullsize = {**imsize, **self.fullsize, **fullsize}

    def set_blocksize(self, blocksize={}):
        """Set the size of the block."""

        self.blocksize = {**self.fullsize, **self.blocksize, **blocksize}

    def set_blockmargin(self, blockmargin={}):
        """Set the margins of the block."""

        bm_im = {d: 0 for d in self.blocksize.keys()}
        self.blockmargin = {**bm_im, **self.blockmargin, **blockmargin}

    def generate_blocks(self, block_template, blocks0=False):
        """Create Block objects."""

        axlab = [al for al in self.fullsize.keys()]
        elsize = [self._elsize[al] for al in axlab] if self._elsize else [1] * len(axlab)

        margin = dict(zip(axlab, [0] * len(axlab))) if blocks0 else self.blockmargin

        if self.filepaths:
            starts, stops = self.get_bounds_for_files(margin)
        else:
            starts, stops = self.get_bounds_for_grid(margin)

        return self._generate_blocks(starts, stops, block_template, axlab, elsize)

    def get_bounds_for_grid(self, margin):
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

    def get_bounds_for_files(self, margin):
        """Generate slices for a list of files."""

        stops = []
        for filepath in self.filepaths:
            if '.csv' in filepath:  # test for featur3r: TODO: generalize
                stops.append([0, 0, 0])
            else:
                im = Image(filepath)
                im.load()
                stops.append(im.dims)
                im.close()

        starts = [[0] * len(stop) for stop in stops]

        return starts, stops

    def _generate_blocks(self, starts, stops, block_template, axlab, elsize):
        """"""

        blocker_info = {k:v for k, v in vars(self).items()
                        if k in self._parameter_table.keys()}

        blocks = []
        for b_idx, (start, stop) in enumerate(zip(starts, stops)):
            block = Block(
                id=self._suffix_formats['b'].format(b=b_idx),
                idx=b_idx,
                path=self._get_blockpath(block_template, b_idx),
                slices=[slice(sta, sto) for sta, sto in zip(start, stop)],
                axlab=axlab,
                elsize=elsize,
                blocker_info=blocker_info,
                )
            blocks.append(block)

        return blocks

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

        return bfile + '/{ods}'

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
        if self.truncate_to_dataset:
            bounds = [ds_start, ds_stop]
        elif self.truncate_to_margin:
            bounds = [ds_start - margin, ds_stop + margin]

        if self.truncate_to_dataset or self.truncate_to_margin:
            starts = np.clip(starts, bounds[0], np.inf).astype(int)
            stops = np.clip(stops, -np.inf, bounds[1]).astype(int)

        return starts, stops

    def _margins(self, fc, fC, blocksize, margin, fullsize):
        """Return coordinates (fullstack and block) corrected for margin.

        fc and fC are start and stop of the slice in the dataset voxel space
        """

        final_block = fC >= fullsize
        if self.shift_final_block_inward and final_block:
            bc = margin
            bC = blocksize - margin  # always full blocksize
            fc = fullsize
            fC = fullsize - blocksize
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

    def get_filepaths(self, filepat):
        """Set the filepaths by globbing the directory."""

        if not os.path.isabs(filepat):
            filepat = os.path.join(self.datadir, filepat)

        def glob_h5(directory, s):

            if '.h5' in s:
                h5stem, ids = s.split('.h5')
                s = f'{h5stem}.h5'
            else:
                ids = ''

            fmat = os.path.join(directory, s)
            filepaths = sorted(glob.glob(fmat))
            filepaths = [f'{filepath}{ids}' for filepath in filepaths]

            return filepaths

        if '.h5' in filepat:
            p = filepat.split('.h5')
            dir, base = os.path.split(p[0])
            fp = '{}.h5{}'.format(base, p[1])
            return glob_h5(os.path.abspath(dir), self._pat2mat(fp))
            # full h5 path???
        elif '.csv' in self.image_in:  # test for featur3r: TODO: generalize
            fps = sorted(glob.glob(os.path.abspath(self._pat2mat(filepat))))
            suffix = self.image_in.split('{f}')[-1]
            return [fp.replace(suffix, '.csv') for fp in fps]
        else:
            return sorted(glob.glob(os.path.abspath(self._pat2mat(filepat))))

    def _prep_paths_blockfiles(self, paths, block, key='blockfiles', reps={}):
        """Format path to blockfile."""

        blockbase = self._blocks[block.idx].path.replace('.h5/{ods}', '')
        filepath = os.path.join(self.datadir, blockbase)
        filestem = os.path.basename(filepath)

        reps['b'] = block.idx
        reps['f'] = filestem

        return self._prep_paths(paths, reps=reps)

    def get_padding(self, block, dims):
        """Return pad_width from slices overshoot and truncated slices."""

        padding, slices = [], []

        for bslc, al, m in zip(block.slices, block.axlab, dims):

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

    def _find_squeezed_dims(self, im=None, squeezed=''):
        """Determine dimensions that are squeezed in the blocks."""

        if not self.squeeze:
            pass

        elif isinstance(self.squeeze, str):
            squeezed = self.squeeze

        elif isinstance(self.squeeze, bool):
            for al in im.axlab:
                if self.blocksize[al] == 1 and self.blockmargin[al] == 0:
                    squeezed += al

        return squeezed, tuple(im.axlab.index(al) for al in squeezed)

    def read_block(self, im, block):
        """Read block data as Image object with the same axes layout."""

        pad_width, slices = self.get_padding(block, im.dims)

        im.slices = slices
        data = im.slice_dataset(squeeze=False)

        if pad_width is not None:
            data = np.pad(data, pad_width, mode=self.pad_mode, **self.pad_kwargs)

        ref_props = im.get_props2()

        return self._create_block_image(data, ref_props)

    def _create_block_image(self, data, props, path=''):
        """Create Image object from data and reference Image (internal use)."""

        props['path'] = path
        props['shape'] = data.shape
        props['dims'] = data.shape
        props['dtype'] = data.dtype
        props['slices'] = None

        im = Image(**props)
        im.create()
        im.ds[:] = data

        return im

    def write_block(self, im, block, ods='data', squeeze='', outlayout='', remove_margins=False, merged_output=False):
        """Write a block to (merged) datafile."""

        # TODO: squeeze if self.squeeze=True and dim=1?
        squeeze = set(list(squeeze))
        if outlayout:
            squeeze &= set(list(im.axlab)) - set(list(outlayout))
        squeeze = ''.join(squeeze)
        if squeeze:
            im.squeeze(squeeze)

        if outlayout:  # TODO: sanity checking
            im.transpose(outlayout)

        # default to full range for all dims
        slcs_block = {al: slice(0, sh) for al, sh in zip(im.axlab, im.dims)}
        slcs_full = {al: slice(0, sh) for al, sh in zip(im.axlab, im.dims)}

        if remove_margins or merged_output:  # set to margins for all dims in block.axlab
            slcs_block, slcs_full = self.get_slices(block, slcs_block, slcs_full)

        slcs_block = tuple(slcs_block[al] for al in im.axlab)
        slcs_full = tuple(slcs_full[al] for al in im.axlab)

        im.slices = slcs_block
        data = im.slice_dataset(squeeze=False)

        if merged_output:
            # TODO!
            mo = Image('test_out.h5/data', permission='r+')
            mo.load() # comm=None
            mo.slices = tuple(slcs_full[al] for al in mo.axlab)
            mo.write(data.astype(mo.dtype))
            mo.close()

        else:

            props = self._find_outputprops_blocks(im, block, ods, slcs_block)
            self.create_datafile(props, squeeze, data=data.astype(props['dtype']))

    def get_slices(self, block, slcs_in, slcs_out):

        merge_axes = [al for al, fs in self.fullsize.items() if self.blocksize[al] >= fs]
        for al in block.axlab:
            l = block.slices[block.axlab.index(al)].start
            u = block.slices[block.axlab.index(al)].stop
            (ol, ou), (il, iu) = self._margins(
                l, u,
                self.blocksize[al],
                self.blockmargin[al],
                self.fullsize[al],
            )
            slcs_in[al] = slice(il, iu)  # into the block
            slcs_out[al] = slice(ol, ou)  # into the full dataset

        return slcs_in, slcs_out

    def _create_output_merged(self):
        """..."""

        self.block = self._blocks[0]

        inputs = self._prep_paths_blockfiles(self.inputpaths['predict'], self.block, key='data')
        outputs = self._prep_paths_blockfiles(self.outputs, self.block)

        self.input_image = Image(inputs['data'], permission='r')
        self.input_image.load()
        props = self._find_outputprops(self.input_image, self.block)
        props['path'] = 'test_out.h5/data'
        props['dims'] = props['shape'] = [self.fullsize[al] for al in props['axlab']]
        squeezed, _ = self._find_squeezed_dims(self.input_image)
        self.input_image.close()

        self.create_datafile(props)

        self.input_image, self.output_image, self.block = None, None, None

    def _find_outputprops(self, im, block):

        props = im.get_props2()
        props['permission'] = 'r+'
        props['path'] = os.path.join(self.datadir, block.path.format(ods='data'))  # outputs['data']  # FIXME
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = self.chunksize or [max(64, self.blockmargin[al])
                                             if al in 'xy' else s
                                             for al, s in zip(im.axlab, im.dims)]
        props['shape'] = list(im.slices2shape(list(block.slices)))
        props['dims'] = list(im.slices2shape(list(block.slices)))
        props['slices'] = None

        return props

    def _find_outputprops_blocks(self, im, block, ods='data', slices=None):

        # reset slices to full block
        if slices is None:
            im.slices = None
            im.set_slices()
        else:
            im.slices = slices

        props = im.get_props2()

        props['permission'] = 'r+'

        props['path'] = os.path.join(self.datadir, block.path.format(ods=ods))

        props['dtype'] = self.datatype or props['dtype']

        default_chunks = self._default_chunks(dict(zip(im.axlab, im.dims)))
        props['chunks'] = self.chunksize or default_chunks

        props['shape'] = props['dims'] = list(im.slices2shape(im.slices))

        props['slices'] = None

        return props

    def create_datafile(self, props, squeeze='', data=None):

        mo = Image(**props)
        mo.squeeze(squeeze)
        mo.create()
        if data is not None:
            mo.write(data.astype(mo.dtype))
        mo.close()

    def _default_chunks(self, dims, min_chunksize_zyx=32, full_dimension=True):

        chunksize = [max(min_chunksize_zyx, self.blockmargin[al])
                     if al in 'zyx' else s if full_dimension else 1
                     for al, s in dims.items()]

        return chunksize

    def view(self, input=[], images=[], labels=[], settings={}):
        """View blocks with napari."""

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, str):
            input = input or self._blocks[0].path.replace('/{ods}', '')
        elif isinstance(input, (int, float)):
            input = self._blocks[input].path.replace('/{ods}', '')
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
                'ppar': ('volumes', 'output_ND', 'datatype', 'chunksize'),
                'spar': ('_n_workers', 'blocksize', 'blockmargin', 'blocks'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'volumes': {},
            'output_ND': False,
            'datatype': '',  # TODO
            'chunksize': [],  # TODO

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
        vols += ['data'] if self.output_ND else []

        bpat = self._build_path(moduledir='blocks',
                                prefixes=[self.prefix, 'blocks'],
                                suffixes=[{'b': 'p'}])

        blockfiles = self.outputpaths['blockinfo']['blockfiles']

        self._paths.update({
            'split': {
                'inputs': {
                    'data': self.inputpaths['blockinfo']['data'],
                    'bias': '',
                    },
                'outputs': {
                    **{'blockfiles': blockfiles},
                    **{ods: f"{blockfiles}/{ods}" for ods in vols},
                    },
                },
            })

        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

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

        self.input_image = Image(inputs['data'], permission='r')
        self.input_image.load()

        # FIXME: it reads the whole block, which may not be required
        im = self.read_block(self.input_image, block)

        mos = self.process_block(im)

        for volname, (mo, outargs) in mos.items():
            self.write_block(mo, block, volname, **outargs)

        self.input_image.close()

    def process_block(self, im):
        """Process datablock."""

        mos = {}

        if self.output_ND:
            outargs = {
                'squeeze': '',
                'outlayout': '',
                'remove_margins': self.remove_margins,
                'merged_output': self.merged_output,
                }
            mos['data'] = (im, outargs)

        if not self.volumes:
            return mos

        if 'c' not in im.axlab:
            print('No channel axis label; cannot perform channel averaging')
            return

        outargs = {
            'squeeze': 'ct',
            'outlayout': '',
            'remove_margins': self.remove_margins,
            'merged_output': self.merged_output,
            }

        c_axis = im.axlab.index('c')
        ref_props = im.get_props()
        ref_props['shape'][c_axis] = 1

        voldicts = self._get_voldicts(im, ref_props)

        # aggrgegate weighted sums in output['data']
        ch_idxs = set([l for k, v in voldicts.items() for l in v['idxs']])
        for ch_idx in ch_idxs:
            im.slices[c_axis] = slice(ch_idx, ch_idx + 1, None)
            data = im.slice_dataset(squeeze=False).astype('float')
            for name, output in voldicts.items():
                if ch_idx in output['idxs']:
                    data *= output['weights'][output['idxs'].index(ch_idx)]
                    output['data'] += data

        for name, output in voldicts.items():
            output['data'] /= len(output['idxs'])
            result = output['data'].astype(ref_props['dtype'])
            mo = self._create_block_image(result, ref_props)
            mos[output['ods']] = (mo, outargs)

        return mos

    def _get_voldicts(self, im, props):

        idxs = [i for i in range(im.dims[im.axlab.index('c')])]
        vols = {k: v for k, v in self.volumes.items()}
        for k, ov in vols.items():
            default = {
                'ods': k,
                'idxs': idxs,
                'weights': [1] * len(idxs),
                'dtype': self.datatype or props['dtype'],
                'data': np.zeros(props['shape'], dtype='float'),
                }
            vols[k] = {**default, **ov}

        return vols

    def view(self, input=[], images=[], labels=[], settings={}):

        images = images or self._images
        labels = labels or self._labels

        if isinstance(input, str):
            input = input or self._blocks[0].path.replace('/{ods}', '')
        elif isinstance(input, (int, float)):
            input = self._blocks[input].path.replace('/{ods}', '')
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
            'merge': ['volumes'],
            'postprocess': [],
            })

        self._parameter_sets.update({
            'merge': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('fullsize', 'datatype', 'elsize', 'inlayout', 'squeeze'),
                'spar': ('_n_workers', 'blocksize', 'blockmargin', 'blocks', 'volumes'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        self._parameter_table.update({})

        default_attr = {
            'volumes': [],
            'datatype': '',
            'elsize': [],
            'inlayout': '',
            'squeeze': '',
            '_volumes': {},
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

        bpat = self._build_path(suffixes=[{'b': 'p'}])
        bmat = self._build_path(suffixes=[{'b': '?'}])

        # I don't think this used: blockfiles are used
        self._paths.update({
            'merge': {
                'inputs': {
                    # 'data': ,  # TODO: template...
                    ods: f"{bmat}.{vol['format'] if 'format' in vol.keys() else 'h5'}/{ods}" for volume in self._volumes for ods, vol in volume.items()
                    },
                'outputs': {
                    **{ods: self.get_outputpath(volume) for volume in self._volumes for ods, vol in volume.items()},
                    **{f'{ods}_ulabels': self.get_outputpath(volume, 'npy') for volume in self._volumes for ods, vol in volume.items() if 'is_labelimage' in vol.keys() and vol['is_labelimage']},
                    },
                },
            'postprocess': {
                'inputs': {
                    ods: self.get_outputpath(volume, fullpath=False) for volume in self._volumes for ods, vol in volume.items()
                    },
                'outputs': {
                    'aggregate': self._build_path(prefixes=['merged'], ext='h5'),
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

        arglist = self._prep_step('merge', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._mergeblocks, arglist)

    def _mergeblocks(self, volume_idx):
        """Merge blocks of data into a single hdf5 file."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        volume = self._volumes[volume_idx]
        volname = list(volume.keys())[0]

        ukey = f'{volname}_ulabels'
        ulabelpath = outputs[ukey] if ukey in outputs.keys() else ''

        # Select a subset blocks
        self.blocks = self.blocks or list(range(len(self._blocks)))




        # CASES:
        # - blocklayout to identical layout
        # - new axes
        # - squeezed axes in block-data
        # - squeezed axes in output data



        # Properties of the input blocks



        block0 = self._blocks[self.blocks[0]]
        im = Image(block0.path.format(ods=volname), permission='r')
        im.load()
        props = im.get_props2()
        im.close()

        props['path'] = outputs[volname]  #f'{outstem}.h5/{volname}'
        props['permission'] = 'r+'
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = props['chunks'] or None
        # FIXME: squeezed blocks with dim=3 (eg split over T) have 4d elsize but 3d axlab
        # props['axlab'] = self.inlayout or props['axlab']
        # props['elsize'] = self._elsize or props['elsize']

        props['axlab'] = ''.join(block0.axlab)
        props['elsize'] = block0.elsize


        merge_axes = [al for al, fs in self.fullsize.items() if self.blocksize[al] >= fs]

        dims = self.fullsize

        # FIXME: im can be squeezed
        # dims = [self.fullsize[al] if al in merge_axes else props['dims'][im.axlab.index(al)]
        #         for al in props['axlab']]


        # dims = [self.fullsize[al] if al in merge_axes else props['dims'][block0.axlab.index(al)]
        #         for al in props['axlab']]

        dims = [self.fullsize[al] for al in props['axlab']]

        props['dims'] = props['shape'] = dims
        props['slices'] = None
        props['chunks'] = None



        for ax in self.squeeze:
            props = im.squeeze_props(props, dim=props['axlab'].index(ax))


        im = Image(inputs['data'], permission='r')
        im.load()
        _, squeezed_idxs = self._find_squeezed_dims(im)
        im.close()


        mo = LabelImage(**props)
        mo.create()

        # Merge the blocks sequentially (TODO: reintroduce MPI with h5-para).
        ulabels = set([])
        for block_idx in self.blocks:
            block = self._blocks[block_idx]
            blockpath = block.path.format(ods=volname)
            print('processing volume {} block {} from {}'.format(volname, block.id, blockpath))
            im = Image(blockpath, permission='r')
            im.load(load_data=False)
            self.set_slices(im, mo, block, merge_axes)
            data = im.slice_dataset(squeeze=False)
            data = np.expand_dims(data, axis=squeezed_idxs)
            mo.write(data)
            im.close()

            if ulabelpath:
                ulabels |= set(np.unique(data))

        if ulabelpath:
            np.save(ulabelpath, np.array(ulabels))
            mo.ds.attrs['maxlabel'] = max(ulabels)

        im.close()
        mo.close()

    def set_slices(self, im, mo, block, merge_axes=''):
        """Set the block's slices on the input block and output volume.

        for the axes which need to be merged
        # TODO: handle shift_final_block_inward
        """

        merge_axes = merge_axes or im.axlab
        for al in merge_axes:
            l = block.slices[block.axlab.index(al)].start
            u = block.slices[block.axlab.index(al)].stop
            (ol, ou), (il, iu) = self._margins(
                l, u,
                self.blocksize[al],
                self.blockmargin[al],
                self.fullsize[al],
            )
            im.slices[im.axlab.index(al)] = slice(il, iu)
            mo.slices[mo.axlab.index(al)] = slice(ol, ou)

    # def _get_merge_axes(self, im, block, merge_axes=''):
    #     """Set the block's slices on the input block and output volume."""
    #
    #     for al in im.axlab:
    #         # 1. check if the volume was split over dimension K
    #         ax_idx = block.axlab.index(al)
    #         axis_blocked = block.slices[ax_idx].stop != self.fullsize[al]
    #         # 2. check if merging is desired (default: True if axis_blocked)
    #         if axis_blocked:
    #             merge_axes += al
    #
    #     return merge_axes

    def postprocess(self):

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

        self._prep_step('postprocess', {})

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        tgt_file = outputs['aggregate']
        tgt_dir  = os.path.dirname(tgt_file)
        f = h5py.File(tgt_file, 'w')
        for volume in self.volumes:
            ids = list(volume.keys())[0]
            # linked_path = inputs[ids]  # absolute path
            linked_path = os.path.relpath(inputs[ids], tgt_dir)  # relative path
            ext_file = pathlib.Path(linked_path).as_posix()
            create_ext_link(f, ids, ext_file, ids)

    def get_outputpath(self, volume, ext='', fullpath=True):

        ids = list(volume.keys())[0]
        try:
            suf = volume[ids]['suffix'].replace('/', '-')
        except:
            suf = ids.replace('/', '-')

        outstem = self._build_filestem(prefixes=[self.prefix, suf], use_fallback=False)

        if not ext:
            ext = volume[ids]['format'] if 'format' in volume[ids].keys() else 'h5'

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
