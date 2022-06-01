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

from stapl3d import parse_args, Stapl3r, Image, LabelImage, format_, split_filename

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
    """Block."""

    def __init__(self, id='', idx=0, path='', slices={}, axlab='', elsize=[], blocker_info={}):

        self.id = id
        self.idx = idx

        self.path = path
        self.slices = slices

        self.axlab = axlab
        self.elsize = elsize

        self.blocker_info = blocker_info

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
            'blocksize': {},
            'blockmargin': {},
            'fullsize': {},
            'truncate_to_dataset': True,
            'truncate_to_margin': False,
            'shift_final_block_inward': False,
            'elsize': {},
            'blocks': [],
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

        os.makedirs('blocks', exist_ok=True)
        bpat = self._build_path(moduledir='blocks', prefixes=[self.prefix, 'blocks'], suffixes=[{'b': 'p'}])
        # bpat = self._build_path(suffixes=[{'b': 'p'}])

        self._paths = {
            'blockinfo': {
                'inputs': {
                    'data': self.image_in,  # TODO: via inputpath/biasfield output
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

        arglist = self._prep_step('blockinfo', kwargs)
        for i, block in enumerate(self._blocks):
            if i in self.blocks:
                print(f'Block {i:05d} with id {block.id} refers to region {block.slices}')

    def _prep_paths_blockfiles(self, paths, block, key='blockfiles', reps={}):

        filepath = self._blocks[block.idx].path.replace('.h5/{ods}', '')
        filestem = os.path.basename(filepath)

        reps['b'] = block.idx
        reps['f'] = filestem

        return self._prep_paths(paths, reps=reps)

    def _prep_blocks(self):

        step = 'blockinfo'  #self.step
        inpaths = self.inputpaths[step]['data']

        if '{b' in inpaths or '{f' in inpaths:
            self.filepaths = self.get_filepaths(inpaths)
            self.set_fullsize(self.filepaths[0])
        elif os.path.isdir(self.image_in):
            self.filepaths = sorted(glob.glob(os.path.join(self.image_in, '*.czi')))  # TODO: flexible extension
            self.set_fullsize(self.filepaths[0])
        else:
            self.filepaths = []
            self.set_fullsize(inpaths)

        self.set_blocksize()
        self.set_blockmargin()

        if '{f' in inpaths:
            self.outputpaths[step]['blockfiles'] = os.path.join('blocks', '{f}.h5')

        blockfiles = self.outputpaths[step]['blockfiles']

        self._blocks = self.generate_blocks(blockfiles)
        self._blocks0 = self.generate_blocks(blockfiles, blocks0=True)

    def set_fullsize(self, image_in, fullsize={}):

        try:
            im = Image(image_in, permission='r')
            im.load(load_data=False)
            im.close()
            imsize = dict(zip(im.axlab, im.dims))
            self.elsize = dict(zip(im.axlab, im.elsize))
        except:
            imsize = {}

        self.fullsize = {**imsize, **self.fullsize, **fullsize}

    def set_blocksize(self, blocksize={}):

        self.blocksize = {**self.fullsize, **self.blocksize, **blocksize}

    def set_blockmargin(self, blockmargin={}):

        bm_im = {d: 0 for d in self.blocksize.keys()}
        self.blockmargin = {**bm_im, **self.blockmargin, **blockmargin}

    def generate_blocks(self, block_template, blocks0=False):

        if self.filepaths:
            blocks = self.get_blocks_from_files(block_template, blocks0)
        else:
            blocks = self.get_blocks_from_grid(block_template, blocks0)

        return blocks

    def get_blocks_from_grid(self, block_template, blocks0=False):

        axlab = [al for al in self.fullsize.keys()]
        elsize = [self.elsize[al] for al in axlab] if self.elsize else [1] * len(axlab)

        imslices = {al: slice(0, self.fullsize[al]) for al in axlab}
        shape = {al: len(range(*slc.indices(slc.stop))) for al, slc in imslices.items()}
        blocksize = self.blocksize or shape
        blocksize = {al: bs if bs else shape[al] for al, bs in blocksize.items()}

        margin = dict(zip(axlab, [0] * len(axlab))) if blocks0 else self.blockmargin

        starts, stops, = {}, {}
        for al in axlab:
            starts[al], stops[al] = self._get_blockbounds(
                imslices[al].start,
                imslices[al].stop,
                blocksize[al],
                margin[al],
                self.truncate_to_dataset,
                self.truncate_to_margin,
                self.shift_final_block_inward,
                )

        ndim = len(axlab)
        starts = tuple(starts[dim] for dim in axlab)
        stops = tuple(stops[dim] for dim in axlab)
        startsgrid = np.array(np.meshgrid(*starts))
        stopsgrid = np.array(np.meshgrid(*stops))
        starts = np.transpose(np.reshape(startsgrid, [ndim, -1]))
        stops = np.transpose(np.reshape(stopsgrid, [ndim, -1]))

        blocks = []
        for b_idx, (start, stop) in enumerate(zip(starts, stops)):
            block = Block(
                id=self._suffix_formats['b'].format(b=b_idx),
                idx=b_idx,
                path=block_template.format(b=b_idx) + '/{ods}',
                slices=[slice(sta, sto) for sta, sto in zip(start, stop)],
                axlab=axlab,
                elsize=elsize,
                blocker_info={k:v for k, v in vars().items() if k in self._parameter_table.keys()},
                )
            blocks.append(block)

        return blocks

    def get_blocks_from_files(self, block_template, blocks0=False):

        axlab = [al for al in self.fullsize.keys()]
        elsize = [self.elsize[al] for al in axlab] if self.elsize else [1] * len(axlab)

        from stapl3d.preprocessing import shading
        stops = []
        for filepath in self.filepaths:
            if '.czi' in filepath:
                from stapl3d.preprocessing import shading
                iminfo = shading.get_image_info(filepath)
                stops.append(iminfo['dims_zyxc'])
            elif '.csv' in filepath:  # test for featur3r: TODO: generalize
                stops.append([0, 0, 0])
            else:
                im = Image(filepath)
                im.load()
                props = im.get_props()
                stops.append(props['shape'])
                im.close()

        # starts = [dict(zip(axlab, [0] * len(axlab))) for imsize in imsizes]
        # stops = [dict(zip(axlab, imsize)) for imsize in imsizes]
        starts = [[0] * len(stop) for stop in stops]

        def get_path(filepaths, block_template, b_idx):

            if '{b' in block_template:
                bfile = block_template.format(b=b_idx)
            elif '{f' in block_template:
                fp = filepaths[b_idx]
                if '.h5' in fp:
                    fstem = os.path.basename(fp.split('.h5')[0])
                else:
                    fstem = os.path.basename(os.path.splitext(fp)[0])
                bfile = block_template.format(f=fstem)
            return bfile + '/{ods}'

        blocks = []
        for b_idx, (start, stop) in enumerate(zip(starts, stops)):
            block = Block(
                id=self._suffix_formats['b'].format(b=b_idx),
                idx=b_idx,
                path=get_path(self.filepaths, block_template, b_idx),
                slices=[slice(sta, sto) for sta, sto in zip(start, stop)],
                axlab=axlab,
                elsize=elsize,
                blocker_info={k:v for k, v in vars().items() if k in self._parameter_table.keys()},
                )
            blocks.append(block)

        return blocks

    def _get_blockbounds(self, ds_start, ds_stop, blocksize, margin,
                         truncate_to_dataset=True, truncate_to_margin=False,
                         shift_final_block_inward=False):
        """Get the block range for a dimension."""

        # blocks
        starts = range(ds_start, ds_stop, blocksize)
        stops = np.array(starts) + blocksize

        # blocks with margin
        starts = np.array(starts) - margin
        stops = np.array(stops) + margin

        # boundary truncations
        if truncate_to_dataset:
            starts = np.clip(starts, ds_start, np.inf).astype(int)

        if truncate_to_dataset:
            stop = ds_stop
        elif truncate_to_margin:
            stop = ds_stop + margin
        else:
            stop = stops[-1]

        stops = np.clip(stops, -np.inf, stop).astype(int)

        if shift_final_block_inward:
            stops[-1] = stop
            starts[-1] = stop - (blocksize + 2 * margin)

        return starts, stops

    def _margins(self, fc, fC, blocksize, margin, fullsize):
        """Return coordinates (fullstack and block) corrected for margin."""

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

        if '.h5' in self.image_in:
            p = self.image_in.split('.h5')
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

    def _get_filepaths(self, blocks=[]):

        if not blocks:
            blocks = list(range(len(self.blocks)))

        filepaths = []
        for block in blocks:
            block = self._blocks[block[0]]
            outputs = self._prep_paths(self.outputs, reps={'b': block.idx})
            filepaths.append(outputs['blockfiles'])

        return filepaths

    def view(self, input=[], images=[], labels=[], settings={}):

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
            'squeeze': '',
            'datatype': '',
            'chunksize': [],
            'pad_mode': 'constant',
            'pad_kwargs': {},
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

        # FIXME: moduledir (=step_id?) can vary
        prev_path = {
            'moduledir': 'biasfield', 'module_id': 'biasfield',
            'step_id': 'biasfield', 'step': 'postprocess',
            'ioitem': 'outputs', 'output': 'aggregate',
            }
        datapath = self._get_inpath(prev_path)
        if datapath == 'default':
            datapath = self._build_path(
                moduledir=prev_path['moduledir'],
                prefixes=[self.prefix, prev_path['module_id']],
                ext='h5',
                )

        prev_path = {
            'moduledir': 'biasfield', 'module_id': 'biasfield',
            'step_id': 'biasfield', 'step': 'postprocess',
            'ioitem': 'outputs', 'output': 'aggregate_ds',
            }
        biaspath = self._get_inpath(prev_path)
        if biaspath == 'default':  # FIXME: has to default to not correcting?
            biaspath = self._build_path(
                moduledir=prev_path['moduledir'],
                prefixes=[self.prefix, prev_path['module_id']],
                suffixes=['ds'],
                ext='h5',
                )

        vols = list(self.volumes.keys())
        vols += ['data'] if self.output_ND else []

        os.makedirs('blocks', exist_ok=True)
        bpat = self._build_path(moduledir='blocks',
                                prefixes=[self.prefix, 'blocks'],
                                suffixes=[{'b': 'p'}])

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        self._paths.update({
            'split': {
                'inputs': {
                    'data': self.inputpaths['blockinfo']['data'],
                    'bias': '',  #f'{biaspath}/bias',
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
        """Average membrane and nuclear channels and write as blocks."""

        arglist = self._prep_step('split', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._split_with_combinechannels, arglist)

    def _split_with_combinechannels(self, block_idx):
        """Average channels and write as blocks."""

        block = self._blocks[block_idx]

        inputs = self._prep_paths_blockfiles(self.inputs, block, key='data')
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        # INPUT
        infile = inputs['data']
        """
        if self.filepaths:
            infile = self.filepaths[block.idx]
        else:
            infile = inputs['data'].format(b=block.idx)
        """

        print('Writing block with id {} to {}'.format(block.id, block.path))

        im = Image(infile, permission='r')
        im.load()

        pad_width, slices = get_padding(block, im.dims)
        im.slices = slices

        bf, bias_dsfacs = None, []
        if inputs['bias']:
            bf = Image(inputs['bias'], permission='r')
            bf.load()
            bias_dsfacs = [round(bf.elsize[bf.axlab.index(dim)] /
                                 im.elsize[im.axlab.index(dim)])
                           for dim in bf.axlab]

        # OUTPUTS Image, 3D and ND
        props = im.get_props2()
        props['path'] = block.path.format(ods='data')  # outputs['data']
        props['permission'] = 'r+'
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = self.chunksize or [max(64, self.blockmargin[al])
                                             if al in 'xy' else s
                                             for al, s in zip(im.axlab, im.dims)]
        props['shape'] = list(im.slices2shape(list(block.slices)))
        props['dims'] = list(im.slices2shape(list(block.slices)))
        props['slices'] = None

        squeezed, squeezed_idxs = self._find_squeezed_dims(im)

        if not self.volumes:  # no need for channel averaging

            mo_ND = Image(**props)
            mo_ND.squeeze(squeezed)
            mo_ND.create()

            data = im.slice_dataset(squeeze=False).astype('float')
            data = biasfield_correction(data, im.slices, bf, bias_dsfacs)
            data = np.pad(data, pad_width, mode=self.pad_mode, **self.pad_kwargs)
            data = np.squeeze(data, axis=squeezed_idxs)

            mo_ND.write(data.astype(mo_ND.dtype))
            mo_ND.close()

            return


        if 'c' not in im.axlab:
            print('No channel axis label; cannot perform channel averaging')
            return





        if self.output_ND:
            mo_ND = Image(**props)
            mo_ND.squeeze(squeezed)
            mo_ND.create()

        mo_3D = Image(**props)
        mo_3D.squeeze('c')

        idxs = [i for i in range(im.dims[im.axlab.index('c')])]
        vols = {k: v for k, v in self.volumes.items()}
        for k, ov in vols.items():
            default = {
                'ods': k,
                'idxs': idxs,
                'weights': [1] * len(idxs),
                'dtype': mo_3D.dtype,
                'data': np.zeros(mo_3D.shape, dtype='float'),
                }
            vols[k] = {**default, **ov}

        # Collect all channel indices to process.
        ch_idxs = set([])
        if self.output_ND:
            ch_idxs |= set(list(range(im.slices[im.axlab.index('c')].start, im.slices[im.axlab.index('c')].stop)))
        ch_idxs |= set([l for k, v in vols.items() for l in v['idxs']])

        for ch_idx in ch_idxs:

            im.slices[im.axlab.index('c')] = slice(ch_idx, ch_idx + 1, None)

            data = im.slice_dataset(squeeze=False).astype('float')
            data = biasfield_correction(data, im.slices, bf, bias_dsfacs)
            data = np.pad(data, pad_width, mode=self.pad_mode, **self.pad_kwargs)
            data = np.squeeze(data, axis=squeezed_idxs)

            if self.output_ND:
                if 'c' not in squeezed:
                    mo_ND.slices[mo_ND.axlab.index('c')] = slice(ch_idx, ch_idx + 1, None)
                mo_ND.write(data.astype(mo_ND.dtype))

            for name, output in vols.items():
                if ch_idx in output['idxs']:
                    idx = output['idxs'].index(ch_idx)
                    data *= output['weights'][idx]
                    output['data'] += np.squeeze(data)

        mo_ND.close()

        for name, output in vols.items():
            output['data'] /= len(output['idxs'])
            write_image(mo_3D.get_props2(),
                        block.path.format(ods=output['ods']),
                        output['data'].astype(mo_3D.dtype))

    def _find_squeezed_dims(self, im=None, squeezed=''):

        if not self.squeeze:
            pass

        elif isinstance(self.squeeze, str):
            squeezed = self.squeeze

        elif isinstance(self.squeeze, bool):
            for al in im.axlab:
                if self.blocksize[al] == 1 and self.blockmargin[al] == 0:
                    squeezed += al

        return squeezed, tuple(im.axlab.index(al) for al in squeezed)

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

        # inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        volume = self._volumes[volume_idx]
        volname = list(volume.keys())[0]

        ukey = f'{volname}_ulabels'
        ulabelpath = outputs[ukey] if ukey in outputs.keys() else ''

        # Properties of the input blocks
        im = Image(self._blocks[self.blocks[0]].path.format(ods=volname), permission='r')
        im.load()
        props = im.get_props2()
        im.close()

        props['path'] = outputs[volname]  #f'{outstem}.h5/{volname}'
        props['permission'] = 'r+'
        props['axlab'] = self.inlayout or props['axlab']
        props['elsize'] = self.elsize or props['elsize']
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = props['chunks'] or None

        merge_axes = ''
        for al in im.axlab:
            # 1. check if the volume was split over dimension K
            block0 = self._blocks[0]
            ax_idx = block0.axlab.index(al)
            axis_blocked = block0.slices[ax_idx].stop != self.fullsize[al]
            # 2. check if merging is desired (default: True if axis_blocked)
            if axis_blocked:
                merge_axes += al

        dims = [self.fullsize[al] if al in merge_axes else props['dims'][im.axlab.index(al)]
                for al in props['axlab']]

        props['dims'] = dims
        props['shape'] = dims
        props['slices'] = None

        for ax in self.squeeze:
            props = im.squeeze_props(props, dim=props['axlab'].index(ax))

        mo = LabelImage(**props)
        mo.create()

        # Select a subset blocks
        self.blocks = self.blocks or list(range(len(self._blocks)))

        # Merge the blocks sequentially (TODO: reintroduce MPI with h5-para).
        ulabels = set([])
        for block_idx in self.blocks:
            block = self._blocks[block_idx]
            blockpath = block.path.format(ods=volname)
            print('processing volume {} block {} from {}'.format(volname, block.id, blockpath))
            im = Image(blockpath, permission='r')
            im.load(load_data=False)
            self.set_slices(im, mo, block, merge_axes)
            data = im.slice_dataset()
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
            input = input or self._blocks[0].path.replace('/{ods}', '')
        elif isinstance(input, (int, float)):
            input = self._blocks[input].path.replace('/{ods}', '')
        elif isinstance(input, list):
            input = input or list(range(len(self._blocks)))

        super().view(input, images, labels, settings)


def blockdict(block_template, suffix_format, start, stop, b_idx, axlab, elsize, blocker_info):

    bdict = {
        'id': suffix_format.format(b=b_idx),
        'idx': b_idx,
        'path': block_template.format(b=b_idx) + '/{ods}',
        'slices': [slice(sta, sto) for sta, sto in zip(start, stop)],
        'axlab': axlab,
        'elsize': elsize,
        'blocker_info': blocker_info,
        }

    return bdict


def get_padding(block, dims):
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


def biasfield_correction(data, slices, bf, bias_dsfacs):

    if bf is None:
        return data

    print('Performing bias field correction.')
    bias = get_bias_field_block(bf, slices, data.shape, bias_dsfacs)
    bias = np.reshape(bias, data.shape)
    data /= bias
    data = np.nan_to_num(data, copy=False)

    return data


def get_bias_field_block(bf, slices, outdims, dsfacs):
    """Retrieve and upsample the biasfield for a datablock."""

    bf.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                 for slc, ds in zip(slices, dsfacs)]
    bf_block = bf.slice_dataset().astype('float32')
    bias = resize(bf_block, outdims, preserve_range=True)

    return bias


def write_image(props, outputpath, data):
    #props = im.get_props2()
    props['path'] = outputpath
    props['permission'] = 'r+'
    props['dtype'] = data.dtype
    mo = Image(**props)
    mo.create()
    mo.write(data)
    return mo


def link_blocks(filepath_in, filepath_out, dset_in, dset_out,
                delete=True, links=True, is_unet=False):

    def delete_dataset(filepath, dset):
        try:
            im = Image(f'{filepath}/{dset}', permission='r+')
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

        im = Image(f'{filepath_in}/{dset_in}', permission='r')
        im.load(load_data=False)

        props = im.get_props()
        if is_unet:
            props['axlab'] = 'zyx'
            props['shape'] = props['shape'][1:]
            props['slices'] = props['slices'][1:]
            props['chunks'] = props['chunks'][1:]

        data = im.slice_dataset(squeeze=True)

        im.close()

        mo = Image(f'{filepath_out}/{dset_out}', permission=mode, **props)
        mo.create()
        mo.write(data)
        mo.close()


if __name__ == "__main__":
    main(sys.argv[1:])
