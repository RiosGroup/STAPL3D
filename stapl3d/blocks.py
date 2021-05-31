#!/usr/bin/env python

"""Average membrane and nuclear channels and write as blocks.

    # TODO: reports
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

    def __init__(self, idx=0, id='', path='', slices={}, **kwargs):

        self.idx = idx
        self.id = id
        self.path = path
        self.slices = slices
        self.shape = {}
        self.margin = {}
        self.affine = np.eye(4)
        for i, slc in enumerate(slices[:3]):  # TODO: zyx selection
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
                'ppar': ('fullsize', 'blocksize', 'blockmargin'),  # 'blocksize_xy', 'blockmargin_xy',
                'spar': ('_n_workers', 'blocks'),
                },
            }

        self._parameter_table = {
            'blocksize': 'Block size',
            'blockmargin': 'Block margin',
        }

        default_attr = {
            'blocksize': {},
            'blockmargin': {},
            'fullsize': {},
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
                pass
                # print(f'Block {i:05d} with id {block.id} refers to region {block.slices}')

    def _prep_blocks(self):

        step = 'blockinfo'  #self.step
        self.set_fullsize(self.inputpaths[step]['data'])
        self.set_blocksize()
        self.set_blockmargin()
        if '{b}' in self.inputpaths[step]['data']:
            self._blocks = self.set_blocks_from_files(self.inputpaths[step]['data'], self.outputpaths[step]['blockfiles'])
            self._blocks0 = self.set_blocks_from_files(self.inputpaths[step]['data'], self.outputpaths[step]['blockfiles'], blocks0=True)
        else:
            self._blocks = self.set_blocks(self.outputpaths[step]['blockfiles'])
            self._blocks0 = self.set_blocks(self.outputpaths[step]['blockfiles'], blocks0=True)

    def set_fullsize(self, image_in, fullsize={}):

        if image_in:
            if '{b}' in image_in:
                image_in = image_in.format(b=0)  # FIXME
            im = Image(image_in, permission='r')
            im.load(load_data=False)
            im.close()
            imsize = dict(zip(im.axlab, im.dims))
        else:
            imsize = {}

        self.fullsize = {**imsize, **self.fullsize, **fullsize}

    def set_blocksize(self, blocksize={}):

        # bs_xy = {d: self.blocksize_xy for d in 'xy' if self.blocksize_xy}
        # self.blocksize = {**self.fullsize, **bs_xy, **self.blocksize, **blocksize}
        self.blocksize = {**self.fullsize, **self.blocksize, **blocksize}

    def set_blockmargin(self, blockmargin={}):

        bm_im = {d: 0 for d in self.blocksize.keys()}
        # bm_xy = {d: self.blockmargin_xy for d in 'xy' if self.blockmargin_xy}
        # self.blockmargin = {**bm_im, **bm_xy, **self.blockmargin, **blockmargin}
        self.blockmargin = {**bm_im, **self.blockmargin, **blockmargin}

    def set_blocks(self, block_template, axlab='zyxct', blocks0=False):
        # TODO: flexible axlab order

        axlab = [al for al in axlab if al in self.fullsize.keys()]
        imslices = {al: slice(0, self.fullsize[al]) for al in axlab}

        shape = {d: len(range(*slc.indices(slc.stop))) for d, slc in imslices.items()}
        blocksize = self.blocksize or shape
        blocksize = {d: bs if bs else shape[d] for d, bs in blocksize.items()}
        if blocks0 or not self.blockmargin:
            margin = dict(zip(axlab, [0] * len(axlab)))
        else:
            margin = self.blockmargin

        starts, stops, = {}, {}
        for d in axlab:
            starts[d], stops[d] = self._get_blockbounds(
                imslices[d].start,
                shape[d],
                blocksize[d],
                margin[d],
                )

        ndim = len(axlab)
        starts = tuple(starts[dim] for dim in axlab)
        stops = tuple(stops[dim] for dim in axlab)
        startsgrid = np.array(np.meshgrid(*starts))
        stopsgrid = np.array(np.meshgrid(*stops))
        starts = np.transpose(np.reshape(startsgrid, [ndim, -1]))
        stops = np.transpose(np.reshape(stopsgrid, [ndim, -1]))

        def blockdict(start, stop, axlab, b_idx):
            slcs = [slice(sta, sto) for sta, sto in zip(start, stop)]
            id = self._suffix_formats['b'].format(b=b_idx)
            # idxs = [axlab.index(d) for d in 'xyzct']
            # id = idstring.format(
            #     slcs[idxs[0]].start, slcs[idxs[0]].stop,
            #     slcs[idxs[1]].start, slcs[idxs[1]].stop,
            #     slcs[idxs[2]].start, slcs[idxs[2]].stop,
            #     )
            bpat = block_template.format(b=b_idx) + '/{ods}'
            return {'slices': slcs, 'id': id, 'path': bpat, 'idx': b_idx}

        blocks = [Block(**blockdict(start, stop, axlab, b_idx))
                  for b_idx, (start, stop) in enumerate(zip(starts, stops))]

        return blocks

    def set_blocks_from_files(self, inputpat, block_template, axlab='zyxct', blocks0=False):

        axlab = [al for al in axlab if al in self.fullsize.keys()]

        stops = []
        for idx in [1, 2]:  # FIXME: glob files
            from stapl3d.preprocessing import shading
            iminfo = shading.get_image_info(inputpat.format(b=idx))
            stops.append(iminfo['dims_zyxc'])

        # starts = [dict(zip(axlab, [0] * len(axlab))) for imsize in imsizes]
        # stops = [dict(zip(axlab, imsize)) for imsize in imsizes]
        starts = [[0] * len(stop) for stop in stops]

        def blockdict(start, stop, axlab, b_idx):
            slcs = [slice(sta, sto) for sta, sto in zip(start, stop)]
            id = self._suffix_formats['b'].format(b=b_idx)
            bpat = block_template.format(b=b_idx) + '/{ods}'  # TODO: filename or h5-path pattern?
            return {'slices': slcs, 'id': id, 'path': bpat, 'idx': b_idx}

        blocks = [Block(**blockdict(start, stop, axlab, b_idx))
                  for b_idx, (start, stop) in enumerate(zip(starts, stops))]

        return blocks

    def _get_blockbounds(self, offset, shape, blocksize, margin):
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

    def _margins(self, fc, fC, blocksize, margin, fullsize):
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

        images = images or self._images

        if isinstance(input, str):
            super().view(input, images, labels, settings)
        elif type(input) == int or float:
            filepath = self._abs(self.outputpaths['blockinfo']['blockfiles'].format(b=input))
            super().view(filepath, images, labels, settings)
        else:
            input = input or [0, 1]
            super().view_blocks(input, images, labels, settings)


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
            'datatype': '',
            'chunksize': [],
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
        bpat = self._build_path(moduledir='blocks', prefixes=[self.prefix, 'blocks'], suffixes=[{'b': 'p'}])

        self._paths.update({
            'split': {
                'inputs': {
                    'data': f'{datapath}/data',
                    'bias': f'{biaspath}/bias',
                    },
                'outputs': {
                    **{'blockfiles': f"{bpat}.h5"},
                    **{ods: f"{bpat}.h5/{ods}" for ods in vols},
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

    def _split_with_combinechannels(self, block):
        """Average membrane and nuclear channels and write as blocks."""

        reps = {'b': block} if '{b}' in self.inputs else {}
        inputs = self._prep_paths(self.inputs, reps=reps)
        outputs = self._prep_paths(self.outputs, reps={'b': block})

        # INPUT
        infile = inputs['data'].format(b=block)

        block = self._blocks[block]
        print('Writing block with id {} to {}'.format(block.id, block.path))

        im = Image(infile, permission='r')
        im.load()

        bf = None
        if inputs['bias']:
            bf = Image(inputs['bias'], permission='r')
            bf.load()
            bias_dsfacs = [round(bf.elsize[bf.axlab.index(dim)] /
                                 im.elsize[im.axlab.index(dim)])
                           for dim in bf.axlab]

        # OUTPUTS Image, 3D and ND
        props = im.get_props2()
        props['path'] = block.path.format(ods='data')
        props['permission'] = 'r+'
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = self.chunksize or [max(64, self.blockmargin[d]) if d in 'xy' else s for d, s in zip(im.axlab, im.dims)]  # FIXME: if blockmargin is 0
        props['shape'] = list(im.slices2shape(list(block.slices)))
        props['dims'] = list(im.slices2shape(list(block.slices)))
        props['slices'] = None
        mo_ND = Image(**props)
        try:
            if outputs['data']:
                mo_ND.create()
        except KeyError:
            pass

        props['shape'] = list(im.slices2shape(list(block.slices[:3])))
        props['dims'] = list(im.slices2shape(list(block.slices[:3])))
        mo_3D = Image(**props)
        mo_3D.squeeze('ct')


        ## 3D or 4D or 5D input?...


        if 'c' in im.axlab:
            c_axis = im.axlab.index('c')
            idxs = [i for i in range(im.dims[c_axis])]
        else:
            c_axis = None
            idxs = []

        for k, ov in self.volumes.items():

            default = {
                'ods': k,
                'idxs': idxs,
                'weights': [1] * len(idxs),
                'dtype': mo_3D.dtype,
                'data': np.zeros(mo_3D.shape, dtype='float'),
                }
            self.volumes[k] = {**default, **ov}

        im.slices = block.slices

        idxs_set = set([l for k, v in self.volumes.items() for l in v['idxs']])
        for volnr in idxs_set:
            print('volnr', volnr)

            if c_axis is not None:
                im.slices[c_axis] = slice(volnr, volnr + 1, 1)

            data = im.slice_dataset(squeeze=False).astype('float')
            print(data.shape)

            if bf is not None:
                print('Performing bias field correction.')
                bias = get_bias_field_block(bf, im.slices, data.shape, bias_dsfacs)
                bias = np.reshape(bias, data.shape)
                data /= bias
                data = np.nan_to_num(data, copy=False)

            # FIXME: not written if idxs_set is empty
            if self.output_ND:
                if c_axis is not None:
                    mo_ND.slices[c_axis] = slice(volnr, volnr + 1, 1)
                mo_ND.write(data.astype(mo_ND.dtype))

            for name, output in self.volumes.items():
                if volnr in output['idxs']:
                    idx = output['idxs'].index(volnr)
                    data *= output['weights'][idx]
                    print(data.shape)
                    output['data'] += np.squeeze(data)

        mo_ND.close()

        for name, output in self.volumes.items():
            output['data'] /= len(output['idxs'])
            write_image(mo_3D, block.path.format(ods=output['ods']), output['data'].astype(mo_3D.dtype))

    def view(self, input=[], images=[], labels=[], settings={}):

        images = images or self._images

        if isinstance(input, str):
            super().view(input, images, labels, settings)
        elif type(input) == int or float:
            filepath = self._abs(self.outputpaths['split']['blockfiles'].format(b=input))
            super().view(filepath, images, labels, settings)
        else:
            input = input or [0, 1]
            super().view_blocks(input, images, labels, settings)

    # def view(self, filepath='', images=['mean', 'memb/mean', 'nucl/mean'], labels=[], block_idx=0, settings={}):
    #
    #     if not filepath:
    #         filepath = self._abs(self.outputpaths['split']['blockfiles'].format(b=block_idx))
    #
    #     super().view(filepath, images, labels, settings)
    #
    # def view_blocks(self, block_idxs=[0, 1], images=['mean', 'memb/mean', 'nucl/mean'], labels=[], settings={}):
    #
    #     super().view_blocks(block_idxs, images, labels, settings)


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
                    ods: f"{bmat}.{vol['format'] if 'format' in vol.keys() else 'h5'}/{ods}" for volume in self.volumes for ods, vol in volume.items()
                    },
                'outputs': {
                    **{ods: self.get_outputpath(volume) for volume in self.volumes for ods, vol in volume.items()},
                    **{f'{ods}_ulabels': self.get_outputpath(volume, 'npy') for volume in self.volumes for ods, vol in volume.items() if 'is_labelimage' in vol.keys() and vol['is_labelimage']},
                    },
                },
            'postprocess': {
                'inputs': {
                    ods: self.get_outputpath(volume, fullpath=False) for volume in self.volumes for ods, vol in volume.items()
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

    def _mergeblocks(self, volume):
        """Merge blocks of data into a single hdf5 file."""

        # inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        ids = list(volume.keys())[0]

        ukey = f'{ids}_ulabels'
        ulabelpath = outputs[ukey] if ukey in outputs.keys() else ''

        # PROPS  # TODO: cleanup
        bpath = self._blocks[0].path.format(ods=ids)
        im = Image(bpath, permission='r')
        im.load()
        props = im.get_props2()
        im.close()

        props['path'] = outputs[ids]  #f'{outstem}.h5/{ids}'
        props['permission'] = 'r+'
        props['axlab'] = self.inlayout or props['axlab']
        props['elsize'] = self.elsize or props['elsize']
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = props['chunks'] or None

        if not self.fullsize:
            self.fullsize = {d: self._blocks[-1].slices[im.axlab.index(d)].stop
                             for d in props['axlab']}

        dims = [self.fullsize[d] for d in im.axlab]
        ndim = im.get_ndim()
        if ndim == 4:
            c_idx = props['axlab'].index('c')
            dims.insert(c_idx, im.ds.shape[c_idx])
        props['shape'] = dims

        for ax in self.squeeze:
            props = im.squeeze_props(props, dim=props['axlab'].index(ax))

        mo = LabelImage(**props)
        mo.create()

        # select block subset  # TODO: also for splitter
        if not self.blocks:
            self.blocks = list(range(len(self._blocks)))
        blocks_wm = [block for i, block in enumerate(self._blocks)
                     if i in self.blocks]
        # blocks_nm = [block for i, block in enumerate(self._blocks_nomargin)
        #              if i in self.blocks]  # unused for now

        # Merge the blocks sequentially (TODO: reintroduce MPI with h5-para).
        ulabels = set([])
        # for block, block_nm in zip(blocks_wm, blocks_nm):
        for block in blocks_wm:
            inpath = block.path.format(ods=ids)
            print('processing volume {} block {} from {}'.format(ids, block.id, inpath))
            im = Image(inpath, permission='r')
            im.load(load_data=False)
            self.set_slices(im, mo, block)
            data = im.slice_dataset()
            mo.write(data)
            im.close()
            if ulabelpath:
                ulabels |= set(np.unique(data))

        if ulabelpath:
            np.save(ulabelpath, np.array(ulabels))

        im.close()
        mo.close()

    def set_slices(self, im, mo, block=None):

        if block is None:
            comps = im.split_path()
            dset_info = split_filename(comps['file'])[0]

        for d in im.axlab:

            if block is None:
                l = dset_info[d.lower()]
                h = dset_info[d.upper()]
            else:
                l = block.slices[im.axlab.index(d)].start
                h = block.slices[im.axlab.index(d)].stop

            bs = self.blocksize[d]
            bm = self.blockmargin[d]
            fs = self.fullsize[d]

            (ol, oh), (il, ih) = self._margins(l, h, bs, bm, fs)
            im.slices[im.axlab.index(d)] = slice(il, ih)
            mo.slices[mo.axlab.index(d)] = slice(ol, oh)

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

        images = images or self._images
        labels = labels or self._labels

        if isinstance(input, str):
            super().view(input, images, labels, settings)
        elif type(input) == int or type(input) == float:
            filepath = self._abs(self.outputpaths['merge']['blockfiles'].format(b=input))
            # if not filepath:
            #     basename = self.format_([self.dataset, self.suffix])
            #     outstem = os.path.join(self.directory, basename)
            #     filepath = '{}.h5'.format(outstem)
            super().view(filepath, images, labels, settings)
        else:
            input = input or [0, 1]
            super().view_blocks(input, images, labels, settings)

    # def view(self, filepath='', images=[], labels=[], vol_idx=0, settings={}):

        # idss = idss or [list(volume.keys())[0] for volume in self.volumes]

        # if not filepath:
        #     filestem = os.path.join(self.directory, self.format_())
        #     filepath = f'{filestem}_ds.h5'
        # if not filepath:
        #     filepath = self._abs(self.outputpaths['estimate']['file'])
        # if not filepath:
        #     basename = self.format_([self.dataset, self.suffix])
        #     outstem = os.path.join(self.directory, basename)
        #     filepath = '{}.h5'.format(outstem)

        # TODO: dask array
        # super().view(filepath, images, labels, settings)
        # super().view_with_napari(filepath, idss, slices={'z': 'ctr'})


def get_bias_field_block(bf, slices, outdims, dsfacs):
    """Retrieve and upsample the biasfield for a datablock."""

    bf.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                 for slc, ds in zip(slices, dsfacs)]
    bf_block = bf.slice_dataset().astype('float32')
    bias = resize(bf_block, outdims, preserve_range=True)

    return bias


def write_image(im, outputpath, data):
    props = im.get_props2()
    props['path'] = outputpath
    props['permission'] = 'r+'
    props['dtype'] = data.dtype
    mo = Image(**props)
    mo.create()
    mo.write(data)
    return mo


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
