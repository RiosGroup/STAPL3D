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
                'ppar': ('blocksize_xy', 'blockmargin_xy', 'fullsize'),
                'spar': ('_n_workers', 'blocksize', 'blockmargin', 'blocks'),
                },
            }

        self._parameter_table = {
            'blocksize': 'Block size',
            'blockmargin': 'Block margin',
        }

        default_attr = {
            'blocksize_xy': 640,
            'blockmargin_xy': 64,
            'blocksize': {},
            'blockmargin': {},
            'fullsize': {},
            'blocks': [],
            '_blocks': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        print('bs', self.blocksize)
        self._init_paths()

        self._init_log()

        self._prep_blocks()

    def _init_paths(self):

        bpat = self._build_path(suffixes=[{'b': 'p'}])

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
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def print_blockinfo(self, **kwargs):

        arglist = self._prep_step('blockinfo', kwargs)
        for i, block in enumerate(self._blocks):
            if i in self.blocks:
                print(f'Block {i:05d} with id {block.id} refers to region {block.slices}')

    def _prep_blocks(self):

        self.set_fullsize(self.inputpaths[self.step]['data'])
        self.set_blocksize()
        self.set_blockmargin()
        self.set_blocks(self.outputpaths[self.step]['blockfiles'])

    def set_fullsize(self, image_in, fullsize={}):

        if image_in:
            im = Image(image_in, permission='r')
            im.load(load_data=False)
            im.close()
            imsize = dict(zip(im.axlab, im.dims))
        else:
            imsize = {}

        self.fullsize = {**imsize, **self.fullsize, **fullsize}

    def set_blocksize(self, blocksize={}):

        bs_xy = {d: self.blocksize_xy for d in 'xy' if self.blocksize_xy}

        self.blocksize = {**self.fullsize, **bs_xy, **self.blocksize, **blocksize}

    def set_blockmargin(self, blockmargin={}):

        bm_im = {d: 0 for d in self.blocksize.keys()}

        bm_xy = {d: self.blockmargin_xy for d in 'xy' if self.blockmargin_xy}

        self.blockmargin = {**bm_im, **bm_xy, **self.blockmargin, **blockmargin}

    def set_blocks(self, block_template, axlab='xyzct'):

        axlab = [al for al in axlab if al in self.fullsize.keys()]
        imslices = {al: slice(0, self.fullsize[al]) for al in axlab}

        shape = {d: len(range(*slc.indices(slc.stop))) for d, slc in imslices.items()}
        blocksize = self.blocksize or shape
        blocksize = {d: bs if bs else shape[d] for d, bs in blocksize.items()}
        margin = self.blockmargin or dict(zip(axlab, [0] * len(axlab)))

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

        self._blocks = blocks

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
                'ppar': ('memb_idxs', 'memb_weights',
                         'nucl_idxs', 'nucl_weights',
                         'mean_idxs', 'mean_weights',
                         'output_channels', 'output_ND',
                         'datatype', 'chunksize',
                         ),
                'spar': ('_n_workers', 'blocksize', 'blockmargin', 'blocks'),
                },
            })

        self._parameter_table.update({
            'memb_idxs': 'Membrane channel indices',
            'memb_weights': 'Membrane channel weights',
            'nucl_idxs': 'Nuclear channel indices',
            'nucl_weights': 'Nuclear channel weights',
            'mean_idxs': 'Channel indices for global mean',
            'mean_weights': 'Channel weights for global mean',
            })

        default_attr = {
            'memb_idxs': None,
            'memb_weights': [],
            'nucl_idxs': None,
            'nucl_weights': [],
            'mean_idxs': None,
            'mean_weights': [],
            'output_channels': None,
            'output_ND': True,
            'datatype': '',
            'chunksize': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths_splitter()

        self._init_log()

        self._prep_blocks()

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

        vols = []
        vols += ['mean'] if self.mean_idxs is not None else []
        vols += ['memb/mean'] if self.memb_idxs is not None else []
        vols += ['nucl/mean'] if self.nucl_idxs is not None else []
        if self.output_channels is not None:
            vols += [f'chan/ch{chan:02d}' for chan in self.output_channels]
        vols += ['data'] if self.output_ND else []

        bpat = self._build_path(suffixes=[{'b': 'p'}])

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

        for step in ['split']:  # self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def split(self, **kwargs):
        """Average membrane and nuclear channels and write as blocks.

        blocksize=[],
        blockmargin=[],
        blocks=[],
        memb_idxs=None,
        memb_weights=[],
        nucl_idxs=None,
        nucl_weights=[],
        mean_idxs=None,
        mean_weights=[],
        output_channels=None,
        output_ND=False,
        bias_image='',
        bias_dsfacs=[1, 64, 64, 1],
        datatype='',
        chunksize=[],
        outputtemplate='',
        """

        arglist = self._prep_step('split', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._split_with_combinechannels, arglist)

    def _split_with_combinechannels(self, block):
        """Average membrane and nuclear channels and write as blocks."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs, reps={'b': block})

        block = self._blocks[block]
        print('Writing block with id {} to {}'.format(block.id, block.path))

        im = Image(inputs['data'], permission='r')
        im.load()

        bf = None
        if inputs['bias']:
            bf = Image(inputs['bias'], permission='r')
            bf.load()
            bias_dsfacs = [round(bf.elsize[bf.axlab.index(dim)] /
                                 im.elsize[im.axlab.index(dim)])
                           for dim in bf.axlab]

        props = im.get_props2()
        props['path'] = block.path.format(ods='data')
        props['permission'] = 'r+'
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = self.chunksize or [self.blockmargin[d] if d in 'xy' else s for d, s in zip(im.axlab, im.dims)]
        props['shape'] = list(im.slices2shape(list(block.slices)))
        props['dims'] = list(im.slices2shape(list(block.slices)))
        props['slices'] = None
        mo_ND = Image(**props)
        try:
            if outputs['data']:
                mo_ND.create()
        except KeyError:
            pass
        mo_3D = Image(**props)
        mo_3D.squeeze('ct')

        c_axis = im.axlab.index('c')
        channel_list = [i for i in range(im.dims[c_axis])]

        ch_idxs = [self.memb_idxs, self.nucl_idxs, self.mean_idxs]
        ch_weights = [self.memb_weights, self.nucl_weights, self.mean_weights]
        ch_out = ['memb/mean', 'nucl/mean', 'mean']
        ch_ids = ['memb', 'nucl', 'mean']

        output_channels = self.output_channels
        if output_channels is not None:

            if output_channels == [-1]:
                output_channels = channel_list

            ids = ["ch{:02d}".format(ch) for ch in output_channels]
            ch_idxs += [[ch] for ch in output_channels]
            ch_weights += [[1.0]] * len(output_channels)
            ch_out += ['chan/{}'.format(ch_id) for ch_id in ids]
            ch_ids += ids

        outputs = {}
        for key, idxs, weights, ods in zip(ch_ids, ch_idxs, ch_weights, ch_out):

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
                'dtype': mo_3D.dtype,
                'data': np.zeros(mo_3D.shape, dtype='float'),
                }

        im.slices = block.slices

        idxs_set = set([l for k, v in outputs.items() for l in v['idxs']])
        for volnr in idxs_set:

            im.slices[c_axis] = slice(volnr, volnr + 1, 1)
            data = im.slice_dataset(squeeze=False).astype('float')
            if bf is not None:
                print('Performing bias field correction.')
                bias = get_bias_field_block(bf, im.slices, data.shape, bias_dsfacs)
                bias = np.reshape(bias, data.shape)
                data /= bias
                data = np.nan_to_num(data, copy=False)

            # FIXME: not written if idxs_set is empty
            if self.output_ND:
                mo_ND.slices[c_axis] = slice(volnr, volnr + 1, 1)
                mo_ND.write(data.astype(mo_ND.dtype))

            for name, output in outputs.items():
                if volnr in output['idxs']:
                    idx = output['idxs'].index(volnr)
                    data *= output['weights'][idx]
                    output['data'] += np.squeeze(data)

        mo_ND.close()

        for name, output in outputs.items():
            output['data'] /= len(output['idxs'])
            write_image(mo_3D, block.path.format(ods=output['ods']), output['data'].astype(mo_3D.dtype))

    def view_with_napari(self, filepath='', idss=['mean', 'memb/mean', 'nucl/mean'], ldss=[], block_idx=0):

        if not filepath:
            filepath = self._abs(self.outputpaths['split']['blockfiles'].format(b=block_idx))

        super().view_with_napari(filepath, idss, ldss=[])



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
            'fullsize': [],
            'datatype': '',
            'elsize': [],
            'inlayout': '',
            'squeeze': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step)

        self._init_paths_merger()

        self._init_log()

        self._prep_blocks()

    def _init_paths_merger(self):

        bpat = self._build_path(suffixes=[{'b': 'p'}])
        bmat = self._build_path(suffixes=[{'b': '?'}])

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
        """

        volumes=[],
        suffix='',
        fullsize=[],
        datatype='',
        elsize=[],
        inlayout='',
        squeeze='',
        is_labelimage=False,
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
            linked_path = os.path.relpath(inputs[ids], tgt_dir)
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

    def view_with_napari(self, filepath='', idss=[], ldss=[], vol_idx=0):

        idss = idss or [list(volume.keys())[0] for volume in self.volumes]

        if not filepath:
            basename = self.format_([self.dataset, self.suffix])
            outstem = os.path.join(self.directory, basename)
            filepath = '{}.h5'.format(outstem)

        # TODO: dask array
        super().view_with_napari(filepath, idss, slices={'z': 'ctr'})


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
