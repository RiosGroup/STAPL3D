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
from stapl3d.preprocessing.biasfield import write_image, get_bias_field_block

import glob
import h5py
import pathlib

logger = logging.getLogger(__name__)


def main(argv):
    """Block operations."""

    steps = ['split', 'merge']
    args = parse_args('blocks', steps, *argv)

    subclasses = {'split': Splitter, 'merge': Merger}

    for step in args.steps:
        blocker = subclasses[step](
            args.image_in,
            args.parameter_file,
            step_id=args.step_id,
            directory=args.outputdir,
            dataset=args.dataset,
            suffix=args.suffix,
            n_workers=args.n_workers,
        )
        blocker._fun_selector[step]()


class Block(object):
    """Block."""

    def __init__(self, id='', path='', slices={}, **kwargs):

        self.id = id
        self.path = path
        self.slices = slices
        self.shape = {}
        self.margin = {}

    def __str__(self):
        return yaml.dump(vars(self), default_flow_style=False)


class Blocker(Stapl3r):
    """Block operations."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Blocker, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector = {
            'blockinfo': self.print_blockinfo,
            }

        default_attr = {
            'step_id': 'blocks',
            'blocksize_xy': 640,
            'blockmargin_xy': 64,
            'blocksize': {},
            'blockmargin': {},
            'blockfiles': [],
            'fullsize': {},
            'filepat': '',
            '_blocks': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self.set_blocksize()
        self.set_blockmargin()
        self.set_blocks()
        self.set_blockfiles()

        self._parsets = {
            'split': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize_xy', 'blockmargin_xy', 'fullsize'),
                'spar': ('n_workers', 'blocksize', 'blockmargin', '_blocks'),
                },
            }

        # TODO: merge with parsets?
        self._partable = {}

    def print_blockinfo(self):
        # TODO
        # print(self.dump_parameters(write=False))
        # for i, block in enumerate(self._blocks):
        #     print('Block {:05d} with id {} refers to sliced region {}'.format(i, block.id, block.slices))
        pass

    def set_blocksize(self, blocksize={}):

        im = Image(self.image_in, permission='r')
        im.load(load_data=False)
        bs_im = dict(zip(im.axlab, im.dims))
        im.close()

        bs_xy = {d: self.blocksize_xy for d in 'xy' if self.blocksize_xy}

        self.blocksize = {**bs_im, **bs_xy, **self.blocksize, **blocksize}

    def set_blockmargin(self, blockmargin={}):

        bm_im = {d: 0 for d in self.blocksize.keys()}

        bm_xy = {d: self.blockmargin_xy for d in 'xy' if self.blockmargin_xy}

        self.blockmargin = {**bm_im, **bm_xy, **self.blockmargin, **blockmargin}

    def set_blocks(self):
        """

        """

        im = Image(self.image_in, permission='r')
        im.load()
        im.close()

        outputstem = os.path.join(self.directory, self.format_())
        path_tpl = outputstem + '_{bid}.h5/{ods}'

        imslices = dict(zip(im.axlab, im.slices))
        shape = {d: len(range(*slc.indices(slc.stop))) for d, slc in imslices.items()}
        blocksize = self.blocksize or shape
        blocksize = {d: bs if bs else shape[d] for d, bs in blocksize.items()}
        margin = self.blockmargin or dict(zip(im.axlab, [0] * len(im.axlab)))

        starts, stops, = {}, {}
        for d in im.axlab:
            starts[d], stops[d] = self._get_blockbounds(
                imslices[d].start,
                shape[d],
                blocksize[d],
                margin[d],
                )

        ndim = len(im.axlab)
        starts = tuple(starts[dim] for dim in im.axlab)
        stops = tuple(stops[dim] for dim in im.axlab)
        startsgrid = np.array(np.meshgrid(*starts))
        stopsgrid = np.array(np.meshgrid(*stops))
        starts = np.transpose(np.reshape(startsgrid, [ndim, -1]))
        stops = np.transpose(np.reshape(stopsgrid, [ndim, -1]))

        def blockdict(start, stop, axlab, b_idx):
            idstring = self._suffix_formats['b']
            idxs = [axlab.index(d) for d in 'xyzct']
            slcs = [slice(sta, sto) for sta, sto in zip(start, stop)]
            id = idstring.format(b_idx)
            # id = idstring.format(
            #     slcs[idxs[0]].start, slcs[idxs[0]].stop,
            #     slcs[idxs[1]].start, slcs[idxs[1]].stop,
            #     slcs[idxs[2]].start, slcs[idxs[2]].stop,
            #     )
            bpat = path_tpl.format(bid=id, ods='{ods}')
            return {'slices': slcs, 'id': id, 'path': bpat}

        blocks = [Block(**blockdict(start, stop, im.axlab, b_idx))
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

    def set_blockfiles(self):

        self._set_filepat()
        pat = os.path.join(self.directory, self.filepat)
        self.blockfiles = sorted(glob.glob(pat))

    def _set_filepat(self):

        if not self.filepat:
            matcher = self._suffix_formats['b'].format(0).replace('0', '?')
            self.filepat ='{}_{}.h5'.format(self.format_(), matcher)

class Splitter(Blocker):
    """Block splitting."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Splitter, self).__init__(
            image_in, parameter_file,
            module_id='blocks',
            **kwargs,
            )

        self._fun_selector = {
            'blockinfo': self.print_blockinfo,
            'split': self.split,
            }

        default_attr = {
            'blocks': [],
            'memb_idxs': None,
            'memb_weights': [],
            'nucl_idxs': None,
            'nucl_weights': [],
            'mean_idxs': None,
            'mean_weights': [],
            'output_channels': None,
            'output_ND': False,
            'bias_image': '',
            'bias_dsfacs': [],
            'datatype': '',
            'chunksize': [],
            'outputtemplate': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._parsets = {
            'split': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('memb_idxs', 'memb_weights',
                         'nucl_idxs', 'nucl_weights',
                         'mean_idxs', 'mean_weights',
                         'output_channels',
                         'output_ND',
                         'bias_image', 'bias_dsfacs',
                         'datatype', 'chunksize',
                         'outputtemplate',
                         ),
                'spar': ('n_workers', 'blocksize', 'blockmargin', 'blocks'),
                },
            }

        # TODO: merge with parsets?
        self._partable = {
            'memb_idxs': 'Membrane channel indices',
            'memb_weights': 'Membrane channel weights',
            'nucl_idxs': 'Nuclear channel indices',
            'nucl_weights': 'Nuclear channel weights',
            'mean_idxs': 'Channel indices for global mean',
            'mean_weights': 'Channel weights for global mean',
            }

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

        self.set_parameters('split', kwargs)
        arglist = self._get_arglist(['_blocks'])
        self.set_n_workers(len(arglist))
        self.dump_parameters(step=self.step)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._split_with_combinechannels, arglist)

    def _split_with_combinechannels(self, block):
        """Average membrane and nuclear channels and write as blocks."""

        print('Writing block with id {} to {}'.format(block.id, block.path))

        im = Image(self.image_in, permission='r')
        im.load()

        bf = None
        if self.bias_image:
            bf = Image(self.bias_image, permission='r')
            bf.load()

        props = im.get_props2()
        props['path'] = block.path.format(ods='data')
        props['permission'] = 'r+'
        props['dtype'] = self.datatype or props['dtype']
        props['chunks'] = self.chunksize or [self.blockmargin[d] if d in 'xy' else s for d, s in zip(im.axlab, im.dims)]
        props['shape'] = list(im.slices2shape(list(block.slices)))
        props['dims'] = list(im.slices2shape(list(block.slices)))
        props['slices'] = None
        mo_ND = Image(**props)
        if self.output_ND:
            mo_ND.create()
            print(mo_ND.dtype)
        mo_3D = Image(**props)
        mo_3D.squeeze('ct')
        print(mo_3D.dtype)

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
                bias = get_bias_field_block(bf, im.slices, data.shape, self.bias_dsfacs)
                bias = np.reshape(bias, data.shape)
                data /= bias
                data = np.nan_to_num(data, copy=False)

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
            filepath = self.blockfiles[block_idx]
        super().view_with_napari(filepath, idss, ldss=[])


class Merger(Blocker):
    """Block merging."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Merger, self).__init__(
            image_in, parameter_file,
            module_id='blocks',
            **kwargs,
            )

        self._fun_selector = {
            'blockinfo': self.print_blockinfo,
            'merge': self.merge,
            'postprocess': self.postprocess,
            }

        default_attr = {
            'volumes': [],
            'blocks': [],
            'fullsize': [],
            'datatype': '',
            'elsize': [],
            'inlayout': '',
            'squeeze': '',
            'is_labelimage': False,
            'ulabelpath': '',
            '_outputpaths': [],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._parsets = {
            'merge': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),  # TODO
                'spar': ('n_workers', 'blocksize', 'blockmargin', 'volumes', 'blocks'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('n_workers',),
                },
            }

        self._partable = {}

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

        self.set_parameters('merge', kwargs)
        arglist = self._get_arglist(['volumes'])
        self.set_n_workers(len(arglist))
        self.dump_parameters(step=self.step)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._mergeblocks, arglist)

    def _mergeblocks(self, volume):
        """Merge blocks of data into a single hdf5 file."""

        ids = list(volume.keys())[0]
        outstem = self.get_outputpath(volume)

        ulabelpath = ''
        if 'is_labelimage' in volume[ids].keys():
            # or ... TODO: detect ulabel attribute h5?
            if volume[ids]['is_labelimage']:
                ulabelpath = self.ulabelpath or '{stem}_{vol}_ulabels.npy'.format(stem=outstem, vol=ids)

        # PROPS  # TODO: cleanup
        bpath = self._blocks[0].path.format(ods=ids)
        im = Image(bpath, permission='r')
        im.load()
        props = im.get_props2()
        im.close()

        props['path'] = '{stem}.h5/{vol}'.format(stem=outstem, vol=ids)
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

        basename = self.format_([self.dataset, self.suffix])
        filestem = os.path.join(self.directory, basename)
        tgt_file = '{}.h5'.format(filestem)
        tgt_dir  = os.path.dirname(tgt_file)
        f = h5py.File(tgt_file, 'w')
        for volume in self.volumes:
            filestem = self.get_outputpath(volume)
            inputfile = '{}.h5'.format(filestem)
            linked_path = os.path.relpath(inputfile, tgt_dir)
            ext_file = pathlib.Path(linked_path).as_posix()
            ids = list(volume.keys())[0]
            create_ext_link(f, ids, ext_file, ids)

    def get_outputpath(self, volume):

        ids = list(volume.keys())[0]
        try:
            suf = volume[ids]['suffix'].replace('/', '-')
        except:
            suf = ids.replace('/', '-')
        basename = self.format_([self.dataset, self.suffix, suf])
        outstem = os.path.join(self.directory, basename)

        return outstem

    def view_with_napari(self, filepath='', idss=[], ldss=[], vol_idx=0):

        idss = idss or [list(volume.keys())[0] for volume in self.volumes]

        if not filepath:
            basename = self.format_([self.dataset, self.suffix])
            outstem = os.path.join(self.directory, basename)
            filepath = '{}.h5'.format(outstem)

        # TODO: dask array
        super().view_with_napari(filepath, idss, slices={'z': 'ctr'})


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
