#!/usr/bin/env python

"""Resegment the dataset block boundaries.

"""

import os
import sys
import time
import logging
import pickle
import shutil
import itertools
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from skimage.segmentation import relabel_sequential
from skimage.color import label2rgb

from stapl3d import parse_args, Stapl3r, Image, MaskImage, LabelImage
from stapl3d.blocks import Block, Block3r
from stapl3d.segmentation import segment
from stapl3d.reporting import gen_orthoplot

logger = logging.getLogger(__name__)


def main(argv):
    """Resegment the dataset block boundaries."""

    steps = ['relabel', 'estimate']
    args = parse_args('zipping', steps, *argv)

    zipp3r = Zipp3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        zipp3r._fun_selector[step]()


class Zipp3r(Block3r):
    _doc_main = """Resegment the dataset block boundaries."""
    _doc_attr = """

    Zipp3r Attributes
    ----------
    ids_labels: str, default 'segm/labels'
        Input labelvolume (seamed).
    ods_labels: str, default 'segm/labels_zip'
        Output labelvolume (zipped).
    ods_zipmask: str, default 'segm/labels_zipmask'
        Output mask of resegmented area.
    ods_blocks: str, default ''
        Output labelvolume with block indices (optional).
    force_relabel_sequential: bool, default False
        Force sequential labels in the relabel step (slow).
    segmentation_id: str, default 'segmentation'
        Parameter file entry to retrieve segmentation parameters from.
    zipsteps: str, default 'zyxq'
        Zipsteps to perform, typically '<axislabels> + q' ('q' is for zipquads).
    seams: list, default [0, 1, ..., N]
        Linear indices to the seams to resegment.
    """
    _doc_meth = """

    Zipp3r Methods
    --------
    run
        Run all steps in the Zipp3r module.
    relabel
        Relabel with unique labels over all blocks.
    estimate
        Resegment the dataset block boundaries.
    view
        View volumes with napari.
    """
    _doc_exam = """

    Zipp3r Examples
    --------
    # TODO
    """
    __doc__ = f"{_doc_main}{Stapl3r.__doc__}{_doc_meth}{_doc_attr}{_doc_exam}"

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'zipping'

        super(Zipp3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'relabel': self.relabel,
            'estimate': self.estimate,
            })

        self._parallelization.update({
            'relabel': ['blocks'],
            'estimate': ['seams'],
            })

        self._parameter_sets.update({
            'relabel': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'blocks'),
                },
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'seams'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'ids_labels': 'segm/labels',
            'ods_labels': 'segm/labels_zip',
            'ods_zipmask': 'segm/labels_zipmask',
            'ods_blocks': '',
            '_bg_label': 0,
            'force_relabel_sequential': False,
            'segmentation_id': 'segmentation',
            'zipsteps': 'zyxq',
            'seams': [],
            '_seams': [],
        }  # TODO: to saved paths
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_zipper()

        self._init_log()

        self._prep_blocks()

        axlab = self._blocks[0].axlab
        self._axlab = ''.join([al for al in axlab if al in 'zyx'])

        self._images = []
        self._labels = []

    def _init_paths_zipper(self):

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        stem = self._build_path(
            moduledir='blocks',
            prefixes=[self.prefix, 'blocks'],
            )

        self._paths.update({
            'relabel': {
                'inputs': {
                    'blockfiles': blockfiles,
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    'maxlabelfile': f'{stem}_maxlabels_relabel.txt',
                    }
                },
            'estimate': {
                'inputs': {
                    'blockfiles': blockfiles,
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    'maxlabelfile': f'{stem}_maxlabels_estimate.txt',
                    'stem': stem,
                    'report': blockfiles.replace('.h5', '_zipper.pdf'),  # FIXME
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def _gather_maxlabels(self, ids):
        """Write the maximum of each labelvolume to file."""

        maxlabelfile = self._prep_paths(self.outputs)['maxlabelfile']

        maxlabels = []
        for block in self._blocks:
            image_in = f'{block.path}/{ids}'
            im = Image(image_in, permission='r')
            im.load(load_data=False)
            maxlabels.append(im.ds.attrs['maxlabel'])
            im.close()
            # block.create_dataset(ids, imtype='Label')
            # block_ds_in = block.datasets[ids]
            # block_ds_in.load(load_data=False)
            # maxlabels.append(block_ds_in.image.maxlabel)
            # FIXME: 'h5py object cannot be pickled' when using multiprocessing
            # TODO: remove all references

        np.savetxt(maxlabelfile, maxlabels, fmt='%d')

        return maxlabels

    def relabel(self, **kwargs):
        """Relabel with unique labels over all blocks."""

        t = time.time()

        step = 'relabel'
        arglist = self._prep_step(step, kwargs)
        self._gather_maxlabels(self.ids_labels)

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._relabel_blocks, arglist)

        elapsed = time.time() - t
        print(f'Finished {self._module_id}:{step} in {elapsed:1f} s')
        print(f'---')

    def _relabel_blocks(self, block):
        """Relabel dataset."""

        def write_output(block, ods, props={}, data=None, maxlabel=None):
            block.create_dataset(ods, **props)
            block_ds = block.datasets[ods]
            block_ds.create_image(f'{block.path}/{ods}')
            if data is not None:
                block_ds.write(data, maxlabel=maxlabel)

        block = self._blocks[block]
        outputs = self._prep_paths(self.outputs, reps={'b': block.idx})

        # Retrieve max of all labels up to this block.
        maxlabels = np.loadtxt(outputs['maxlabelfile'], dtype=np.uint32)
        maxlabel = np.sum(maxlabels[:block.idx])

        # Read the input dataset.
        props = dict(imtype='Label')
        block.create_dataset(self.ids_labels, **props)
        block_ds_in = block.datasets[self.ids_labels]
        block_ds_in.read(from_block=True)
        data = block_ds_in.image.ds[:]

        # Relabel the volume.
        if self.force_relabel_sequential:
            # create sequential labels from maxlabel
            data, _, inv = relabel_sequential(data, offset=maxlabel)
            new_maxlabel = len(inv) - 1
        else:
            # increment all foreground labels with maxlabel
            mask = data == self._bg_label
            data[~mask] += maxlabel
            new_maxlabel = maxlabel + block_ds_in.image.maxlabel

        # Write relabeled block, zipmask volume and blockindex volume.
        ds_props = dict(axlab=self._axlab, imtype='Label', dtype='uint32')
        write_output(block, self.ods_labels, ds_props, data, new_maxlabel)

        if self.ods_zipmask:
            ds_props = dict(axlab=self._axlab, imtype='Mask', dtype='bool')
            write_output(block, self.ods_zipmask, ds_props)

        if self.ods_blocks:
            labelvalue = block.idx + 1
            data = np.ones_like(data, dtype='uint32') * labelvalue
            ds_props = dict(axlab=self._axlab, imtype='Label', dtype='uint32')
            write_output(block, self.ods_blocks, ds_props, data, labelvalue)

    def _set_blockmap(self, axislabels='zyx'):
        """Set a map with block indices."""
        # TODO: move to blocks.py and intergrate with seamgrid

        ds = [int( np.ceil( self.fullsize[al] / self.blocksize[al] ) )
              for al in axislabels]
        self.blockmap = np.zeros(ds, dtype='uint16')

        for block in self._blocks:
            bc = [int(block.slices_region[al].start / self.blocksize[al])
                  for al in axislabels]
            self.blockmap[bc[0], bc[1], bc[2]] = block.idx

    def estimate(self, **kwargs):
        """Resegment the dataset block boundaries."""

        self._set_blockmap(axislabels=self._axlab)
        seamgrid_shape = np.array(self.blockmap.shape) - 1
        self._prep_step('estimate')

        for zipstep in self.zipsteps:

            self._gather_maxlabels(self.ods_labels)

            self.zipstep = zipstep

            if zipstep != 'q':
                self._axis = self._axlab.index(self.zipstep)
                self._estimate_ziplines(seamgrid_shape)
            else:
                self._axis = 0
                self._estimate_zipquads(seamgrid_shape)

    def _estimate_ziplines(self, seamgrid_shape, offsets=[0, 1]):
        """Resegment labels on ziplines."""

        for offset in offsets:

            self._seams = []
            for seam_idx in range(offset, seamgrid_shape[self._axis], 2):
                self._seams.append(seam_idx)

            if len(self._seams):
                step = f'{self.zipstep}:{offset}'
                print(f'Running  zipstep:offset "{step}" with seams {self._seams}')
                self.compute_zip_step()

    def _estimate_zipquads(self, seamgrid_shape, offsets=[0, 1]):
        """Resegment labels on zipquads."""

        sg_shape = np.maximum(seamgrid_shape, np.array([1, 1, 1]))

        for (axis_0, axis_1, axis_2) in itertools.product(offsets, repeat=3):
            starts = [axis_0, axis_1, axis_2]
            stops = sg_shape
            steps = [2, 2, 2]

            self._seams = []
            for seam_0 in range(starts[0], stops[0], steps[0]):
                for seam_1 in range(starts[1], stops[1], steps[1]):
                    for seam_2 in range(starts[2], stops[2], steps[2]):
                        seamnumbers = [seam_0, seam_1, seam_2]
                        seam_idx = np.ravel_multi_index(seamnumbers, sg_shape)
                        self._seams.append(seam_idx)

            if len(self._seams):
                step = f'{self.zipstep}:{axis_0},{axis_1},{axis_2}'
                print(f'Running  zipstep:offset "{step}" with seams {self._seams}')
                self.compute_zip_step()

    def compute_zip_step(self, **kwargs):
        """Compute the zip-step."""

        t = time.time()

        step = 'estimate'
        arglist = self._prep_step(step, kwargs)

        n_proc = min(self._n_workers, len(arglist))
        print(f'Submitting {len(arglist):3d} jobs over {n_proc:3d} processes')

        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(self._resegment_seam, arglist)

        elapsed = time.time() - t
        print(f'Finished {self._module_id}:{step} in {elapsed:1f} s')
        print(f'---')

    def _resegment_seam(self, seam_idx):
        """Resegment all pairs of a seam."""

        # Load the current maxlabel and increment with margin for this seam.
        maxlabel_margin = 100000  # TODO: flexible maxlabel_margin
        outputs = self._prep_paths(self.outputs)
        maxlabels = np.loadtxt(outputs['maxlabelfile'], dtype=np.uint32)
        self.maxlabel = max(maxlabels)
        self.maxlabel += seam_idx * maxlabel_margin

        # Run all pairs/quads in the seam.
        seam_pairs = self.get_seam_pairs(seam_idx)
        for pair_idx, pair in enumerate(seam_pairs):
            blocks = [self._blocks[idx] for idx in pair]
            self._resegment_pair(blocks, seam_idx, pair_idx)

    def get_seam_pairs(self, seam_idx):
        """Find all pairs of a seam from the blockmap."""

        seamgrid_shape =  np.array(self.blockmap.shape) - 1
        sg_shape = np.maximum(seamgrid_shape, np.array([1, 1, 1]))
        seamnumbers = np.unravel_index(seam_idx, sg_shape)

        # Slice the blockmap to get the block indices of the blocks around the seam.
        if self.zipstep == 'q':
            slcs = [slice(seamnumbers[d], seamnumbers[d] + 2) for d in range(3)]
            sh = (1, 4) if any(seamgrid_shape == 0) else (1, 8)
        else:
            slcs = [slice(None) for d in range(3)]
            slcs[self._axis] = slice(seam_idx, seam_idx + 2)
            sh = (-1, 2)

        # FIXME: assumes zyx order
        pairs = self.blockmap[tuple(slcs)]
        newaxes = {'x': [0, 1, 2], 'y': [0, 2, 1], 'z': [2, 1, 0], 'q': [0, 1, 2]}
        pairs = np.moveaxis(pairs, [0, 1, 2], newaxes[self.zipstep])
        pairs = np.reshape(pairs, sh)

        return pairs

    def _resegment_pair(self, blocks, seam_idx, pair_idx):
        """Resegment the boundaries between pairs/quads of blocks."""

        outputs = self._prep_paths(self.outputs)

        self.n = dict(zip(self._axlab, [1, 1, 1]))

        blockset = self._reseg_mask(blocks)

        blockset.assemble_dataset(self.ods_labels, 'Label')

        cslcs = {}
        seg = blockset.assembled.datasets[self.ods_labels].image.ds
        cslcs[self.ids_labels] = {dim: get_cslc(seg, ax) for ax, dim in enumerate('zyx')}  # FIXME: inplace?

        ids_data = self.resegment(blockset)

        data_ds = blockset.assembled.datasets[ids_data].image.ds
        cslcs[ids_data] = {dim: get_cslc(data_ds, ax) for ax, dim in enumerate('zyx')}

        mask = blockset.assembled.datasets[self.ods_zipmask].image.ds
        seg[mask] = blockset.assembled.datasets[self.ods_labels].image.ds[mask]  # updated segmentation
        cslcs[self.ods_labels] = {dim: get_cslc(seg, ax) for ax, dim in enumerate('zyx')}

        cslcs[self.ods_zipmask] = {dim: get_cslc(mask, ax) for ax, dim in enumerate('zyx')}

        # blockset.assembled.datasets[self.ods_zipmask].image.ds
        # self.read_margins(blocks, self.ods_zipmask, 'Mask')
        # zipmasks = tuple(block.datasets[self.ods_zipmask].image.ds for block in blocks)
        # zipmask = self.concat_margins(zipmasks, self._axis)
        # zipmask |= mask
        # # ods_zipmask = [block.datasets[self.ods_zipmask] for block in blocks]
        # self.write_margins(zipmask, blocks, self.ods_zipmask, imtype='Mask')

        blockset.write_margins(self.ods_labels)
        blockset.write_margins(self.ods_zipmask)

        stem = outputs['stem']
        filepath = f'{stem}_axis{self._axis:01d}-seam{seam_idx:03d}-j{pair_idx:03d}.pdf'
        margin = self.blockmargin['x']  # FIXME
        self.report(filepath, centreslices=cslcs, axis=self._axis, margin=margin)

    def _reseg_mask(self, blocks):

        capped = {al: False for al in self._axlab}
        labelsets = (set([]),)
        while True:

            # Read the labels into the block objects.
            blockset = Zipset(blocks, self.zipstep, self.n)
            blockset.assemble()
            blockset.read_margins(self.ods_labels, imtype='Label')

            # Get the labelsets on the first iteration.
            if not set().union(*labelsets):
                labelsets = self.get_boundary_labels(blockset)

            # Return if there are no labels on the seam.
            if not set().union(*labelsets):
                return None

            self.create_resegmentation_mask(blockset, labelsets)

            if not self.check_margin(blockset):
                break
            if all(capped.values()):
                break

            n_max = self._get_nmax(blockset)
            for al in self._axlab:
                if self.n[al] < n_max[al]:
                    self.n[al] += 1
                else:
                    capped[al] = True

        return blockset

    def _get_nmax(self, blockset):
        """Calculate how many margin-blocks fit into the dataset."""

        n_max = {}
        for al in self._axlab:
            sizes = []
            for block in blockset.blocks:
                size = block.slices_region[al].stop - block.slices_region[al].start
                sizes.append(size)
            if self.blockmargin[al]:
                n_max[al] = int(block.shape[al] / self.blockmargin[al]) - 1  # FIXME?
            else:
                n_max[al] = 1

        return n_max

    def get_boundary_labels(self, blockset, bg=set([0])):
        """Find the labels on the boundary of blocks."""

        segs_labels = []
        for block, side_selector in zip(blockset.blocks, blockset.side_selectors):
            slices = blockset._get_slices_boundaries(side_selector)
            seg_marg = block.datasets[self.ods_labels].image.ds
            data = seg_marg[tuple([slices[al] for al in blockset.axlab])]
            segs_labels.append(set(np.unique(data)) - bg)

        return tuple(segs_labels)

    def create_resegmentation_mask(self, blockset, labelsets):
        """Select the segments on the seam."""

        slices = blockset._get_slices(include_margin=False)
        for block, labelset, slcs in zip(blockset.blocks, labelsets, slices):

            seg = block.datasets[self.ods_labels].image
            fw_map = [True if l in labelset else False for l in range(0, seg.maxlabel + 1)]
            mask = np.array(fw_map)[seg.ds]

            block.create_dataset(self.ods_zipmask, axlab=blockset.axlab, slices=slcs, imtype='Mask')
            block_ds_in = block.datasets[self.ods_zipmask]
            block_ds_in.dtype = mask.dtype
            block_ds_in.create_image(data=mask)
            block_ds_in.image.ds[:] = mask

        blockset.assemble_dataset(self.ods_zipmask, 'Mask')

    def resegment(self, blockset):
        """Resegment the concatenated margins."""

        # segment3r = segment.Segment3r(max_workers=1)

        params = self._cfg[self.segmentation_id]['estimate']

        segment.seed_volume(blockset, 'seed', params['seed'])

        segment.segment_volume(blockset, 'segment', params['segment'])

        self.update_labelvalues(blockset, key=params['segment']['ods_labels'])

        return params['segment']['watershed']['ids_image']

    def update_labelvalues(self, blockset, key='reseg'):

        blockset.assembled.datasets[key].image.ds += self.maxlabel
        self.maxlabel += max(np.unique(blockset.assembled.datasets[key].image.ds))

    def check_margin(self, blockset):
        """Check if all voxels marked for resegmentation are within margin."""

        mask = blockset.assembled.datasets[self.ods_zipmask].image.ds

        msum = False

        if self.zipstep == 'q':  # check all axes with multiple blocks
            axes = [d for d in range(mask.ndim) if self.blockmap.shape[d] > 1]
        else:
            axes = [self._axis]

        for axis in axes:
            sh = mask.shape[axis]
            for s in [slice(0, 1), slice(sh - 1, sh)]:
                slc = [slice(None)] * mask.ndim
                slc[axis] = s
                msum = msum | bool(np.sum(mask[tuple(slc)]))

        return msum

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        # idss = ['memb/mean', 'segm/labels_raw', 'segm/labels_zipmask', 'segm/labels_zip']

        # filepath = kwargs['outputs']['blockfiles']
        # kwargs['paths'] = get_paths(f'{filepath}/mean')  # FIXME: assumes dataset is generated under that name

        # kwargs['centreslices'] = get_centreslices(kwargs, idss=idss)

        return kwargs

    def _gen_subgrid(self, f, gs, channel=None):
        """Generate the axes for printing the report."""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(4, 1)

        # t = ['data', 'orig', 'reseg', 'final']
        t = ['memb/prep', 'segm/labels_raw', 'segm/labels_zipmask', 'segm/labels_zip']
        voldict = {name: {'idx': i, 'title': name} for i, name in enumerate(t)}

        for k, vd in voldict.items():
            titles[k] = (vd['title'], 'lcm', 0)
            axdict[k] = gen_orthoplot(f, gs01[vd['idx']], 5, 1)

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict={}):
        """Plot graph with shading image."""

        idss = ['memb/prep', 'segm/labels_raw', 'segm/labels_zipmask', 'segm/labels_zip']

        cslcs = info_dict['centreslices']
        for k, v in cslcs.items():
            cslcs[k]['x'] = cslcs[k]['x'].transpose()

        aspects = ['equal', 'auto', 'auto']
        for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

            data = cslcs[idss[0]][dim]

            lines = (info_dict['axis'] == 0) | (info_dict['axis'] != i)

            if (dim == 'z') & (info_dict['axis'] == 1):
                fs = [0, data.shape[1]]
                ds_b = data.shape[0]
            elif (dim == 'z') & (info_dict['axis'] == 2):
                fs = [0, data.shape[0]]
                ds_b = data.shape[1]
            else:
                fs = [0, data.shape[0]]
                ds_b = data.shape[1]

            margin = info_dict['margin']
            pp = [margin * i for i in range(1, int(ds_b/margin))]
            if pp:
                del pp[int(len(pp)/2)]  # deletes the line on the seam

            ax = axdict[idss[0]][i]
            ax.imshow(data, aspect=aspect, cmap='gray')
            ax.axis('off')
            for v in idss[1:]:
                ax = axdict[v][i]
                labels = cslcs[v][dim]
                clabels = label2rgb(labels, image=None, bg_label=0)
                ax.imshow(clabels, aspect=aspect)
                ax.axis('off')

            if lines:
                for m in pp:
                    if (dim == 'z') & (info_dict['axis'] == 0):
                        for v in idss:
                            axdict[v][i].plot(fs, [m, m], '--', linewidth=1, color='w')
                            axdict[v][i].plot([m, m], fs, '--', linewidth=1, color='w')
                    elif (dim == 'z') & (info_dict['axis'] == 1):
                        for v in idss:
                            axdict[v][i].plot(fs, [m, m], '--', linewidth=1, color='w')
                    elif (dim == 'z') & (info_dict['axis'] == 2):
                        for v in idss:
                            axdict[v][i].plot([m, m], fs, '--', linewidth=1, color='w')
                    else:
                        for v in idss:
                            axdict[v][i].plot([m, m], fs, '--', linewidth=1, color='w')

    def view(self, input=[], images=[], labels=[], settings={}):

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

    def view_pairs(self, pairs={}, images=[], labels=[], slices={}):

        view_block_layout = [v for v in self.view_block_layout]

        for i, pair in pairs.items():

            if not i:
                self.view_block_layout = ['margins', 'blocks', 'fullsize']
            else:
                self.view_block_layout = ['margins', 'blocks']

            self.view_blocks(pair, images, labels, slices, zips=True)

            # TODO: only do 'fullsize' on one pair
            # TODO: avoid double plotting?
            # TODO: contrast limits equal over all pairs

        self.view_block_layout = view_block_layout


def get_cslc(data, axis=0):
    """Get centreslices from a numpy array."""

    slcs = [slice(0, s, 1) for s in data.shape]
    cslc = int(data.shape[axis] / 2)
    slcs[axis] = slice(cslc, cslc+1, 1)

    return np.copy(np.squeeze(data[tuple(slcs)]))


class Zipset(object):
    """Blockset for zipping."""

    def __init__(self, blocks=[], zipstep='', n={'z': 1, 'y': 1, 'x': 1}, axlab='zyx'):

        self.blocks = blocks
        self.zipstep = zipstep
        self.axlab = axlab
        self.n = n

        self.side_selectors = self._get_side_selectors()

        self.assembled = self.assemble()

    def _get_side_selectors(self):
        """Return side selectors for the blockset.
        
        Side selectors indicat for each axis
        whether to select the upper or lower section
        of the block in a blockset.
        # NOTE: for the first block side selectors are all upper,
        # for the final block side selectors are all lower.
        #   => i.e. these determine the spatial extent of the blockset
        """

        side_selector_dict = {
            2: [{self.zipstep: 'upper'},
                {self.zipstep: 'lower'}],
            4: [{'x': 'upper',  'y': 'upper'},
                {'x': 'lower', 'y': 'upper'},
                {'x': 'upper',  'y': 'lower'},
                {'x': 'lower', 'y': 'lower'}],
            8: [{'x': 'upper',  'y': 'upper',  'z': 'upper'},
                {'x': 'lower', 'y': 'upper',  'z': 'upper'},
                {'x': 'upper',  'y': 'lower', 'z': 'upper'},
                {'x': 'lower', 'y': 'lower', 'z': 'upper'},
                {'x': 'upper',  'y': 'upper',  'z': 'lower'},
                {'x': 'lower', 'y': 'upper',  'z': 'lower'},
                {'x': 'upper',  'y': 'lower', 'z': 'lower'},
                {'x': 'lower', 'y': 'lower', 'z': 'lower'}],
        }

        return side_selector_dict[len(self.blocks)]

    def _get_slices(self, include_margin=False):
        """Get slices for blocks in the blockset at n * blockmargin."""

        slices = []
        for block, side_selector in zip(self.blocks, self.side_selectors):
            slc = block._get_zip_slices(side_selector, self.n, include_margin)
            slices.append(slc)

        return slices

    def _get_slices_boundaries(self, side_selector):
        """Slice into seam of the blockdata (excluding margin)."""

        extent = 1
        slice_selector = {
            'upper': slice(-extent, None),
            'lower': slice( None, extent),
            }

        slices = {al: slice(None) for al in self.axlab}

        for al, side in side_selector.items():
            slices[al] = slice_selector[side]

        return slices

    def _get_slices_assembled(self, side_selector):
        """"Get slices for assembled blockset."""

        bm = self.blocks[0].blocker_info['blockmargin']  # FIXME: handle more gracefully

        slices = {al: slice(None) for al in self.axlab}

        for al, side in side_selector.items():
            mn = bm[al] * (self.n[al] + 1)  # TODO: figure out the +1 prefered setting
            # mn = bm[al] * self.n[al]
            slice_selector = {
                'upper': slice( None, mn),
                'lower': slice(-mn, None),
                }
            slices[al] = slice_selector[side]

        return slices

    def assemble(self, name='ASSEMBLY'):
        """Create a single block from a blockset."""

        ref_block = self.blocks[0]

        assembled = Block()
        assembled.id = name
        assembled.axlab = ref_block.axlab
        assembled.elsize = ref_block.elsize
        assembled.blocker_info = ref_block.blocker_info

        slcs = self._get_slices(include_margin=False)
        assembled.slices = {
            al: slice(slcs[0][al].start, slcs[-1][al].stop)
            for al in ref_block.axlab
            }
        assembled.shape = {
            al: slc.stop - slc.start
            for al, slc in assembled.slices.items()
            }

        return assembled

    def assemble_dataset(self, ids, imtype=''):
        """Assemble a dataset from the blockset."""

        data = self.concat_margins(ids)
        props = dict(axlab=self.axlab, imtype=imtype, dtype=data.dtype)
        self.assembled.create_dataset(ids, **props)
        self.assembled.datasets[ids].create_image(data=data)

    def read_margins(self, ids='segm/labels', imtype='Label'):
        """Read a set of blocks and slice along the block margins."""

        slices = self._get_slices(include_margin=False)
        for block, slcs in zip(self.blocks, slices):
            # print(block.idx, slices)
            self.read_blockdata(block, ids, slices=slcs, imtype=imtype)

    def read_blockdata(self, block, ids, imtype='', slices=[]):
        """Read a block margin into a block-dataset."""

        props = dict(axlab=self.axlab, imtype=imtype, slices=slices)
        block.create_dataset(ids, **props)
        block_ds = block.datasets[ids]
        block_ds.slices_blockfile = self.full2block(block, slices)
        block_ds.create_image()
        block_ds.read(from_block=True)  # , padded=False
        # NOTE: padded not passed! it is always True 
        # and would switch between slices_region_blockfile and slices_blockfile

    def write_margins(self, ids, imtype=''):
        """Write margin datablocks back to file."""

        slices = self._get_slices(include_margin=True)
        for block, slcs, side_selector in zip(self.blocks, slices, self.side_selectors):
            # Slice the margin from the concatenated block.
            data = self.assembled.datasets[ids].image.ds
            slices_assembled = self._get_slices_assembled(side_selector)
            data_block = data[tuple(slices_assembled.values())]
            # Write the data to file.
            self.write_blockdata(data_block, block, ids, imtype=imtype, slices=slcs)

    def write_blockdata(self, data, block, ids, imtype='', slices=[]):
        """Read a block margin into a block-dataset."""

        # Create block dataset # FIXME: maybe not needed
        props = dict(axlab=self.axlab, imtype=imtype)  #, slices=slices
        block.create_dataset(ids, **props)
        block_ds = block.datasets[ids]
        # block_ds.dtype = 'int'  # FIXME
        block_ds.create_image(block_ds.path)  

        # block_ds.write(data)  # FIXME / TODO # self.full2block(block, slices)

        block_ds.image.load()
        slcs = [self.full2block(block, slices)[al] for al in block_ds.axlab]
        block_ds.image.slices = slcs
        self.update_vol(block_ds.image, data)
        block_ds.image.close()

    def update_vol(self, im, data):

        if isinstance(im, LabelImage):
            im.ds.attrs['maxlabel'] = max(im.ds.attrs['maxlabel'], np.amax(data))
            comps = im.split_path()

        im.write(data)

    def full2block(self, block, slices):
        return {al: self._f2b(al, block, slices) for al in self.axlab}

    def _f2b(self, al, block, slices):
        offset = block.slices_region[al].start - block.slices_region_blockfile[al].start
        return slice(slices[al].start - offset, slices[al].stop - offset)

    def concat_margins(self, ids):
        """Concatenate the margins of neighbouring datablocks."""

        # FIXME: assuming zyx

        ims = tuple(block.datasets[ids].image.ds for block in self.blocks)

        if len(ims) == 2:
            axis = self.axlab.index(self.zipstep)
            return np.concatenate((ims[0], ims[1]), axis=axis)
        elif len(ims) == 4:  # FIXME: assuming yx with z at full range
            return np.concatenate((
                    np.concatenate((ims[0], ims[1]), axis=2),
                    np.concatenate((ims[2], ims[3]), axis=2),
                ), axis=1)
        elif len(ims) == 8:  # TODO: verify
            return np.concatenate((
                    np.concatenate((
                        np.concatenate((ims[0], ims[1]), axis=2),
                        np.concatenate((ims[2], ims[3]), axis=2)),
                        axis=1),
                    np.concatenate((
                        np.concatenate((ims[4], ims[5]), axis=2),
                        np.concatenate((ims[6], ims[7]), axis=2)),
                        axis=1),
                ), axis=0)

if __name__ == "__main__":
    main(sys.argv[1:])
