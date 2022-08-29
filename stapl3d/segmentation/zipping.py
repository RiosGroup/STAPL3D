#!/usr/bin/env python

"""Resegment the dataset block boundaries.

    TODO: consider merging relabel and copyblocks into a preproc step
    TODO: save affine as attribute

"""

import os
import sys
import logging
import pickle
import shutil
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from glob import glob

from scipy import ndimage as ndi

from skimage.segmentation import relabel_sequential, watershed
from skimage.color import label2rgb

from stapl3d import parse_args, Stapl3r, Image, LabelImage, MaskImage, wmeMPI, split_filename
from stapl3d.blocks import Block3r
from stapl3d import get_paths  # TODO: into Image/Stapl3r
from stapl3d.reporting import (
    gen_orthoplot,
    get_cslc,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """Resegment the dataset block boundaries."""

    steps = ['relabel', 'copyblocks', 'zipping']
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
    """Segment cells from membrane and nuclear channels."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'zipping'

        super(Zipp3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'relabel': self.relabel,
            'copyblocks': self.copyblocks,
            'estimate': self.estimate,
            })

        self._parallelization.update({
            'relabel': ['blocks'],
            'copyblocks': ['blocks'],
            'estimate': ['blocks'],
            })

        self._parameter_sets.update({
            'relabel': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'blocks'),
                },
            'copyblocks': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'blocks'),
                },
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'seamgrid': [],
            'ids_labels': 'segm/labels',
            'ods_labels': 'segm/labels_zip',
            'ods_zipmask': 'segm/labels_zipmask',
            'ods_blocks': '',
            'ods_copied': '',
            'segmentation_id': 'segmentation',
        }  # TODO: to saved paths
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_zipper()

        self._init_log()

        self._prep_blocks()

        self._set_zip_layout()

        self._images = ['nucl/prep', 'memb/prep']
        self._labels = ['segm/labels', 'segm/labels_zipmask', 'segm/labels_zip']

    def _init_paths_zipper(self):

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        stem = self._build_path(moduledir='blocks', prefixes=[self.prefix, 'blocks'])

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
            'copyblocks': {
                'inputs': {
                    'blockfiles': blockfiles,
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    'maxlabelfile': f'{stem}_maxlabels_copyblocks.txt',
                    }
                },
            'estimate': {
                'inputs': {
                    'blockfiles': blockfiles,
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    'maxlabelfile': f'{stem}_maxlabels_estimate.txt',
                    'stem': f'{stem}',
                    'report': blockfiles.replace('.h5', '_zipper.pdf'),  # FIXME
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def _set_zip_layout(self):
        """Determine the number of seams in the blocked dataset."""

        # im = Image(self.image_in)
        # im.load(load_data=False)
        # X = im.dims[im.axlab.index('x')]
        # Y = im.dims[im.axlab.index('y')]
        # im.close()
        # print(self.image_in, X, Y, self.blocksize)

        nx = int( np.ceil( self.fullsize['x'] / self.blocksize['x'] ) )
        ny = int( np.ceil( self.fullsize['y'] / self.blocksize['y'] ) )
        n_seams_yx = [ny - 1, nx - 1]
        seams = list(range(np.prod(n_seams_yx)))
        self.seamgrid = np.reshape(seams, n_seams_yx)

    def relabel(self, **kwargs):
        """."""

        step_id = 'relabel'
        outputs = self._prep_paths(self.outputpaths[step_id])
        maxlabelfile = outputs['maxlabelfile']
        arglist = self._prep_step(step_id, kwargs)
        filepaths = self._get_filepaths(arglist)
        get_maxlabels_from_attribute(filepaths, self.ids_labels, maxlabelfile)

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._relabel_blocks, arglist)

    def _relabel_blocks(self, block):
        """Relabel dataset sequentially."""

        block = self._blocks[block]
        inputs = self._prep_paths(self.inputs, reps={'b': block.idx})
        outputs = self._prep_paths(self.outputs, reps={'b': block.idx})

        maxlabels = np.loadtxt(outputs['maxlabelfile'], dtype=np.uint32)
        maxlabel = np.sum(maxlabels[:block.idx])

        filepath = inputs['blockfiles']

        image_in = f'{filepath}/{self.ids_labels}'
        im = LabelImage(image_in)
        im.load()
        data = im.slice_dataset()

        bg_label = 0
        mask = data == bg_label

        force_sequential = False
        if force_sequential:
            data, fw, _ = relabel_sequential(data, offset=maxlabel)
        else:
            data[~mask] += maxlabel

        outpath =f'{filepath}/{self.ods_labels}'
        mo = write_output(outpath, data, props=im.get_props(), imtype='Label')

        try:
            maxlabel_block = im.ds.attrs['maxlabel']
        except KeyError:
            maxlabel_block = None

        if force_sequential or (maxlabel_block is None):
            mo.set_maxlabel()
        else:
            mo.maxlabel = maxlabel + maxlabel_block

        mo.ds.attrs.create('maxlabel', mo.maxlabel, dtype='uint32')
        mo.close()

    def copyblocks(self, **kwargs):
        """."""

        step_id = 'copyblocks'
        outputs = self._prep_paths(self.outputpaths[step_id])
        maxlabelfile = outputs['maxlabelfile']
        arglist = self._prep_step(step_id, kwargs)
        filepaths = self._get_filepaths(arglist)
        get_maxlabels_from_attribute(filepaths, self.ods_labels, maxlabelfile)

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._copy_blocks, arglist)

    def _copy_blocks(self, block_idx):

        def write_maxlabel_attribute(im, mo):
            # TODO make ulabels/maxlabel standard attributes to write into LabelImages.
            try:
                maxlabel = im.ds.attrs['maxlabel']
            except:
                mo.set_maxlabel()
                maxlabel = mo.maxlabel
            mo.ds.attrs.create('maxlabel', maxlabel, dtype='uint32')

        block = self._blocks[block_idx]
        inputs = self._prep_paths(self.inputs, reps={'b': block.idx})

        filepath = inputs['blockfiles']
        image_in = f'{filepath}/{self.ids_labels}'
        im = LabelImage(image_in)
        im.load()

        #  or '_peaks' in pf
        if self.ods_zipmask:
            outpath =f'{filepath}/{self.ods_zipmask}'
            data = np.zeros(im.dims, dtype='bool')
            mo = write_output(outpath, data, props=im.get_props(), imtype='Mask')
            mo.close()

        if self.ods_blocks:
            outpath =f'{filepath}/{self.ods_blocks}'
            data = np.ones(im.dims, dtype='uint16') * (block_idx + 1)
            mo = write_output(outpath, data, props=im.get_props(), imtype='Label')
            write_maxlabel_attribute(im, mo)
            mo.close()

        if self.ods_copied:
            outpath =f'{filepath}/{self.ods_copied}'
            data = im.slice_dataset()
            mo = write_output(outpath, data, props=im.get_props(), imtype='Label')
            write_maxlabel_attribute(im, mo)
            mo.close()

        im.close()

    def estimate(self, **kwargs):
        """Resegment the dataset block boundaries."""

        step_id = 'estimate'
        outputs = self._prep_paths(self.outputpaths[step_id])
        maxlabelfile = outputs['maxlabelfile']
        arglist = self._prep_step(step_id, kwargs)
        filepaths = self._get_filepaths(arglist)
        get_maxlabels_from_attribute(filepaths, self.ids_labels, maxlabelfile)


        # Arguments to `resegment_block_boundaries`
        outputstem = outputs['stem']
        args = [filepaths, 0, [-1, -1, -1], maxlabelfile]
        n_workers = self._n_workers  # max_workers here?!  handled in _prep_step?

        # ziplines
        for axis, n_seams in zip([1, 2], self.seamgrid.shape):
            n_proc = min(n_workers, int(np.ceil(n_seams / 2)))
            for offset in [0, 1]:
                self.compute_zip_step(args, axis=axis, starts=[offset, 0], stops=[n_seams, 1], steps=[2, 2], n_proc=n_proc)
                get_maxlabels_from_attribute(filepaths, self.ods_labels, maxlabelfile)

        # zipquads
        for start_y in [0, 1]:
            for start_x in [0, 1]:
                self.compute_zip_step(args, axis=0, starts=[start_y, start_x], stops=self.seamgrid.shape, steps=[2, 2], n_proc=n_workers)
                get_maxlabels_from_attribute(filepaths, self.ods_labels, maxlabelfile)

    def compute_zip_step(self, args, axis, starts, stops, steps, n_proc):
        """Compute the zip-step."""

        arglist = self._get_args(args, axis, starts, stops, steps)
        print('Submitting {:3d} jobs over {:3d} processes'.format(len(arglist), n_proc))
        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(self._resegment_block_boundaries, arglist)

    def _get_args(self, args, axis, starts, stops, steps):
        """Replace the `axis` and `seamnumbers` arguments
        with values specific for sequential zip-steps.

        axis = 0: zip-quads
        axis = 1: zip-lines over Y
        axis = 2: zip-lines over X
        seamnumbers: start-stop-step triplets (with step=2)
        """

        arglist = []
        for seam_y in range(starts[0], stops[0], steps[0]):
            for seam_x in range(starts[1], stops[1], steps[1]):
                seamnumbers = [-1, seam_y, seam_x]
                args[1] = axis
                if axis == 0:
                    args[2] = [seamnumbers[d] if d != axis else -1 for d in [0, 1, 2]]
                else:
                    args[2] = [seam_y if d == axis else -1 for d in [0, 1, 2]]
                arglist.append(tuple(args))

        return arglist

    def _resegment_block_boundaries(self, filepaths, axis=2, seamnumbers=[-1, -1, -1], maxlabel=''):
        """Resegment the dataset block boundaries."""

        # NB: images_in are sorted in xyz-order while most processing will map to zyx
        # TODO: may want to switch to xyz here; or perhaps sort according to zyx for consistency

        self.axis = axis
        self.seamnumbers = seamnumbers
        self.maxlabel = maxlabel
        blockmargin = [self.blockmargin[d] for d in 'zyx']
        self.margin = blockmargin[2] if self.axis == 0 else blockmargin[self.axis]

        filepaths.sort()
        info = self.get_block_info(filepaths)
        self.set_blockmap(info)
        self.set_seamnumber()
        self.prep_maxlabel(filepaths, self.ods_labels)

        # loop over all pairs in a zipline
        for j, pair in enumerate(self.get_seam_pairs()):

            self.info_ims = tuple(info[idx] for idx in pair)

            self.j = j

            self.set_nmax()
            if self._n_max < 2:
                continue

            self.process_pair()
            # self.process_pair(
            #     ids_nucl,
            #     ids_memb_chan,
            #     find_peaks,
            #     peaks_thr,
            #     peaks_size,
            #     peaks_dil_footprint,
            #     compactness)


    def get_block_info(self, filepaths):
        """Gather image info on each block."""

        info = {}
        for i, filepath in enumerate(filepaths):
            info[i] = {}
            info[i]['filepath'] = filepath
            for j, dim in enumerate('zyx'):
                info[i][dim] = self._blocks0[i].slices[j].start
            for j, dim in enumerate('ZYX'):
                info[i][dim] = self._blocks0[i].slices[j].stop
            info[i]['blockcoords'] = [
                int(info[i]['z'] / self.blocksize['z']),
                int(info[i]['y'] / self.blocksize['y']),
                int(info[i]['x'] / self.blocksize['x']),
                ]

        return info

    def set_blockmap(self, info):
        """Set a map with block indices."""

        ds = np.amax(np.array([v['blockcoords'] for k, v in info.items()]), axis=0) + 1
        self.blockmap = np.zeros(ds, dtype='uint16')
        for k, v in info.items():
            bc = v['blockcoords']
            self.blockmap[bc[0], bc[1], bc[2]] = k

    def get_seam_pairs(self):

        ad = {0: {'ax': [1, 2], 'tp': [0, 1], 'sh': ( 1, 4)},
              1: {'ax': [self.axis], 'tp': [1, 0], 'sh': (-1, 2)},
              2: {'ax': [self.axis], 'tp': [0, 1], 'sh': (-1, 2)}}

        slcs = [slice(self.seamnumbers[d], self.seamnumbers[d] + 2)
                if d in ad[self.axis]['ax'] else slice(None)
                for d in range(3)]
        pairs = np.squeeze(self.blockmap[tuple(slcs)])

        pairs = np.reshape(np.transpose(pairs, ad[self.axis]['tp']), ad[self.axis]['sh'])

        return pairs

    def set_seamnumber(self):

        if self.axis == 0:
            seamgrid_shape = [self.blockmap.shape[1] - 1, self.blockmap.shape[2] - 1]
            self.seamnumber = np.ravel_multi_index(self.seamnumbers[1:], seamgrid_shape)
        else:
            self.seamnumber = self.seamnumbers[self.axis]

    def prep_maxlabel(self, filelist='', ids='', maxlabel_margin=100000):

        if self.maxlabel == 'attrs':
            maxlabels = get_maxlabels_from_attribute(filelist, ids, '')
            self.maxlabel = max(maxlabels)

        try:
            self.maxlabel = int(self.maxlabel)
        except ValueError:
            maxlabels = np.loadtxt(self.maxlabel, dtype=np.uint32)
            self.maxlabel = max(maxlabels)

        self.maxlabel += self.seamnumber * maxlabel_margin

    def set_nmax(self, n_cap=10):
        """Calculate how many margin-blocks fit into the dataset."""

        sizes = []
        if self.axis == 2:
            sizes += [info_im['X'] - info_im['x'] for info_im in self.info_ims]
        elif self.axis == 1:
            sizes += [info_im['Y'] - info_im['y'] for info_im in self.info_ims]
        elif self.axis == 0:
            sizes += [info_im['X'] - info_im['x'] for info_im in self.info_ims]
            sizes += [info_im['Y'] - info_im['y'] for info_im in self.info_ims]

        self._n_max = min(n_cap, int(np.amin(np.array(sizes)) / self.margin))

    # def process_pair(
    #     self,
    #     ids_nucl='',
    #     ids_memb_chan='memb/prep',
    #     find_peaks=False,
    #     peaks_thr=1.16,
    #     peaks_size=[11, 19, 19],
    #     peaks_dil_footprint=[3, 7, 7],
    #     compactness=0.80,
    #     ):
    def process_pair(self):
        """Resegment the boundaries between pairs/quads of blocks."""

        outputs = self._prep_paths(self.outputpaths['estimate'])

        self.n = min(self._n_max, 3)

        while True:

            # get a resegmentation mask
            segs, segs_ds, masks, masks_ds, mask = self.get_resegmentation_mask()

            # return if no labels on block boundaries
            if segs is None:
                return
                # return maxlabel, report

            # increase margin if needed
            if self.check_margin(mask) and self.n < self._n_max:
                self.n += 1
            else:
                break

        # report slices
        cslcs = {}
        cslcs['orig'] = {dim: get_cslc(segs_ds, ax) for ax, dim in enumerate('zyx')}




        from stapl3d.segmentation import segment
        elsize = segs[0].elsize
        params = self._cfg[self.segmentation_id]['estimate']

        pars = params['seed']
        data, data_ds = self.read_images(pars['ids_image'], 'Image', concat=True)
        seed_mask, seed_mask_ds = self.read_images(pars['ids_mask'], 'Mask', concat=True)
        seeds_ds = segment.seed_volume_data(pars, data_ds, seed_mask_ds, elsize)

        # TODO: negative seed for zip
        if 'segment' in params.keys():
            pars = params['segment']
            data, data_ds = self.read_images(pars['ids_image'], 'Image', concat=True)
            # seeds, seeds_ds = self.read_images(pars['ids_labels'], 'Label')
            # if 'ids_mask' in pars.keys():
            #     ws_mask = ... & mask
            # else:
            #     ws_mask = mask
            ws = segment.segment_volume_data(pars, data_ds, seeds_ds, mask, elsize)
        else:
            ws = seeds_ds

        # pars = self._cfg['segmentation']['estimate']['filter']
        # im = segment.filter_segments(filepath, 'filter', params['filter'], save_steps=True)




        ws_ulabels = np.unique(ws)
        ws_max = max(ws_ulabels)
        n_newlabels = len(ws_ulabels) - 1
        ws += self.maxlabel
        self.maxlabel += ws_max
        print('{:10d} labels in final watershed with maxlabel={:10d}'.format(n_newlabels, ws_max))

        segs_ds[mask] = ws[mask]
        self.write_margin(segs, segs_ds)  # resegmentation
        mask_reseg = masks_ds | mask
        self.write_margin(masks, mask_reseg)  # resegmentation mask

        # report slices
        # if ids_memb_chan:
        # else:
        #     cslcs['data'] = {dim: get_cslc(edts_ds, ax) for ax, dim in enumerate('zyx')}
        cslcs['data'] = {dim: get_cslc(data_ds, ax) for ax, dim in enumerate('zyx')}
        cslcs['reseg'] = {dim: get_cslc(ws, ax) for ax, dim in enumerate('zyx')}
        cslcs['final'] = {dim: get_cslc(segs_ds, ax) for ax, dim in enumerate('zyx')}
        filepath = '{}_axis{:01d}-seam{:03d}-j{:03d}.pdf'.format(
            outputs['stem'],
            self.axis,
            self.seamnumber,
            self.j,
            )
        self.report(filepath, centreslices=cslcs, axis=self.axis, margin=self.margin)

    def get_resegmentation_mask(self):
        """Find the mask of segments for resegmentation."""

        # read the margin of the labelimage/zipmask blocks
        segm_imgs, segm_dsets = self.read_images(self.ods_labels, 'Label')
        mask_imgs, mask_dsets = self.read_images(self.ods_zipmask, 'Mask')

        # find the labels that are on the seam
        # TODO: keep seam-labelset for next iteration
        labelsets = self.get_labels(segm_dsets)
        # check if there any labels in the combined set
        if not set().union(*labelsets):
            return None, None, None, None, None

        # create boolean forward maps
        fw_maps = tuple([True if l in labelset else False for l in range(0, seg.maxlabel + 1)] for labelset, seg in zip(labelsets, segm_imgs))
        # create the resegmentation mask
        rseg_dsets = tuple(np.array(fw_map)[segm_dset] for fw_map, segm_dset in zip(fw_maps, segm_dsets))

        # concatenate the margins of the block set
        segm_dset = self.concat_images(segm_dsets)
        mask_dset = self.concat_images(mask_dsets)
        rseg_dset = self.concat_images(rseg_dsets)

        return segm_imgs, segm_dset, mask_imgs, mask_dset, rseg_dset

    def read_images(self, ids='segm/labels', imtype='Label', include_margin=False, concat=False):
        """Read a set of block and slice along the block margins."""

        # get block images
        imgs = tuple(read_image(info_im, ids=ids, imtype=imtype) for info_im in self.info_ims)

        # slice the margins
        self.set_to_margin_slices(imgs, include_margin)
        imgs_marg = tuple(img.slice_dataset() for img in imgs)
        if concat:
            imgs_marg = self.concat_images(imgs_marg)

        return imgs, imgs_marg

    def concat_images(self, ims):
        """Concatenate the margins of neighbouring datablocks."""

        if self.axis == 0:  # NOTE: axis=0 hijacked for quads
            return np.concatenate((np.concatenate((ims[0], ims[1]), axis=2), np.concatenate((ims[2], ims[3]), axis=2)), axis=1)
        else:
            return np.concatenate((ims[0], ims[1]), axis=self.axis)

    def check_margin(self, mask):
        """Check if all voxels marked for resegmentation are within margin."""
        # NOTE: axis=0 hijacked for quads

        msum = False

        if self.axis == 1 or self.axis == 0:
            m1sum = np.sum(mask[:, 0, :])
            m2sum = np.sum(mask[:, -1, :])
            msum = msum | bool(m1sum) | bool(m2sum)

        if self.axis == 2 or self.axis == 0:
            m1sum = np.sum(mask[:, :, 0])
            m2sum = np.sum(mask[:, :, -1])
            msum = msum | bool(m1sum) | bool(m2sum)

        return msum

    def set_to_margin_slices(self, segs, include_margin=False):
        """"Set slices for selecting margins."""
        # axis=2, margin=64, n=2,

        def slice_ll(margin, margin_n):
            return slice(margin, margin_n, 1)

        def slice_ur(seg, axis, margin, margin_n):
            start = seg.dims[axis] - margin_n
            stop = seg.dims[axis] - margin
            return slice(start, stop, 1)

        mn = self.margin * self.n
        if include_margin:  # select data including the full margin strip
            m = 0
        else:  # select only the part within the block-proper (i.e. minus margins)
            m = self.margin

        axes = {2: 'x', 1: 'y', 0: 'z'}
        if self.axis > 0:
            ax = segs[0].axlab.index(axes[self.axis])
            segs[0].slices[ax] = slice_ur(segs[0], ax, m, mn)  # left block
            segs[1].slices[ax] = slice_ll(m, mn)  # right block

        elif self.axis == 0:  # NOTE: axis=0 hijacked for quads
            ax2 = segs[0].axlab.index(axes[2])  # 'x'
            ax1 = segs[0].axlab.index(axes[1])  # 'y'
            # left-bottom block
            segs[0].slices[ax2] = slice_ur(segs[0], ax2, m, mn)
            segs[0].slices[ax1] = slice_ur(segs[0], ax1, m, mn)
            # right-bottom block
            segs[1].slices[ax2] = slice_ll(m, mn)
            segs[1].slices[ax1] = slice_ur(segs[1], ax1, m, mn)
            # left-top block
            segs[2].slices[ax2] = slice_ur(segs[2], ax2, m, mn)
            segs[2].slices[ax1] = slice_ll(m, mn)
            # right-top block
            segs[3].slices[ax2] = slice_ll(m, mn)
            segs[3].slices[ax1] = slice_ll(m, mn)

    def write_margin(self, ims, data):
        """Write margin datablocks back to file."""

        def update_vol(im, d):
            if isinstance(im, LabelImage):
                # NOTE: is it even possible that im.ds.attrs['maxlabel'] > np.amax(d)?
                # new maxlabel of the block is  the max of the old and the max of the newly written subblock
                im.ds.attrs['maxlabel'] = max(im.ds.attrs['maxlabel'], np.amax(d))
                comps = im.split_path()
                print('new maxlabel for {}: {:d}'.format(comps['fname'], im.ds.attrs['maxlabel']))
            im.write(d)

        mn = self.margin * self.n

        self.set_to_margin_slices(ims, include_margin=True)
        if self.axis == 2:
            update_vol(ims[0], data[:, :, :mn])
            update_vol(ims[1], data[:, :, -mn:])
        elif self.axis == 1:
            update_vol(ims[0], data[:, :mn, :])
            update_vol(ims[1], data[:, -mn:, :])
        elif self.axis == 0:  # NOTE: axis=0 hijacked for quads
            update_vol(ims[0], data[:, :mn, :mn])
            update_vol(ims[1], data[:, :mn, -mn:])
            update_vol(ims[2], data[:, -mn:, :mn])
            update_vol(ims[3], data[:, -mn:, -mn:])

    def get_labels(self, segs_marg, include_margin=False, bg=set([0])):
        """Find the labels on the boundary of blocks."""

        # NOTE: if include_margin: <touching the boundary and into the margin>
        margin = self.margin
        b = margin if include_margin else 1

        if self.axis == 2:

            seg1_labels = set(np.unique(segs_marg[0][:, :, -b:]))
            seg1_labels -= bg
            seg2_labels = set(np.unique(segs_marg[1][:, :, :b]))
            seg2_labels -= bg

            return seg1_labels, seg2_labels

        elif self.axis == 1:

            seg1_labels = set(np.unique(segs_marg[0][:, -b:, :]))
            seg1_labels -= bg
            seg2_labels = set(np.unique(segs_marg[1][:, :b, :]))
            seg2_labels -= bg

            return seg1_labels, seg2_labels

        elif self.axis == 0:  # NOTE: axis=0 hijacked for quads

            seg1_labels = set(np.unique(segs_marg[0][:, -margin:, -b:]))
            seg1_labels |= set(np.unique(segs_marg[0][:, -b:, -margin:]))
            seg1_labels -= bg
            seg2_labels = set(np.unique(segs_marg[1][:, -margin:, :b]))
            seg2_labels |= set(np.unique(segs_marg[1][:, -b:, :margin]))
            seg2_labels -= bg
            seg3_labels = set(np.unique(segs_marg[2][:, :margin, -b:]))
            seg3_labels |= set(np.unique(segs_marg[2][:, :b, -margin:]))
            seg3_labels -= bg
            seg4_labels = set(np.unique(segs_marg[3][:, :margin, :b]))
            seg4_labels |= set(np.unique(segs_marg[3][:, :b, :margin]))
            seg4_labels -= bg

            return seg1_labels, seg2_labels, seg3_labels, seg4_labels

    def _get_info_dict(self, **kwargs):

        if 'parameters' in kwargs.keys():
            p = kwargs['parameters']
        else:
            p = self.load_dumped_pars()
            kwargs['parameters'] = p

        return kwargs

    def _gen_subgrid(self, f, gs, channel=None):
        """Generate the axes for printing the report."""

        axdict, titles = {}, {}
        gs0 = gs.subgridspec(2, 1, height_ratios=[1, 10])
        axdict['p'] = self._report_axes_pars(f, gs0[0])

        gs01 = gs0[1].subgridspec(4, 1)

        t = ['data', 'orig', 'reseg', 'final']
        voldict = {name: {'idx': i, 'title': name} for i, name in enumerate(t)}

        for k, vd in voldict.items():
            titles[k] = (vd['title'], 'lcm', 0)
            axdict[k] = gen_orthoplot(f, gs01[vd['idx']], 5, 1)

        self._add_titles(axdict, titles)

        return axdict

    def _plot_images(self, f, axdict, info_dict={}):
        """Plot graph with shading image."""

        cslcs = info_dict['centreslices']
        for k, v in cslcs.items():
            cslcs[k]['x'] = cslcs[k]['x'].transpose()

        aspects = ['equal', 'auto', 'auto']
        for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

            data = cslcs['data'][dim]

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

            ax = axdict['data'][i]
            ax.imshow(data, aspect=aspect, cmap='gray')
            ax.axis('off')
            for v in ['orig', 'reseg', 'final']:
                ax = axdict[v][i]
                labels = cslcs[v][dim]
                clabels = label2rgb(labels, image=None, bg_label=0)
                ax.imshow(clabels, aspect=aspect)
                ax.axis('off')

            if lines:
                for m in pp:
                    if (dim == 'z') & (info_dict['axis'] == 0):
                        for v in ['data', 'orig', 'reseg', 'final']:
                            axdict[v][i].plot(fs, [m, m], '--', linewidth=1, color='w')
                            axdict[v][i].plot([m, m], fs, '--', linewidth=1, color='w')
                    elif (dim == 'z') & (info_dict['axis'] == 1):
                        for v in ['data', 'orig', 'reseg', 'final']:
                            axdict[v][i].plot(fs, [m, m], '--', linewidth=1, color='w')
                    elif (dim == 'z') & (info_dict['axis'] == 2):
                        for v in ['data', 'orig', 'reseg', 'final']:
                            axdict[v][i].plot([m, m], fs, '--', linewidth=1, color='w')
                    else:
                        for v in ['data', 'orig', 'reseg', 'final']:
                            axdict[v][i].plot([m, m], fs, '--', linewidth=1, color='w')

    def view(self, input=[], images=[], labels=[], settings={}):

        images = images or self._images
        labels = labels or self._labels

        if isinstance(input, str):
            super().view(input, images, labels, settings)
        elif type(input) == int or type(input) == float:
            filepath = self._abs(self.outputpaths['estimate']['blockfiles'].format(b=input))
            super().view(filepath, images, labels, settings)
        else:
            input = input or [0, 1]
            super().view_blocks(input, images, labels, settings)


def write_output(outpath, out, props, imtype='Label'):
    """Write data to an image on disk."""

    props['dtype'] = out.dtype
    if imtype == 'Label':
        mo = LabelImage(outpath, **props)
    elif imtype == 'Mask':
        mo = MaskImage(outpath, **props)
    else:
        mo = Image(outpath, **props)
    mo.create()
    mo.write(out)

    return mo


def read_image(im_info, ids='segm/labels', imtype='Label'):
    """"Read a h5 dataset as Image object."""

    filepath = im_info['filepath']
    image_in = f'{filepath}/{ids}'

    if imtype == 'Label':
        im = LabelImage(image_in)
    elif imtype == 'Mask':
        im = MaskImage(image_in)
    else:
        im = Image(image_in)

    im.load(load_data=False)

    if imtype == 'Label':
        im.set_maxlabel()

    return im


def get_maxlabels_from_attribute(filelist, ids, maxlabelfile):

    maxlabels = []

    for datafile in filelist:
        image_in = '{}/{}'.format(datafile, ids)
        im = Image(image_in, permission='r')
        im.load(load_data=False)
        maxlabels.append(im.ds.attrs['maxlabel'])
        im.close()

    if maxlabelfile:
        np.savetxt(maxlabelfile, maxlabels, fmt='%d')

    return maxlabels


if __name__ == "__main__":
    main(sys.argv[1:])
