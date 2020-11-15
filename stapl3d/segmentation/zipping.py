#!/usr/bin/env python

"""Resegment the dataset block boundaries.

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

from stapl3d import (
    parse_args_common,
    get_outputdir,
    get_blockfiles,
    get_blockinfo,
    get_params,
    get_n_workers,
    get_paths,
    Image,
    LabelImage,
    MaskImage,
    wmeMPI,
    split_filename,
    )

from stapl3d.reporting import (
    gen_orthoplot,
    load_parameters,
    get_centreslice,
    get_centreslices,
    get_zyx_medians,
    get_cslc,
    )


def main(argv):
    """"Resegment the dataset block boundaries.

    """

    step_ids = ['relabel', 'copyblocks', 'zipping']
    fun_selector = {
        'relabel': relabel,
        'copyblocks': copyblocks,
        'zipping': estimate,
        }

    args, mapper = parse_args_common(step_ids, fun_selector, *argv)

    for step, step_id in mapper.items():
        fun_selector[step](
            args.image_in,
            args.parameter_file,
            step_id,
            args.outputdir,
            args.n_workers,
            )


def estimate(
    image_in,
    parameter_file,
    step_id='zipping',
    outputdir='',
    n_workers=0,
    blocksize=[],
    blockmargin=[],
    blockrange=[],
    blocks=[],
    grp='segm',
    ids='labels_zip',
    postfix='',
    ids_nucl='',
    ids_memb_chan='memb/prep',
    find_peaks=False,
    peaks_thr=1.0,
    peaks_size=[11, 19, 19],
    peaks_dil_footprint=[3, 7, 7],
    compactness=0.80,
    ):
    """Correct z-stack shading."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, fallback='blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')
    segparams = get_params(locals().copy(), parameter_file, 'segmentation')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    blocksize, blockmargin, _ = get_blockinfo(image_in, parameter_file, params)

    path_int = '{}/{}'.format(params['grp'], params['ids'])
    dataset = os.path.splitext(get_paths(image_in)['fname'])[0]
    filename = '{}_maxlabels_{}.txt'.format(dataset, params['ids'])
    maxlabelfile = os.path.join(outputdir, filename)
    maxlabels = get_maxlabels_from_attribute(filepaths, path_int, maxlabelfile)

    # Arguments to `resegment_block_boundaries`
    images_in = ['{}/{}'.format(filepath, path_int) for filepath in filepaths]
    blocksize = blocksize[:3]
    blockmargin = blockmargin[:3]
    axis = 0
    seamnumbers = [-1, -1, -1]
    mask_dataset = ''
    relabel = False
    maxlabel = maxlabelfile
    in_place = True
    outputstem = os.path.join(outputdir, dataset)
    save_steps = False
    args = [images_in, blocksize, blockmargin, axis, seamnumbers, mask_dataset,
            relabel, maxlabel, in_place,
            segparams['ids_nucl'], segparams['ids_memb_chan'],
            segparams['find_peaks'], segparams['peaks_thr'], segparams['peaks_size'],
            segparams['peaks_dil_footprint'], segparams['compactness'],
            outputstem, save_steps]

    n_seams_yx, seamgrid = get_zip_layout(image_in, blocksize)

    n_workers = get_n_workers(len(blocks), subparams)

    for axis, n_seams in zip([1, 2], n_seams_yx):
        n_proc = min(n_workers, int(np.ceil(n_seams / 2)))
        for offset in [0, 1]:
            compute_zip_step(
                args, axis, seamgrid,
                starts=[offset, 0], stops=[n_seams, 1], steps=[2, 2],
                n_proc=n_proc,
            )
            maxlabels = get_maxlabels_from_attribute(filepaths, path_int, maxlabelfile)

    for start_y in [0, 1]:
        for start_x in [0, 1]:
            compute_zip_step(
                args, axis=0, seamgrid=seamgrid,
                starts=[start_y, start_x], stops=n_seams_yx, steps=[2, 2],
                n_proc=n_workers,
            )
            maxlabels = get_maxlabels_from_attribute(filepaths, path_int, maxlabelfile)


def get_zip_layout(image_in, blocksize):
    """Determine the number of seams in the blocked dataset."""

    im = Image(image_in)
    im.load(load_data=False)
    X = im.dims[im.axlab.index('x')]
    Y = im.dims[im.axlab.index('y')]
    im.close()
    from math import ceil
    nx = int(ceil(X/blocksize[im.axlab.index('x')]))
    ny = int(ceil(Y/blocksize[im.axlab.index('y')]))
    n_seams_yx = [ny - 1, nx - 1]
    seams = list(range(np.prod(n_seams_yx)))
    seamgrid = np.reshape(seams, n_seams_yx)

    return n_seams_yx, seamgrid


def get_arglist(args, axis, starts, stops, steps):
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
            args[3] = axis
            if axis == 0:
                args[4] = [seamnumbers[d] if d != axis else -1 for d in [0, 1, 2]]
            else:
                args[4] = [seam_y if d == axis else -1 for d in [0, 1, 2]]
            arglist.append(tuple(args))

    return arglist


def compute_zip_step(args, axis, seamgrid, starts, stops, steps, n_proc):
    """Compute the zip-step."""

    arglist = get_arglist(args, axis, starts, stops, steps)
    print('submitting {:3d} jobs over {:3d} processes'.format(len(arglist), n_proc))

    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.starmap(resegment_block_boundaries, arglist)


def resegment_block_boundaries(
    images_in,
    blocksize,
    blockmargin=[0, 64, 64],
    axis=2,
    seamnumbers=[-1, -1, -1],
    mask_dataset='',
    relabel=False,
    maxlabel='',
    in_place=False,
    ids_nucl='',
    ids_memb_chan='memb/prep',
    find_peaks=False,
    peaks_thr=1.16,
    peaks_size=[11, 19, 19],
    peaks_dil_footprint=[3, 7, 7],
    compactness=0.80,
    outputstem='',
    save_steps=False,
    ):
    """Resegment the dataset block boundaries."""

    # NB: images_in are sorted in xyz-order while most processing will map to zyx
    # TODO: may want to switch to xyz here; or perhaps sort according to zyx for consistency
    images_in.sort()

    step = 'resegment'

    # paths = get_paths(images_in[0], -1, 0, outputstem, step, save_steps)
    paths = {}
    paths['out_base'] = outputstem
    paths['out_h5'] = '{}.h5/{}'.format(paths['out_base'], '{}')
    paths['main'] = paths['steps'] = paths['out_h5']
    paths['params'] = '{}-params.pickle'.format(paths['out_base'], step)
    report = {
        'parameters': locals(),
        'paths': paths,
        'medians': {},
        'centreslices': {}
        }

    info, filelist, ids = get_block_info(images_in, blocksize, blockmargin)
    blockmap = get_blockmap(info)
    # FIXME: assuming call with single axis for now
    seamnumber = seamgrid_ravel_multi_index(blockmap, seamnumbers, axis)
    maxlabel = prep_maxlabel(maxlabel, seamnumber, filelist, ids)
    print('starting with maxlabel = {:12d}\n'.format(maxlabel))

    report['j'] = 0
    report['seam'] = seamnumber
    pairs = get_seam_pairs(blockmap, seamnumbers, axis)

    for pair in pairs:

        print('{:03d}: pair {} over axis {}'.format(report['j'], pair, axis))

        margin = blockmargin[2] if axis == 0 else blockmargin[axis]

        info_ims = tuple(info[idx] for idx in pair)

        n_max = find_nmax(info_ims, axis, margin)
        n_max = min(10, n_max)
        if n_max < 2:
            print('n_max', n_max)
            continue

        n = min(n_max, 3)

        report['axis'] = axis
        report['margin'] = margin
        maxlabel, report = process_pair(
            info_ims,
            ids,
            margin,
            axis,
            maxlabel,
            n,
            n_max,
            ids_nucl,
            ids_memb_chan,
            find_peaks,
            peaks_thr,
            peaks_size,
            peaks_dil_footprint,
            compactness,
            report)
        report['j'] += 1

        print('maxlabel = {:08d}\n'.format(maxlabel))


def get_block_info(images_in, blocksize, margins=[0, 64, 64]):
    """Get info on the blocks of a dataset."""

    inputfiles = []
    for image_in in images_in:
        fstem, ids = image_in.split('.h5')
        inputfiles.append('{}.h5'.format(fstem))

    info = {}
    for i, inputfile in enumerate(inputfiles):
        # NOTE: after inputfiles.sort(), i is a linear index
        # into image info incrementing first z then y then x
        info[i] = split_filename(inputfile)[0]
        info[i]['inputfile'] = inputfile
        zyx = [info[i][dim] + margins[j] if info[i][dim] > 0 else 0
               for j, dim in enumerate('zyx')]
        info[i]['blockcoords'] = [
            int(zyx[0] / blocksize[0]),
            int(zyx[1] / blocksize[1]),
            int(zyx[2] / blocksize[2]),
            ]

    return info, inputfiles, ids[1:]


def get_blockmap(info):
    """Get a map of block indices."""

    ds = np.amax(np.array([v['blockcoords'] for k, v in info.items()]), axis=0) + 1
    blockmap = np.zeros(ds, dtype='uint16')
    for k, v in info.items():
        bc = v['blockcoords']
        blockmap[bc[0], bc[1], bc[2]] = k

    return blockmap


def get_seam_pairs(blockmap, seamnumbers, axis):

    ad = {0: {'ax': [1, 2], 'tp': [0, 1], 'sh': (1,4)},
          1: {'ax': [axis], 'tp': [1, 0], 'sh': (-1, 2)},
          2: {'ax': [axis], 'tp': [0, 1], 'sh': (-1, 2)}}

    slcs = [slice(seamnumbers[d], seamnumbers[d] + 2)
            if d in ad[axis]['ax'] else slice(None)
            for d in range(3)]
    pairs = np.squeeze(blockmap[tuple(slcs)])

    pairs = np.reshape(np.transpose(pairs, ad[axis]['tp']), ad[axis]['sh'])

    return pairs


def prep_maxlabel(maxlabel, seamnumber, filelist='', ids='', maxlabel_margin=100000):

    if maxlabel == 'attrs':
        maxlabels = get_maxlabels_from_attribute(filelist, ids, '')
        maxlabel = max(maxlabels)
        src = 'attributes'

    try:
        maxlabel = int(maxlabel)
        src = 'integer argument'
    except ValueError:
        maxlabels = np.loadtxt(maxlabel, dtype=np.uint32)
        maxlabel = max(maxlabels)
        src = 'textfile'

    print('read maxlabel {:12d} from {}'.format(maxlabel, src))

    maxlabel += seamnumber * maxlabel_margin

    return maxlabel


def seamgrid_ravel_multi_index(blockmap, seamnumbers, axis):

    if axis == 0:
        seamgrid_shape = [blockmap.shape[1] - 1, blockmap.shape[2] - 1]
        seamnumber = np.ravel_multi_index(seamnumbers[1:], seamgrid_shape)
    else:
        seamnumber = seamnumbers[axis]

    print('linear seamindex = {}'.format(seamnumber))

    return seamnumber


def find_nmax(info_ims, axis=2, margin=64):
    """Calculate how many margin-blocks fit into the dataset."""

    sizes = []
    if axis == 2:
        sizes += [info_im['X'] - info_im['x'] for info_im in info_ims]
    elif axis == 1:
        sizes += [info_im['Y'] - info_im['y'] for info_im in info_ims]
    elif axis == 0:
        sizes += [info_im['X'] - info_im['x'] for info_im in info_ims]
        sizes += [info_im['Y'] - info_im['y'] for info_im in info_ims]

    n_max = int(np.amin(np.array(sizes)) / margin)

    return n_max


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

    fname = '{}_{}'.format(im_info['base'], im_info['postfix'])
    fstem = os.path.join(im_info['datadir'], fname)
    if imtype == 'Label':
        im = LabelImage('{}.h5/{}'.format(fstem, ids))
    elif imtype == 'Mask':
        im = MaskImage('{}.h5/{}'.format(fstem, ids))
    else:
        im = Image('{}.h5/{}'.format(fstem, ids))
    im.load(load_data=False)
    if imtype == 'Label':
        im.set_maxlabel()

    return im


def read_images(info_ims, ids='segm/labels', imtype='Label',
                axis=2, margin=64, n=2, include_margin=False, concat=False):
    """Read a set of block and slice along the block margins."""

    segs = tuple(read_image(info_im, ids=ids, imtype=imtype) for info_im in info_ims)

    set_to_margin_slices(segs, axis, margin, n, include_margin)

    segs_marg = tuple(seg.slice_dataset() for seg in segs)

    if concat:
        segs_marg = concat_images(segs_marg, axis)

    return segs, segs_marg


def set_to_margin_slices(segs, axis=2, margin=64, n=2, include_margin=False):
    """"Set slices for selecting margins."""

    def slice_ll(margin, margin_n):
        return slice(margin, margin_n, 1)

    def slice_ur(seg, axis, margin, margin_n):
        start = seg.dims[axis] - margin_n
        stop = seg.dims[axis] - margin
        return slice(start, stop, 1)

    mn = margin * n
    if include_margin:  # select data including the full margin strip
        m = 0
    else:  # select only the part within the block-proper (i.e. minus margins)
        m = margin

    if axis > 0:
        segs[0].slices[axis] = slice_ur(segs[0], axis, m, mn)  # left block
        segs[1].slices[axis] = slice_ll(m, mn)  # right block

    elif axis == 0:  # NOTE: axis=0 hijacked for quads
        # left-bottom block
        segs[0].slices[2] = slice_ur(segs[0], 2, m, mn)
        segs[0].slices[1] = slice_ur(segs[0], 1, m, mn)
        # right-bottom block
        segs[1].slices[2] = slice_ll(m, mn)
        segs[1].slices[1] = slice_ur(segs[1], 1, m, mn)
        # left-top block
        segs[2].slices[2] = slice_ur(segs[2], 2, m, mn)
        segs[2].slices[1] = slice_ll(m, mn)
        # right-top block
        segs[3].slices[2] = slice_ll(m, mn)
        segs[3].slices[1] = slice_ll(m, mn)


def get_labels(segs_marg, axis=2, margin=64, include_margin=False, bg=set([0])):
    """Find the labels on the boundary of blocks."""

    # NOTE: if include_margin: <touching the boundary and into the margin>
    b = margin if include_margin else 1

    if axis == 2:

        seg1_labels = set(np.unique(segs_marg[0][:, :, -b:]))
        seg1_labels -= bg
        seg2_labels = set(np.unique(segs_marg[1][:, :, :b]))
        seg2_labels -= bg

        return seg1_labels, seg2_labels

    elif axis == 1:

        seg1_labels = set(np.unique(segs_marg[0][:, -b:, :]))
        seg1_labels -= bg
        seg2_labels = set(np.unique(segs_marg[1][:, :b, :]))
        seg2_labels -= bg

        return seg1_labels, seg2_labels

    elif axis == 0:  # NOTE: axis=0 hijacked for quads

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


def check_margin(mask, axis):
    """Check if all voxels marked for resegmentation are within margin."""

    msum = False

    if axis == 1 or axis == 0:  # NOTE: axis=0 hijacked for quads
        m1sum = np.sum(mask[:, 0, :])
        m2sum = np.sum(mask[:, -1, :])
        msum = msum | bool(m1sum) | bool(m2sum)

    if axis == 2 or axis == 0:  # NOTE: axis=0 hijacked for quads
        m1sum = np.sum(mask[:, :, 0])
        m2sum = np.sum(mask[:, :, -1])
        msum = msum | bool(m1sum) | bool(m2sum)

    return msum


def write_margin(ims, data, axis, margin, n):
    """Write margin datablocks back to file."""

    def update_vol(im, d):
        if isinstance(im, LabelImage):
            # NOTE: is it even possible that im.ds.attrs['maxlabel'] > np.amax(d)?
            # new maxlabel of the block is  the max of the old and the max of the newly written subblock
            im.ds.attrs['maxlabel'] = max(im.ds.attrs['maxlabel'], np.amax(d))
            comps = im.split_path()
            print('new maxlabel for {}: {:d}'.format(comps['fname'], im.ds.attrs['maxlabel']))
        im.write(d)

    mn = margin * n

    set_to_margin_slices(ims, axis, margin, n, include_margin=True)
    if axis == 2:
        update_vol(ims[0], data[:, :, :mn])
        update_vol(ims[1], data[:, :, -mn:])
    elif axis == 1:
        update_vol(ims[0], data[:, :mn, :])
        update_vol(ims[1], data[:, -mn:, :])
    elif axis == 0:  # NOTE: axis=0 hijacked for quads
        update_vol(ims[0], data[:, :mn, :mn])
        update_vol(ims[1], data[:, :mn, -mn:])
        update_vol(ims[2], data[:, -mn:, :mn])
        update_vol(ims[3], data[:, -mn:, -mn:])


def get_resegmentation_mask(info_ims, ids, axis=2, margin=64, n=2):
    """Find the mask of segments for resegmentation."""

    # read the margin of the labelimage/reseg_mask blocks
    segs, segs_dss = read_images(info_ims, ids, 'Label',
                                 axis, margin, n, include_margin=False)
    masks, masks_dss = read_images(info_ims,
                                   '{}_reseg_mask'.format(ids), 'Mask',
                                   axis, margin, n, include_margin=False)

    # find the labels that are on the seam and check if there any labels in the combined set
    # labelsets = get_labels(segs_dss, axis, margin, include_margin=True)
    ### NOTE: switching to include margin to test/fix the bug that cells in the resegmentation appear to have much smaller volume
    ### NOTE: the bug appears to be in extracting regionprops rather than resegmentation
    ### TODO: test a version where a new peak detection is performed
    # TODO: keep seam-labelset for next iteration
    labelsets = get_labels(segs_dss, axis, margin, include_margin=False)
    for labelset, seg in zip(labelsets, segs):
        comps = seg.split_path()
        print('found {:d} labels on boundary in {}'.format(len(labelset), comps['fname']))
    is_empty = True if not set().union(*labelsets) else False
    if is_empty:
        return True, None, None, None, None, None

    # create boolean forward maps for the resegmentation mask
    fw_maps = tuple([True if l in labelset else False
                     for l in range(0, seg.maxlabel + 1)]
                    for labelset, seg in zip(labelsets, segs))

    masks_reseg = tuple(np.array(fw_map)[seg_dss]
                        for fw_map, seg_dss in zip(fw_maps, segs_dss))

    # concatenate the margins of the block set
    masks_reseg = concat_images(masks_reseg, axis)
    masks_ds = concat_images(masks_dss, axis)
    segs_ds = concat_images(segs_dss, axis)

    return is_empty, segs, segs_ds, masks, masks_ds, masks_reseg


def process_pair(
    info_ims,
    ids='segm/labels_zip',
    margin=64,
    axis=2,
    maxlabel=1,
    n=2,
    n_max=4,
    ids_nucl='',
    ids_memb_chan='memb/prep',
    find_peaks=False,
    peaks_thr=1.16,
    peaks_size=[11, 19, 19],
    peaks_dil_footprint=[3, 7, 7],
    compactness=0.80,
    report=None,
    ):
    """Resegment the boundaries between pairs/quads of blocks."""

    while True:

        # get a resegmentation mask
        reseg = get_resegmentation_mask(info_ims, ids, axis, margin, n)
        is_empty, segs, segs_ds, masks, masks_ds, mask = reseg

        # no need to do anything if there are no labels on the block boundaries
        if is_empty:
            print('got an empty labelset')
            return maxlabel, report

        # see if there is reason to increase the margin
        invalid = check_margin(mask, axis)
        if invalid and n < n_max:
            n += 1
            print('increased margin_n to {} x {} pixels'.format(n, margin))
        else:
            break

    c_slcs = {dim: get_cslc(segs_ds, ax) for ax, dim in enumerate('zyx')}
    report['centreslices']['orig'] = c_slcs

    if ids_nucl:
        seeds, seeds_ds = read_images(info_ims, ids_nucl, 'Image',
                                      axis, margin, n, include_margin=False,
                                      concat=True)
    else:
        edts, edts_ds = read_images(info_ims, 'segm/seeds_mask_edt', 'Image',
                                    axis, margin, n, include_margin=False,
                                    concat=True)
        peaks, peaks_ds = read_images(info_ims, 'segm/peaks', 'Mask',
                                      axis, margin, n, include_margin=False,
                                      concat=True)
    if ids_memb_chan:
        membs, membs_ds = read_images(info_ims, ids_memb_chan, 'Image',
                                      axis, margin, n, include_margin=False,
                                      concat=True)

    if False:  # FIXME
        csol, csol_ds = read_images(info_ims, 'segm/labels_csol_mask', 'Mask',
                                    axis, margin, n, include_margin=False,
                                    concat=True)

    # FIXME: need to handle double peaks here?
    if find_peaks:
        from stapl3d.segmentation.segment import find_local_maxima
        new_peaks = find_local_maxima(edts_ds, peaks_size, peaks_thr)
        peaks_ds[mask] = new_peaks[mask]
        write_margin(peaks, peaks_ds, axis, margin, n)
        save_steps = False
        if save_steps:
            peaks_dil, peaks_dil_ds = read_images(info_ims, 'segm/seeds_peaks_dil', 'Mask',
                                                  axis, margin, n, include_margin=False,
                                                  concat=True)
            from stapl3d.segmentation.segment import create_footprint
            footprint = create_footprint(peaks_dil_footprint)
            from skimage.morphology import binary_dilation
            new_peaks_dil = binary_dilation(new_peaks, selem=footprint)
            peaks_dil_ds[mask] = new_peaks_dil[mask]
            write_margin(peaks_dil, peaks_dil_ds, axis, margin, n)

    if ids_nucl:
        seeds_ds[~mask] = 0
        print('{:10d} new peaks are used'.format(len(np.unique(seeds_ds))))
    else:
        peaks_ds[~mask] = 0
        seeds_ds, n_labels = ndi.label(peaks_ds)
        print('{:10d} new peaks are used'.format(n_labels))

    # FIXME: write the segment function with mask option and call it from here
    if False:
        wsmask = np.logical_and(mask, edts_ds > peaks_thr)
        ws = watershed(-edts_ds, peaks_labeled, mask=wsmask)
        wsmask = np.logical_and(mask, ~csol_ds)
        cells = watershed(memb_ds, ws, mask=wsmask)
        if ids_memb_chan:
            ws = watershed(membs_ds, ws, mask=mask, compactness=compactness)
    else:
        if ids_nucl:
            ws = seeds_ds
        else:
            wsmask = np.logical_and(mask, edts_ds > peaks_thr)
            ws = watershed(-edts_ds, seeds_ds, mask=wsmask)
        if ids_memb_chan:
            ws = watershed(membs_ds, ws, mask=mask, compactness=compactness)

    ws_ulabels = np.unique(ws)
    ws_max = max(ws_ulabels)
    n_newlabels = len(ws_ulabels) - 1

    print('{:10d} labels in final watershed with maxlabel={:10d}'.format(n_newlabels, ws_max))
    print('incrementing ws by maxlabel {:10d}'.format(maxlabel))
    ws += maxlabel
    print('incrementing maxlabel by ws_max {:10d}'.format(ws_max))
    maxlabel += ws_max

    segs_ds[mask] = ws[mask]
    write_margin(segs, segs_ds, axis, margin, n)  # resegmentation

    mask_reseg = masks_ds | mask
    write_margin(masks, mask_reseg, axis, margin, n)  # resegmentation mask

    # TODO: adapt blocks_ds to identify unique segments
    # FIXME: they may cross into different blocks!
    # NOTE: may fix this by directly exporting regionprops as well here???
    # NB: need a bias field-corrected imaris file
    # NB: need to work out the slicing of the margins into the imaris file;
    # NB: or need to cut the bias-field corrected volume into blocks
    # NB: need to exclude segments in the resegmentation mask for the regular regionprops
    # export_regionprops(seg_path, data_path, bias_path='', csv_path='')
    # write_margin(blocks, blocks_ds, axis, margin, n)  # block boundaries

    if ids_memb_chan:
        c_slcs = {dim: get_cslc(membs_ds, ax) for ax, dim in enumerate('zyx')}
        report['centreslices']['data'] = c_slcs
    else:
        c_slcs = {dim: get_cslc(edts_ds, ax) for ax, dim in enumerate('zyx')}
        report['centreslices']['data'] = c_slcs
    c_slcs = {dim: get_cslc(ws, ax) for ax, dim in enumerate('zyx')}
    report['centreslices']['reseg'] = c_slcs
    c_slcs = {dim: get_cslc(segs_ds, ax) for ax, dim in enumerate('zyx')}
    report['centreslices']['final'] = c_slcs

    im_info = info_ims[0]
    fname = '{}_{}'.format(im_info['base'], im_info['postfix'])
    fstem = os.path.join(im_info['datadir'], fname)
    fpath = '{}.h5/{}'.format(fstem, 'segm/labels_memb_del')
    generate_report(fpath, report)

    return maxlabel, report


def concat_images(ims, axis=2):
    """Concatenate the margins of neighbouring datablocks."""

    if axis == 0:  # NOTE: axis=0 hijacked for quads
        return np.concatenate((np.concatenate((ims[0], ims[1]), axis=2),
                               np.concatenate((ims[2], ims[3]), axis=2)),
                              axis=1)
    else:
        return np.concatenate((ims[0], ims[1]), axis=axis)


def create_resegmentation_mask_datasets(info):
    """Create datasets to hold the full resegmentation mask."""

    for k,v in info.items():
        print(k, v['postfix'])
        seg = read_image(v, ids='segm/labels_memb_del', imtype='Label')
        data = np.zeros(seg.dims, dtype='bool')
        comps = seg.split_path()
        outpath = '{}.h5{}_{}'.format(comps['base'], comps['int'], 'fixmask')
        im = write_output(outpath, data, props=seg.get_props(), imtype='Mask')
        im.close()


def relabel(
    image_in,
    parameter_file,
    step_id='relabel',
    outputdir='',
    n_workers=0,
    blocks=[],
    grp='segm',
    ids='labels',
    postfix='_zip',
    ):
    """Correct z-stack shading."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, 'blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    path_int = '{}/{}'.format(params['grp'], params['ids'])
    dataset = os.path.splitext(get_paths(image_in)['fname'])[0]
    filename = '{}_maxlabels_{}.txt'.format(dataset, params['ids'])
    maxlabelfile = os.path.join(outputdir, filename)
    maxlabels = get_maxlabels_from_attribute(filepaths, path_int, maxlabelfile)

    arglist = [
        (
            '{}/{}'.format(filepath, path_int),
            block_idx,
            maxlabelfile,
            params['postfix'],
        )
        for block_idx, filepath in zip(blocks, filepaths)]

    n_workers = get_n_workers(len(blocks), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(relabel_parallel, arglist)


def relabel_parallel(image_in, block_idx, maxlabelfile, pf='_zip'):

    maxlabels = np.loadtxt(maxlabelfile, dtype=np.uint32)
    maxlabel = np.sum(maxlabels[:block_idx])

    seg = LabelImage(image_in)
    seg.load(load_data=False)

    relabel_block(seg, pf, maxlabel)


def relabel_block(im, pf='_zip', maxlabel=1, bg_label=0, force_sequential=False):
    """Relabel dataset sequentially."""

    data = im.slice_dataset()
    mask = data == bg_label

    if force_sequential:
        data, fw, _ = relabel_sequential(data, offset=maxlabel)
    else:
        data[~mask] += maxlabel

    comps = im.split_path()
    outpath = '{}.h5{}{}'.format(comps['base'], comps['int'], pf)
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

    return mo


def copyblocks(
    image_in,
    parameter_file,
    step_id='copyblocks',
    outputdir='',
    n_workers=0,
    blocks=[],
    grp='segm',
    ids='labels_zip',
    postfix='',
    ):
    """Correct z-stack shading."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, 'blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    path_int = '{}/{}'.format(params['grp'], params['ids'])
    dataset = os.path.splitext(get_paths(image_in)['fname'])[0]
    filename = '{}_maxlabels_{}.txt'.format(dataset, params['ids'])
    maxlabelfile = os.path.join(outputdir, filename)
    maxlabels = get_maxlabels_from_attribute(filepaths, path_int, maxlabelfile)

    arglist = [
        (
            '{}/{}'.format(filepath, path_int),
            block_idx,
            params['postfix'],
        )
        for block_idx, filepath in zip(blocks, filepaths)]

    n_workers = get_n_workers(len(blocks), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(copy_blocks_parallel, arglist)


def copy_blocks_parallel(image_in, block_idx, postfix=''):

    im = LabelImage(image_in)
    im.load(load_data=False)

    vols = {}
    if postfix:
        vols['{}'.format(postfix)] = ['Label', 'uint32']
    vols['{}_reseg_mask'.format(postfix)] = ['Mask', 'bool']
    # vols['block_idxs'] = ['Label', 'uint16']
    # vols = {postfix: ['Mask', 'bool']}

    for pf, (imtype, dtype) in vols.items():
        copy_h5_dataset(im, imtype, pf, dtype, k=block_idx)


def copy_h5_dataset(im, imtype='Label', pf='_fix', dtype='uint32', k=0):
    """Copy an image to a new h5 dataset."""

    comps = im.split_path()
    if '_reseg_mask' in pf: #  or '_peaks' in pf
        outpath = '{}.h5{}{}'.format(comps['base'], comps['int'], pf)
        data = np.zeros(im.dims, dtype='bool')
    elif pf == 'block_idxs':
        outpath = '{}.h5/segm/{}'.format(comps['base'], pf)
        data = np.ones(im.dims, dtype='uint16') * (k+1)
    else:
        outpath = '{}.h5{}{}'.format(comps['base'], comps['int'], pf)
        data = im.slice_dataset()

    mo = write_output(outpath, data, props=im.get_props(), imtype=imtype)

    # TODO make ulabels/maxlabel standard attributes to write into LabelImages.
    if imtype == 'Label':
        try:
            maxlabel = im.ds.attrs['maxlabel']
        except:
            mo.set_maxlabel()
            maxlabel = mo.maxlabel
        mo.ds.attrs.create('maxlabel', maxlabel, dtype='uint32')
        # print('block {}: maxlabel {:10d}'.format(k+1, maxlabel))

    mo.close()
    im.close()

    return mo


def delete_blocks_parallel(image_in, block_idx, postfix='_fix'):

    im = LabelImage(image_in)
    im.load(load_data=False)

    vols = {postfix: 'uint32',
            '{}_reseg_mask'.format(postfix): 'bool',
            'block_idxs': 'uint16'}

    for pf, dtype in vols.items():
        delete_h5_dataset(im, pf=pf)


def delete_h5_dataset(im, pf='_fix'):
    """Copy an image to a new h5 dataset."""

    comps = im.split_path()
    ids = '{}{}'.format(comps['int'], pf)
    print(ids)
    del im.file[ids]

    im.close()


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


def plot_images(axs, info_dict={}):
    """Show images in report."""

    centreslices = info_dict['centreslices']
    meds = info_dict['medians']
    vmax = info_dict['plotinfo']['vmax']

    aspects = ['equal', 'auto', 'auto']
    aspects = ['equal', 4, 4]
    # aspects = ['equal', 'equal', 'equal']
    for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

        data = centreslices['data'][dim]

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

        axs[0][i].imshow(data, aspect=aspect, cmap='gray')

        labels = centreslices['orig'][dim]
        clabels = label2rgb(labels, image=None, bg_label=0)
        axs[1][i].imshow(clabels, aspect=aspect)

        labels = centreslices['reseg'][dim]
        clabels = label2rgb(labels, image=None, bg_label=0)
        axs[2][i].imshow(clabels, aspect=aspect)

        labels = centreslices['final'][dim]
        clabels = label2rgb(labels, image=None, bg_label=0)
        axs[3][i].imshow(clabels, aspect=aspect)

        if lines:
            for m in pp:
                if (dim == 'z') & (info_dict['axis'] == 0):
                    for j in range(0, 4):
                        axs[j][i].plot(fs, [m, m], '--', linewidth=1, color='w')
                        axs[j][i].plot([m, m], fs, '--', linewidth=1, color='w')
                elif (dim == 'z') & (info_dict['axis'] == 1):
                    for j in range(0, 4):
                        axs[j][i].plot(fs, [m, m], '--', linewidth=1, color='w')
                elif (dim == 'z') & (info_dict['axis'] == 2):
                    for j in range(0, 4):
                        axs[j][i].plot([m, m], fs, '--', linewidth=1, color='w')
                else:
                    for j in range(0, 4):
                        axs[j][i].plot([m, m], fs, '--', linewidth=1, color='w')

        for a in axs:
            a[i].axis('off')


def add_titles(axs, info_dict):
    """Add plot titles to upper row of plot grid."""

    # TODO
    return


def generate_report(image_in, info_dict={}, ioff=True):
    """Generate a QC report of the segmentation process."""

    report_type = 'reseg'

    # Turn interactive plotting on/off.
    if ioff:
        plt.ioff()
    else:
        plt.ion()

    # Get paths from image if info_dict not provided.
    if not info_dict:
        im = Image(image_in)
        info_dict['paths'] = im.split_path()  # FIXME: out_base
        info_dict['paths']['out_base'] = info_dict['paths']['base']
        im.close()
        info_dict['parameters'] = get_parameters(info_dict, report_type)
        # info_dict['centreslices'] = get_centreslices(info_dict, idss=[])
        info_dict['medians'] = {}

    # Create the axes.
    figsize = (18, 9)
    gridsize = (1, 4)
    f = plt.figure(figsize=figsize, constrained_layout=False)
    gs0 = gridspec.GridSpec(gridsize[0], gridsize[1], figure=f)
    # axs = [gen_orthoplot(f, gs0[j, i]) for j in range(0, 2) for i in range(0, 4)]
    axs = [gen_orthoplot(f, gs0[j, i])
           for j in range(0, gridsize[0])
           for i in range(0, gridsize[1])]

    # Plot the images and graphs. vmaxs = [15000] + [5000] * 7
    info_dict['plotinfo'] = {'vmax': 10000}
    plot_images(axs, info_dict)

    # Add annotations and save as pdf.
    reseg_id = 'axis{:01d}-seam{:03d}-j{:03d}'.format(info_dict['axis'], info_dict['seam'], info_dict['j'])
    header = 'mLSR-3D Quality Control'
    figtitle = '{}: {} \n {}'.format(
        header,
        report_type,
        reseg_id,
        )
    figpath = '{}_{}_{}.pdf'.format(
        info_dict['paths']['out_base'],
        report_type,
        reseg_id,
        )
    print('writing report to {}'.format(figpath))
    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    add_titles(axs, info_dict)
    f.savefig(figpath)
    plt.close(f)


if __name__ == "__main__":
    main(sys.argv[1:])
