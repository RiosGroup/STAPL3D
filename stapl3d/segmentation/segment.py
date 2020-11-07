#!/usr/bin/env python

"""Segment cells from membrane and nuclear channels.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from glob import glob

import time

from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes

from skimage.color import label2rgb
from skimage.transform import resize
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import (
    gaussian,
    median,
    threshold_sauvola,
    )
from skimage.morphology import (
    label,
    remove_small_objects,
    opening,
    binary_opening,
    binary_closing,
    binary_dilation,
    binary_erosion,
    ball,
    disk,
    )

from stapl3d import (
    get_outputdir,
    get_params,
    get_blockfiles,
    get_n_workers,
    Image,
    LabelImage,
    MaskImage,
    wmeMPI,
    get_image,
    split_filename,
    )

from stapl3d.reporting import (
    # gen_orthoplot,
    load_parameters,
    get_centreslice,
    get_centreslices,
    get_zyx_medians,
    get_cslc,
    )

logger = logging.getLogger(__name__)


def main(argv):
    """"Segment cells from membrane and nuclear channels.

    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-i', '--image_in',
        required=True,
        help='path to image file',
        )
    parser.add_argument(
        'parameter_file',
        help='path to yaml parameter file',
        )
    parser.add_argument(
        '-o', '--outputdir',
        required=False,
        help='path to output directory',
        )

    args = parser.parse_args()

    estimate(args.image_in, args.parameter_file, args.outputdir)


def estimate(
    image_in,
    parameter_file,
    outputdir='',
    n_workers=0,
    blocks=[],
    step_id='segmentation',
    ):
    """Segment cells from membrane and nuclear channels."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, fallback='blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    arglist = [(filepath, params, True) for filepath in filepaths]

    n_workers = get_n_workers(len(blocks), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(cell_segmentation, arglist)


def cell_segmentation(
    filepath,
    params,
    save_steps=True,
    ):

    logging.basicConfig(filename='{}.log'.format(filepath), level=logging.INFO)
    report = {'parameters': locals()}

    for step_key, pars in params.items():

        t = time.time()

        if step_key.startswith('prep'):
            im = prep_volume(filepath, step_key, pars, save_steps)
        elif step_key.startswith('mask'):
            im = mask_volume(filepath, step_key, pars, save_steps)
        elif step_key.startswith('combine'):
            im = combine_volumes(filepath, step_key, pars, save_steps)
        elif step_key == 'seed':
            im = seed_volume(filepath, step_key, pars, save_steps)
        elif step_key == 'segment':
            im = segment_volume(filepath, step_key, pars, save_steps)

        elapsed = time.time() - t
        print('{} took {:1f} s'.format(step_key, elapsed))


def prep_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    print(image_in)
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    # if 'shift_planes' in pars.keys():
    #     p = pars['shift_planes']
    try:
        p = pars['shift_planes']
        print(p)
    except KeyError:
        pass
    else:
        data = shift_channel(data, n_planes=p['n_planes'])
        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im)

    try:
        p = pars['opening']
        print(p)
    except KeyError:
        pass
    else:
        try:
            selem = p['selem']
        except KeyError:
            selem = None
        data = opening(data, selem=selem, out=data)
        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im)

    try:
        p = pars['filter']
        print(p)
    except KeyError:
        pass
    else:

        if p['type'] in ['median', 'gaussian']:
            if p['inplane']:
                data = smooth_channel_inplane(data, p['sigma'], p['type'])
            else:
                # TODO
                data = smooth_channel(data, p['sigma'], p['type'])
        elif p['type'] == 'dog':
            data = smooth_dog(data, im.elsize, p['sigma1'], p['sigma2'])
            data = data.astype('float16')

        if 'postfix' in p.keys(): write(data, image_in, p['postfix'], im)

    im = write(data, '{}/'.format(filepath), pars['ods_image'], im, 'Image')

    return im


def mask_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    try:
        p = pars['threshold']
        print(p)
    except KeyError:
        pass
    else:
        mask = data > pars['threshold']

    try:
        p = pars['sauvola']
        print(p)
    except KeyError:
        pass
    else:

        if 'absmin' in p.keys():
            mask = data > p['absmin']
        else:
            mask = np.zeros_like(data, dtype='bool')

        if 'k' in p.keys():
            thr = threshold_sauvola(data, window_size=p['window_size'], k=p['k'])
            mask_sauvola = data > thr
            mask &= mask_sauvola

        if 'threshold' in p.keys():
            mask_threshold = data > p['threshold']
            mask_threshold = binary_closing(mask_threshold)
            mask |= mask_threshold

    try:
        p = pars['fill']
        print(p)
    except KeyError:
        pass
    else:
        binary_fill_holes(mask, output=mask)
        if 'postfix' in p.keys(): write(mask, image_in, p['postfix'], im, 'MaskImage')

    try:
        p = pars['erode']
        print(p)
    except KeyError:
        pass
    else:
        # FIXME: this may or may not be the desired primary output
        disk_erosion = disk(p['disk'])
        mask_ero = np.zeros_like(mask, dtype='bool')
        for i, slc in enumerate(mask):  # FIXME: assuming zyx here
            mask_ero[i, :, :] = binary_erosion(slc, disk_erosion)
        if 'postfix' in p.keys(): write(mask_ero, image_in, p['postfix'], im, 'MaskImage')

    im = write(mask, '{}/'.format(filepath), pars['ods_mask'], im, 'MaskImage')

    return im


def combine_volumes(filepath, step_key, pars, save_steps=True):
    # TODO: make more general and versatile

    if step_key.endswith('masks'):

        image_in = '{}/{}'.format(filepath, pars['ids_nucl'])
        im = MaskImage(image_in)
        im.load()
        mask_nucl = im.slice_dataset().astype('bool')
        im.close()

        image_in = '{}/{}'.format(filepath, pars['ids_memb'])
        im = MaskImage(image_in)
        im.load()
        mask_memb = im.slice_dataset().astype('bool')
        im.close()

        try:
            p = pars['erode_nucl']
        except KeyError:
            pass
        else:
            disk_erosion = disk(p['disk'])
            for i, slc in enumerate(mask_nucl):  # FIXME: assuming zyx here
                mask_nucl[i, :, :] = binary_erosion(slc, disk_erosion)

        mask = np.logical_and(mask_nucl, ~mask_memb)

        try:
            p = pars['opening_footprint']
        except KeyError:
            pass
        else:
            footprint = create_footprint(p)
            mask = binary_opening(mask, footprint, out=mask)

    im = write(mask, '{}/'.format(filepath), pars['ods_mask'], im, 'MaskImage')

    return im


def seed_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    image_in = '{}/{}'.format(filepath, pars['ids_mask'])
    im = Image(image_in)
    im.load()
    mask = im.slice_dataset()
    im.close()

    try:
        p = pars['edt']
    except KeyError:
        pass
    else:
        edt = distance_transform_edt(mask, sampling=np.absolute(im.elsize))
        # mask = im.ds[:].astype('uint32')
        # edt = edt.edt(mask, anisotropy=im.elsize, black_border=True, order='F', parallel=1)
        # TODO?: leverage parallel
        try:
            threshold = p['threshold']
        except KeyError:
            pass
        else:
            edt[edt > threshold] = 0
        if 'postfix' in p.keys(): write(edt, image_in, p['postfix'], im, 'Image')

    try:
        p = pars['modulate']
    except KeyError:
        pass
    else:
        dog = smooth_dog(data, im.elsize, p['sigma1'], p['sigma2'])
        edt *= normalize_data(dog, a=p['min'], b=p['max'])[0]
        if 'postfix' in p.keys(): write(dog, image_in, p['postfix'], im, 'Image')  # edt and/or dog?

    try:
        p = pars['edt_threshold']
        print(p)
    except KeyError:
        pass
    else:
        mask = edt > pars['edt_threshold']

    try:
        p = pars['peaks']
    except KeyError:
        pass
    else:
        # find peaks in the distance transform
        mask = find_local_maxima(edt, p['window'], p['threshold'])
        try:
            footprint = p['dilate']['footprint']
        except KeyError:
            pass
        else:
            footprint = create_footprint(footprint)
            mask_dil = binary_dilation(mask, selem=footprint)
            if 'postfix' in p['dilate'].keys(): write(mask_dil, image_in, p['dilate']['postfix'], im, 'MaskImage')

    try:
        p = pars['label']
    except KeyError:
        pass
    else:
        seeds = ndi.label(mask)[0]

    try:
        p = pars['seeds']
    except KeyError:
        pass
    else:
        if 'threshold' in p.keys():
            seeds = watershed(-edt, seeds, mask=edt > p['threshold'])

    im = write(seeds, '{}/'.format(filepath), pars['ods_labels'], im, 'LabelImage')

    return im


def segment_volume(filepath, step_key, pars, save_steps=True):

    image_in = '{}/{}'.format(filepath, pars['ids_image'])
    im = Image(image_in)
    im.load()
    data = im.slice_dataset()
    im.close()

    image_in = '{}/{}'.format(filepath, pars['ids_labels'])
    im = Image(image_in)
    im.load()
    seeds = im.slice_dataset()
    im.close()

    try:
        p = pars['watershed']
    except KeyError:
        pass
    else:
        if 'ids_mask' in p.keys():
            image_in = '{}/{}'.format(filepath, p['ids_mask'])
            im = Image(image_in)
            im.load()
            mask = im.slice_dataset()
            im.close()
        else:
            mask = None

        if 'invert_data' in p.keys():
            data = -data

        if 'voxel_spacing' in p.keys():
            spacing = p['voxel_spacing']
        else:
            spacing = im.elsize

        if 'compactness' in p.keys():
            compactness = p['compactness']
        else:
            compactness = 0.0

        try:
            ws = watershed(data, seeds, mask=mask, compactness=compactness, spacing=spacing)
        except TypeError:
            print('WARNING: possibly not using correct spacing for compact watershed')
            ws = watershed(data, seeds, mask=mask, compactness=compactness)

        if 'ids_mask_post' in p.keys():
            image_in = '{}/{}'.format(filepath, p['ids_mask_post'])
            im = Image(image_in)
            im.load()
            mask = im.slice_dataset().astype('bool')
            im.close()
            ws[~mask] = 0

        if 'postfix' in p.keys(): write(ws, image_in, p['postfix'], im, 'MaskImage')

    try:
        p = pars['filter']
    except KeyError:
        pass
    else:
        image_in = '{}/{}'.format(filepath, p['ids_mask'])
        im = MaskImage(image_in)
        im.load()
        mask = im.slice_dataset().astype('bool')
        im.close()
        maxlabel = max(np.unique(ws))
        ws = delete_labels_in_mask(ws, ~mask, maxlabel)

    im = write(ws, '{}/'.format(filepath), pars['ods_labels'], im, 'LabelImage')

    return im


def normalize_data(data, a=1.00, b=1.01):
    """Normalize data."""

    data = data.astype('float64')
    datamin = np.amin(data)
    datamax = np.amax(data)
    data -= datamin
    data *= (b-a)/(datamax-datamin)
    data += a

    return data, [datamin, datamax]


def gen_outpath(im, pf):
    """Fix us a postfixed output path."""

    comps = im.split_path()
    if im.format == '.nii':
        outpath = "{}{}{}".format(comps['base'], pf, comps['ext'])
    elif im.format == '.h5':
        outpath = "{}{}{}".format(comps['file'], comps['int'], pf)

    return outpath


def write(out, outstem, postfix, ref_im, imtype='Image'):
    """Write an image to disk."""

    print(outstem)
    outstem = outstem or gen_outpath(ref_im, '')
    print(outstem)
    outpath = '{}{}'.format(outstem, postfix)
    print(outpath)

    props = ref_im.get_props()
    props['dtype'] = out.dtype

    if imtype == 'Label':
        mo = LabelImage(outpath, **props)
    elif imtype == 'Mask':
        mo = MaskImage(outpath, **props)
    else:
        mo = Image(outpath, **props)

    mo.create()
    mo.write(out)

    if imtype == 'Label':
        mo.set_maxlabel()
        mo.ds.attrs.create('maxlabel', mo.maxlabel, dtype='uint32')

    return mo


def find_local_maxima(data, size=[13, 13, 3], threshold=0.05, noise_sd=0.0):
    """Find peaks in image."""

    if threshold == -float('Inf'):
        threshold = img.min()

    def add_noise(data, sd=0.001):

        mask = data == 0
        data += np.random.normal(0, sd, (data.shape))
        data[mask] = 0

        return data

    """
    NOTE: handle double identified peaks within the footprint region
    by adding a bit of noise
    (with same height within maximum_filtered image)
    this happens a lot for edt, because of discrete distances
    This has now been handled through modulation of distance transform.
    """
    if noise_sd:
        data = add_noise(data, noise_sd)

    footprint = create_footprint(size)
    image_max = ndi.maximum_filter(data, footprint=footprint, mode='constant')

    mask = data == image_max
    mask &= data > threshold

    coordinates = np.column_stack(np.nonzero(mask))[::-1]

    peaks = np.zeros_like(data, dtype=np.bool)
    peaks[tuple(coordinates.T)] = True

    return peaks


def create_footprint(size=[5, 21, 21]):
    """Create a 3D ellipsoid-like structure element for anisotropic data.

    FIXME: why don't I just get an isotropic ball and delete some slices?
    """

    footprint = np.zeros(size)
    c_idxs = [int(size[0] / 2), int(size[1] / 2), int(size[2] / 2)]
    disk_ctr = disk(c_idxs[1])
    footprint[int(size[0]/2), :, :] = disk_ctr
    for i in range(c_idxs[0]):
        j = 2 + i + 1
        r = int(size[1] / j)
        d = disk(r)
        slc = slice(c_idxs[1]-r, c_idxs[1]+r+1, 1)
        footprint[c_idxs[0]-i-1, slc, slc] = d
        footprint[c_idxs[0]+i+1, slc, slc] = d

    return footprint


def shift_channel(data, n_planes=0, zdim_idx=0):

    if n_planes:
        if zdim_idx == 0:
            data[n_planes:, :, :] = data[:-n_planes, :, :]
            data[:n_planes, :, :] = 0
        elif zdim_idx == 2:
            data[:, :, n_planes:] = data[:, :, :-n_planes]
            data[:, :, :n_planes] = 0

    return data


def smooth_channel(data, sigma=3, filter='median'):

    if filter == 'median':
        k = ball(sigma)  # TODO footprint
        data_smooth = median(data, k)
    elif filter == 'gaussian':
        data_smooth = gaussian(data, sigma=sigma, preserve_range=True)

    return data_smooth


def smooth_channel_inplane(data, sigma=3, filter='median'):

    k = disk(sigma)
    data_smooth = np.zeros_like(data)
    for i, slc in enumerate(data):
        if filter == 'median':
            data_smooth[i, :, :] = median(slc, k)
        elif filter == 'gaussian':
            data_smooth[i, :, :] = gaussian(slc, sigma=sigma, preserve_range=True)

    return data_smooth


def smooth_dog(data, elsize, sigma1, sigma2):

    elsize = np.absolute(elsize)
    s1 = [sigma1 / es for es in elsize]
    s2 = [sigma2 / es for es in elsize]
    dog = gaussian(data, s1) - gaussian(data, s2)

    return dog


# UNUSED?
def upsample_to_block(mask_ds, block_us, dsfacs=[1, 16, 16, 1], order=0):
    """Upsample a low-res mask to a full-res block."""

    comps = block_us.split_path()
    block_info = split_filename(comps['file'])[0]
    slices_us = [slice(block_info['z'], block_info['Z'], None),
                 slice(block_info['y'], block_info['Y'], None),
                 slice(block_info['x'], block_info['X'], None)]

    mask_ds.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                      for slc, ds in zip(slices_us, dsfacs)]
    mask_block = mask_ds.slice_dataset()  #.astype('float32')
    mask_us = resize(mask_block, block_us.dims, preserve_range=True, order=order)

    return mask_us.astype(mask_ds.dtype)


def delete_labels_in_mask(labels, mask, maxlabel=0):
    """Delete the labels found within mask."""

    labels_del = np.copy(labels)

    # TODO: save deleted labelset
    maxlabel = maxlabel or np.amax(labels[:])
    if maxlabel:
        labelset = set(np.unique(labels[mask]))
        fwmap = [True if l in labelset else False for l in range(0, maxlabel + 1)]
        labels_del[np.array(fwmap)[labels]] = 0

    return labels_del


# UNUSED?
def find_border_segments(im):

    segments = set([])
    for idx_z in [0, -1]:
        for idx_y in [0, -1]:
            for idx_x in [0, -1]:
                segments = segments | set(im.ds[idx_z, idx_y, idx_x].ravel())

    return segments


def nucl_mask(maskpath_sauvola, maskpath_absmin):

    mask_sauvola_im = MaskImage(maskpath_sauvola, permission='r')
    mask_sauvola_im.load(load_data=False)
    mask_sauvola = mask_sauvola_im.slice_dataset().astype('bool')

    mask_absmin_im = MaskImage(maskpath_absmin, permission='r')
    mask_absmin_im.load(load_data=False)
    mask_absmin = mask_absmin_im.slice_dataset().astype('bool')

    mask = mask_absmin & mask_sauvola

    return mask


def memb_mask(labels_ds, memb_meth='ip'):
    mask = find_boundaries(labels_ds)
    if memb_meth == 'iso':  # 0) isotropic dilation
        mask = binary_dilation(mask)
    elif memb_meth == 'ip':  # 1) in-plane dilation only
        for i, slc in enumerate(mask):
            mask[i, :, :] = binary_dilation(slc)
    return mask


def subsegment(
    image_in,
    parameter_file,
    outputdir='',
    n_workers=0,
    blocks=[],
    ids='segm/labels_memb_del_relabeled_fix',
    ods_full='',
    ods_memb='',
    ods_nucl='',
    ods_csol='',
    ):
    """Perform N4 bias field correction."""

    step_id = 'subsegment'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, 'blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    arglist = [
        (
            filepath,
            params['ids'],
            params['ods_full'],
            params['ods_memb'],
            params['ods_nucl'],
            params['ods_csol'],
            outputdir,
        )
        for block_idx, filepath in zip(blocks, filepaths)]

    n_workers = get_n_workers(len(blocks), subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(split_segments, arglist)


def split_segments(
    inputfile,
    ids='segm/labels_memb_del_relabeled_fix',
    ods_full='',
    ods_memb='',
    ods_nucl='',
    ods_csol='',
    outputdir='',
    ):

    labels = LabelImage('{}/{}'.format(inputfile, ids))
    labels.load(load_data=False)
    try:
        del labels.file[ods_full]
    except KeyError:
        pass
    labels.file[ods_full] = labels.file[ids]
    labels_ds = labels.slice_dataset()

    if ods_nucl:
        # nuclei
        nuclstem = '{}/{}'.format(inputfile, 'nucl/dapi')  # TODO: flexible naming
        maskpath_sauvola = '{}_mask_sauvola'.format(nuclstem)
        maskpath_absmin = '{}_mask_absmin'.format(nuclstem)
        mask_nucl = nucl_mask(maskpath_sauvola, maskpath_absmin)
        write(mask_nucl, nuclstem, '_mask_nuclei', labels, imtype='Mask')

        outstem = '{}/{}'.format(inputfile, ods_nucl)
        labs = np.copy(labels_ds)
        labs[~mask_nucl] = 0
        write(labs, outstem, '', labels, imtype='Label')

    if ods_memb:
        # membranes  # TODO: may combine with planarity_mask to make it more data-informed
        membstem = '{}/{}'.format(inputfile, 'memb/boundary')
        mask_memb = memb_mask(labels_ds)
        write(mask_memb, membstem, '_mask', labels, imtype='Mask')

        outstem = '{}/{}'.format(inputfile, ods_memb)
        labs = np.copy(labels_ds)
        labs[~mask_memb] = 0
        write(labs, outstem, '', labels, imtype='Label')

    # fstem = '{}/'.format(inputfile)  # TODO: flexible naming
    # # nuclei  # TODO: this has probably already been written as .h5/nucl/dapi_mask
    # nuclstem = '{}/{}'.format(inputfile, 'nucl/dapi')  # TODO: flexible naming
    # maskpath_nucl = '{}_mask'.format(nuclstem)
    # mask_nucl_im = MaskImage(maskpath_nucl, permission='r')
    # mask_nucl_im.load(load_data=False)
    # mask_nucl = mask_nucl_im.slice_dataset().astype('bool')
    # write(mask_nucl, fstem, 'mask_nuclei', labels, imtype='Mask')
    #
    # membstem = '{}/{}'.format(inputfile, 'segm/labels_csol')  # TODO: flexible naming
    # maskpath_memb = '{}_mask'.format(membstem)
    # mask_memb_im = MaskImage(maskpath_memb, permission='r')
    # mask_memb_im.load(load_data=False)
    # mask_memb = mask_memb_im.slice_dataset().astype('bool')
    # write(mask_memb, fstem, 'mask_membrane', labels, imtype='Mask')
    # # membranes  # TODO: may combine with planarity_mask to make it more data-informed
    #
    # mask_csol = ~mask_memb & ~mask_nucl
    # write(mask_csol, fstem, 'mask_cytosol', labels, imtype='Mask')
    #
    # for mask, ods in zip([mask_memb, mask_nucl, mask_csol], [ods_memb, ods_nucl, ods_csol]):
    #     outstem = '{}/{}'.format(inputfile, ods)
    #     labs = np.copy(labels_ds)
    #     labs[~mask] = 0
    #     write(labs, outstem, '', labels, imtype='Label')


def plot_images(axs, info_dict={}):
    """Show images in report."""

    def get_data(prefered, fallback, dimfac):

        try:
            slc = centreslices[prefered][dim] * dimfac
            return slc
        except (TypeError, KeyError):
            print('{} not found: falling back to {}'.format(prefered, fallback))
            try:
                slc = centreslices[fallback][dim] * dimfac
                return slc
            except (TypeError, KeyError):
                print('{} not found: falling back to empty'.format(fallback))
                # TODO: empty image of correct dim
                return None

    centreslices = info_dict['centreslices']
    meds = info_dict['medians']
    vmax = info_dict['plotinfo']['vmax']

    aspects = ['equal', 'equal', 'equal']
    aspects = ['equal', 'auto', 'auto']
    for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

        data_nucl = get_data('nucl/dapi_preprocess', 'chan/ch00', dimfac=3)
        data_memb = get_data('memb/mean_smooth', 'memb/mean', dimfac=5)
        if data_memb is None:
            data_memb = np.zeros(data_nucl.shape)

        if dim == 'x':
            data_nucl = data_nucl.transpose()
            data_memb = data_memb.transpose()

        axs[0][i].imshow(data_nucl, aspect=aspect, cmap='gray')
        axs[4][i].imshow(data_memb, aspect=aspect, cmap='gray')

        try:

            data_edt = centreslices['segm/seeds_edt'][dim]

            if dim == 'x':
                data_edt = data_edt.transpose()

            # labels = centreslices['nucl/dapi_mask'][dim]
            labels = centreslices['mean_mask'][dim].astype('uint8')
            labels[centreslices['nucl/dapi_mask'][dim]] = 2
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_nucl, alpha=0.5, bg_label=0, colors=[[1, 0, 0], [0, 1, 0]])
            axs[1][i].imshow(clabels, aspect=aspect)

            # labels = centreslices['memb/mask'][dim]
            labels = centreslices['mean_mask'][dim].astype('uint8')
            labels[centreslices['memb/planarity_mask'][dim]] = 2
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_memb, alpha=0.5, bg_label=0, colors=[[1, 0, 0], [0, 1, 0]])
            axs[5][i].imshow(clabels, aspect=aspect)

            labels = centreslices['segm/seeds_mask'][dim].astype('uint8')
            labels[centreslices['segm/seeds_peaks_dil'][dim]] = 2
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_nucl, alpha=0.5, bg_label=0, colors=[[1, 0, 0], [0, 1, 0]])
            axs[2][i].imshow(clabels, aspect=aspect)

            labels = centreslices['segm/seeds_peaks_dil'][dim]
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_edt, alpha=0.5, bg_label=0, colors=None)
            axs[6][i].imshow(clabels, aspect=aspect)

            labels = centreslices['segm/labels_edt'][dim]
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_nucl, alpha=0.7, bg_label=0, colors=None)
            axs[3][i].imshow(clabels, aspect=aspect)

            labels = centreslices['segm/labels_memb_del'][dim]
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_memb, alpha=0.3, bg_label=0, colors=None)
            axs[7][i].imshow(clabels, aspect=aspect)

        except (TypeError, KeyError):
            print('not all steps were found: generating simplified report')

            labels = centreslices['segm/labels_memb'][dim]
            if dim == 'x':
                labels = labels.transpose()
            clabels = label2rgb(labels, image=data_memb, alpha=0.3, bg_label=0)
            axs[7][i].imshow(clabels, aspect=aspect)

        for a in axs:
            a[i].axis('off')


def add_titles(axs, info_dict):
    """Add plot titles to upper row of plot grid."""

    # TODO
    return


def gen_orthoplot(f, gs):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_xy = 5
    size_z = 5
    size_t = size_xy + size_z

    gs_sub = gs.subgridspec(size_t, size_t)

    # central: yx-image
    axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy]))
    # middle-bottom: zx-image
    axs.append(f.add_subplot(gs_sub[size_xy:, :size_xy], sharex=axs[0]))
    # right-middle: zy-image
    axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:], sharey=axs[0]))

    return axs


def generate_report(image_in, info_dict={}, ioff=True):
    """Generate a QC report of the segmentation process."""

    report_type = 'seg'

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
        info_dict['parameters'] = load_parameters(info_dict['paths']['out_base'])
        info_dict['centreslices'] = get_centreslices(info_dict, idss=[
            'mean_mask',
            'memb/planarity_mask',
            'memb/mean',
            'memb/mean_smooth',
            'chan/ch00',
            'nucl/dapi_mask',
            'nucl/dapi_preprocess',
            'segm/labels_edt',
            'segm/labels_memb',
            'segm/labels_memb_del',
            'segm/seeds_edt',
            'segm/seeds_mask',
            'segm/seeds_peaks_dil',
            ])
        info_dict['medians'] = {}


    # Create the axes.
    figsize = (18, 9)
    gridsize = (2, 4)
    f = plt.figure(figsize=figsize, constrained_layout=False)
    gs0 = gridspec.GridSpec(gridsize[0], gridsize[1], figure=f)
    axs = [gen_orthoplot(f, gs0[j, i]) for j in range(0, 2) for i in range(0, 4)]

    # Plot the images and graphs. vmaxs = [15000] + [5000] * 7
    info_dict['plotinfo'] = {'vmax': 10000}
    plot_images(axs, info_dict)

    # Add annotations and save as pdf.
    header = 'mLSR-3D Quality Control'
    figtitle = '{}: {} \n {}'.format(
        header,
        report_type,
        info_dict['paths']['fname']
        )
    figpath = '{}_{}.pdf'.format(
        info_dict['paths']['out_base'],
        report_type
        )
    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    add_titles(axs, info_dict)
    f.savefig(figpath)

    info_dict.clear()


if __name__ == "__main__":
    main(sys.argv[1:])
