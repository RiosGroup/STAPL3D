#!/usr/bin/env python

"""Segment cells from membrane and nuclear channels.

"""

import sys
import argparse

import os
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.color import label2rgb

from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
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
    get_image,
    Image, LabelImage, MaskImage,
    split_filename,
    )
from stapl3d.channels import get_n_workers
from stapl3d.reporting import (
    # gen_orthoplot,
    load_parameters,
    get_paths,
    get_centreslice,
    get_centreslices,
    get_zyx_medians,
    get_cslc,
    )


def main(argv):
    """"Segment cells from membrane and nuclear channels.

    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        'memb_mask_file',
        help='path to membrane mask',
        )
    parser.add_argument(
        'memb_chan_file',
        help='path to membrane channel',
        )
    parser.add_argument(
        'dapi_chan_file',
        help='path to dapi channel',
        )
    parser.add_argument(
        'mean_chan_file',
        help='path to mean channel',
        )
    parser.add_argument(
        'parameter_file',
        help='path to yaml parameter file',
        )
    parser.add_argument(
        '-o', '--outputstem',
        help='template for output',
        )
    parser.add_argument(
        '-S', '--save_steps',
        action='store_true',
        help='save intermediate results'
        )

    args = parser.parse_args()

    extract_segments(
        args.memb_mask_file,
        args.memb_chan_file,
        args.dapi_chan_file,
        args.mean_chan_file,
        args.parameter_file,
        args.outputstem,
        args.save_steps,
        )


# def segmentation(
#     parameter_file='',
#     outputstem='',
#     save_steps=False,
#     ):
#     """Average membrane and nuclear channels and write as blocks."""
#
#     params = locals()
#
#     file_params = {}
#     if parameter_file:
#         with open(parameter_file, 'r') as ymlfile:
#             cfg = yaml.safe_load(ymlfile)
#             file_params = cfg['segmentation']
#
#     params.update(file_params)
#
#     if not params['blocks']:
#         n_blocks = get_n_blocks(filepath)  # TODO: with glob?
#         params['blocks'] = list(range(n_blocks))
#
#     n_workers = get_n_workers(len(params['channels']), params)
#
#     arglist = [
#         (
#             filepath,
#             params['blocksize'],
#             params['blockmargin'],
#             [b_idx, b_idx+1],
#             params['bias_image'],
#             params['bias_dsfacs'],
#             params['memb_idxs'],
#             params['memb_weights'],
#             params['nucl_idxs'],
#             params['nucl_weights'],
#             params['mean_idxs'],
#             params['mean_weights'],
#             params['output_channels'],
#             params['datatype'],
#             params['chunksize'],
#             params['outputprefix'],
#             False,
#         )
#         for b_idx in params['blocks']]
#
#     with multiprocessing.Pool(processes=n_workers) as pool:
#         pool.starmap(extract_segments, arglist)


def extract_segments(
    plan_path,
    memb_path,
    dapi_path,
    mean_path,
    parameter_file,
    outputstem='',
    save_steps=False,
    ):

    import yaml
    import pprint
    from types import SimpleNamespace

    with open(parameter_file, "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    n = SimpleNamespace(**cfg["segmentation"])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(n)

    cell_segmentation(
        plan_path,
        memb_path,
        dapi_path,
        mean_path,
        n.dapi_shift_planes,
        n.nucl_opening_footprint,
        n.dapi_filter,
        n.dapi_sigma,
        n.dapi_dog_sigma1,
        n.dapi_dog_sigma2,
        n.dapi_thr,
        n.sauvola_window_size,
        n.sauvola_k,
        n.dapi_absmin,
        n.dapi_erodisk,
        n.dist_max,
        n.peaks_size,
        n.peaks_thr,
        n.peaks_dil_footprint,
        n.compactness,
        n.memb_filter,
        n.memb_sigma,
        n.planarity_thr,
        n.dset_mask_filter,
        n.dset_mask_sigma,
        n.dset_mask_thr,
        n.steps,
        outputstem,
        save_steps,
    )


def cell_segmentation(
    plan_path,
    memb_path,
    dapi_path,
    mean_path,
    dapi_shift_planes=0,
    nucl_opening_footprint=[3, 7, 7],
    dapi_filter='median',
    dapi_sigma=1,
    dapi_dog_sigma1=2,
    dapi_dog_sigma2=4,
    dapi_thr=0,
    sauvola_window_size=[19, 75, 75],
    sauvola_k=0.2,
    dapi_absmin=500,
    dapi_erodisk=0,
    dist_max=5,
    peaks_size=[11, 19, 19],
    peaks_thr=1.0,
    peaks_dil_footprint=[3, 7, 7],
    compactness=0.80,
    memb_filter='median',
    memb_sigma=3,
    planarity_thr=0.0005,
    dset_mask_filter='gaussian',
    dset_mask_sigma=50,
    dset_mask_thr=1000,
    steps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    outputstem='',
    save_steps=False,
    ):

    step = 'segment'
    paths = get_paths(plan_path, -1, 0, outputstem, step, save_steps)
    report = {
        'parameters': locals(),
        'paths': paths,
        'medians': {},
        'centreslices': {}
        }

    # load images
    im_dapi = Image(dapi_path)
    im_dapi.load()
    nucl_props = im_dapi.get_props()

    im_memb = MaskImage(memb_path)
    im_memb.load()
    memb_props = im_memb.get_props()

    im_plan = MaskImage(plan_path)
    im_plan.load()

    # im_dset_mask = Image(dset_mask_path, permission='r')
    # im_dset_mask.load(load_data=False)
    im_mean = Image(mean_path)
    im_mean.load()

    # preprocess dapi channel
    # .h5/nucl/dapi<_shifted><_opened><_preprocess>
    stage = 'nucleus channel'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/nucl/dapi')
    if 0 not in steps:
        op = 'reading'
        im_dapi_pp = get_image('{}{}'.format(outstem, '_preprocess'))
    else:
        op = 'processing'
        im_dapi_pp = preprocess_nucl(
            im_dapi,
            dapi_shift_planes,
            dapi_filter,
            dapi_sigma,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # create a nuclear mask from the dapi channel
    # .h5/nucl/dapi<_mask_thr><_sauvola><_mask><_mask_ero>
    stage = 'nucleus mask'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/nucl/dapi')
    if 1 not in steps:
        op = 'reading'
        im_dapi_mask = get_image('{}{}'.format(outstem, '_mask_ero'))
    else:
        op = 'processing'
        im_dapi_mask = create_nuclear_mask(
            im_dapi_pp,
            dapi_thr,
            sauvola_window_size,
            sauvola_k,
            dapi_absmin,
            dapi_erodisk,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # create a membrane mask from the membrane mean
    # .h5/memb/planarity<_mask>
    stage = 'membrane mask'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/memb/planarity')
    if 2 not in steps:
        op = 'reading'
        im_memb_mask = get_image('{}{}'.format(outstem, '_mask'))
    else:
        op = 'processing'
        im_memb_mask = create_membrane_mask(
            im_plan,
            planarity_thr,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # combine nuclear and membrane mask
    # .h5/segm/seeds<_mask>
    stage = 'mask combination'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/segm/seeds')
    if 3 not in steps:
        op = 'reading'
        im_nucl_mask = get_image('{}{}'.format(outstem, '_mask'))
    else:
        op = 'processing'
        im_nucl_mask = combine_nucl_and_memb_masks(
            im_memb_mask,
            im_dapi_mask,
            nucl_opening_footprint,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # find seeds for watershed
    stage = 'nucleus detection'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/segm/seeds')
    if 4 not in steps:
        op = 'reading'
        im_dt = get_image('{}{}'.format(outstem, '_edt'))
        im_peaks = get_image('{}{}'.format(outstem, '_peaks'))
    # .h5/segm/seeds<_edt><_mask_distmax><_peaks><_peaks_dil>
    else:
        op = 'processing'
        im_dt, im_peaks = define_seeds(
            im_nucl_mask,
            im_memb_mask,
            im_dapi_pp,
            dapi_dog_sigma1,
            dapi_dog_sigma2,
            dist_max,
            peaks_size, peaks_thr,
            peaks_dil_footprint,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # preprocess membrane mean channel
    # .h5/memb/preprocess<_smooth>
    stage = 'membrane channel'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/memb/mean')
    if 5 not in steps:
        op = 'reading'
        im_memb_pp = get_image('{}{}'.format(outstem, '_smooth'))
    else:
        op = 'processing'
        im_memb_pp = preprocess_memb(
            im_memb,
            memb_filter,
            memb_sigma,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # perform watershed from the peaks to fill the nuclei
    # .h5/segm/labels<_edt><_memb>
    stage = 'watershed'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/segm/labels')
    if 6 not in steps:
        op = 'reading'
        im_ws = get_image('{}{}'.format(outstem, '_memb'), imtype='Label')
    else:
        op = 'processing'
        im_ws = perform_watershed(
            im_peaks,
            im_memb_pp,
            im_dt,
            peaks_thr,
            memb_sigma,
            memb_filter,
            compactness,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # generate a dataset mask from the mean of all channels
    # .h5/mean<_smooth><_mask>
    stage = 'dataset mask'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/mean')
    if 7 not in steps:
        op = 'reading'
        im_dset_mask = get_image('{}{}'.format(outstem, '_mask'), imtype='Mask')
    else:
        op = 'processing'
        im_dset_mask = create_dataset_mask(
            im_mean,
            filter=dset_mask_filter,
            sigma=dset_mask_sigma,
            threshold=dset_mask_thr,
            outstem=outstem,
            save_steps=save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # filter the segments with the dataset mask
    # .h5/segm/labels<_memb_del>
    # .h5/mask
    stage = 'segment filter'
    t = time.time()
    outstem = '{}.h5{}'.format(outputstem, '/segm/labels')
    if 8 not in steps:
        im_ws_pp = get_image('{}{}'.format(outstem, '_memb_del'), imtype='Label')
    else:
        op = 'processing'
        im_ws_pp = segmentation_postprocessing(
            im_dset_mask,
            im_ws,
            outstem,
            save_steps,
            )
    elapsed = time.time() - t
    print('{} ({}) took {:1f} s'.format(stage, op, elapsed))

    # write report
    generate_report('{}.h5/{}'.format(outputstem, 'mean_mask'))

    return im_ws_pp



def preprocess_nucl(
    im_nucl,
    shift_planes=0,
    filter='median',
    sigma=1,
    outstem='',
    save_steps=False,
    ):
    """Preprocess nuclear channel.

    - shift in z by an integer number of planes.
    - greyscale opening.
    - in-plane smoothing.
    - difference of gaussians.
    """

    # preprocess dapi: shift in z, opening and smoothing
    # TODO: move this step outside of this segmentation pipeline
    # it's too data-acquisition specific NOTE: (into sumsplit?)
    nucl_pp = shift_channel(im_nucl.ds[:], n_planes=shift_planes)
    if save_steps: write(nucl_pp, outstem, '_shifted', im_nucl)

    selem = None  # TODO
    nucl_pp = opening(nucl_pp, selem=selem, out=nucl_pp)
    if save_steps: write(nucl_pp, outstem, '_opened', im_nucl)

    nucl_pp = smooth_channel_inplane(nucl_pp, sigma, filter)
    im_nucl_pp = write(nucl_pp, outstem, '_preprocess', im_nucl)

    return im_nucl_pp



def dog_nucl(
    im_nucl,
    dog_sigma1=2,
    dog_sigma2=4,
    outstem='',
    save_steps=False,
    ):
    """Preprocess nuclear channel.

    - difference of gaussians.
    """

    elsize = np.absolute(im_nucl.elsize)
    sigma1 = [dog_sigma1 / es for es in elsize]
    sigma2 = [dog_sigma2 / es for es in elsize]
    dog = gaussian(im_nucl.ds[:], sigma1) - gaussian(im_nucl.ds[:], sigma2)
    im_dog = write(dog.astype('float16'), outstem, '_dog', im_nucl)

    return im_dog



def create_nuclear_mask(
    im_nucl,
    threshold=5000.0,
    window_size=[19, 75, 75],
    k=0.2,
    absolute_min_intensity=2000.0,
    disksize_erosion=3,
    outstem='',
    save_steps=False,
    ):
    """Generate a nuclear mask.

    - absolute minumum intensity
    - simple thresholding
    - sauvola thresholding
    - mask erosion
    """

    mask = im_nucl.ds[:] > absolute_min_intensity
    if save_steps: write(mask, outstem, '_mask_absmin', im_nucl)

    if k:
        thr = threshold_sauvola(im_nucl.ds[:], window_size=window_size, k=k)
        mask_sauvola = im_nucl.ds[:] > thr
        if save_steps: write(mask_sauvola, outstem, '_mask_sauvola', im_nucl)
        mask &= mask_sauvola
        if save_steps: write(mask, outstem, '_mask_nuclei', im_nucl)

    if threshold:
        mask_threshold = im_nucl.ds[:] > threshold
        mask_threshold = binary_closing(mask_threshold)
        if save_steps: write(mask_threshold, outstem, '_mask_thr', im_nucl)
        mask |= mask_threshold

    im_mask = write(mask, outstem, '_mask', im_nucl)

    if disksize_erosion:
        disk_erosion = disk(disksize_erosion)
        mask_ero = np.zeros_like(mask, dtype='bool')
        for i, slc in enumerate(mask):  # FIXME: assuming zyx here
            mask_ero[i, :, :] = binary_erosion(slc, disk_erosion)
        im_mask = write(mask_ero, outstem, '_mask_ero', im_nucl)

    return im_mask



def create_membrane_mask(
    im_planarity,
    threshold=0.001,
    outstem='',
    save_steps=False,
    ):
    """Generate a membrane mask.

    - simple thresholding of (preprocessed) membranes
    """

    mask = im_planarity.ds[:] > threshold
    im_mask = write(mask, outstem, '_mask', im_planarity)

    return im_mask



def create_dataset_mask(
    im_mean,
    filter='gaussian',
    sigma=50.0,
    threshold=1000.0,
    outstem='',
    save_steps=False,
    ):
    """Generate a dataset mask.

    - simple thresholding after smoothing with a large kernel
    """

    smooth = smooth_channel_inplane(im_mean.ds[:], sigma, filter)
    if save_steps: write(smooth, outstem, '_smooth', im_mean)

    mask = smooth > threshold
    im_mask = write(mask, outstem, '_mask', im_mean)

    return im_mask



def combine_nucl_and_memb_masks(
    im_memb_mask,
    im_nucl_mask,
    opening_footprint=[],
    outstem='',
    save_steps=False,
    ):
    """Combine the nuclear and membrane mask to separate the nuclei.

    - bitwise AND
    - opening of the result
    """


    comb_mask = np.logical_and(im_nucl_mask.ds[:].astype('bool'),
                               ~im_memb_mask.ds[:].astype('bool'))

    if opening_footprint:
        footprint = create_footprint(opening_footprint)
        combined_mask = binary_opening(comb_mask, footprint, out=comb_mask)

    im_comb_mask = write(comb_mask, outstem, '_mask', im_memb_mask)

    return im_comb_mask



def define_seeds(
    im_nucl_mask,
    im_memb_mask,
    im_nucl=None,
    dog_sigma1=2.0,
    dog_sigma2=4.0,
    distance_threshold=0.0,
    peaks_window=[7, 21, 21],
    peaks_threshold=1.0,
    peaks_dilate_footprint=[3, 7, 7],
    outstem='',
    save_steps=False,
    ):
    """Generate watershed seeds.

    - distance transform on the nuclei
    - maximal distance threshold
    - modulation towards nuclei centre (because discretized distance transform)
    - peak detection
    - dilation (for visualization)
    """

    # calculate distance transform
    elsize = np.absolute(im_nucl_mask.elsize)
    dt = distance_transform_edt(im_nucl_mask.ds[:], sampling=elsize)
    im_dt = write(dt, outstem, '_edt', im_memb_mask)

    # define a mask to constrain the watershed (unused for now)
    if distance_threshold:
        mask_maxdist = dt < distance_threshold
        if save_steps: write(mask_maxdist, outstem, '_mask_distmax', im_memb_mask)

    # modulate dt with normalized DoG dapi
    if im_nucl is not None:
        im_dog = dog_nucl(im_nucl, dog_sigma1, dog_sigma2, outstem, save_steps)
        dt *= normalize_data(im_dog.ds[:], a=1.00, b=1.01)[0]

    # find peaks in the distance transform
    peaks = find_local_maxima(dt, peaks_window, peaks_threshold)
    im_peaks = write(peaks, outstem, '_peaks', im_memb_mask)

    if save_steps:
        footprint = create_footprint(peaks_dilate_footprint)
        peaks_dil = binary_dilation(peaks, selem=footprint)
        write(peaks_dil, outstem, '_peaks_dil', im_memb_mask)

    return im_dt, im_peaks



def preprocess_memb(
    im_memb,
    filter='median',
    sigma=1.0,
    outstem='',
    save_steps=False,
    ):
    """Preprocess the membrame image before watershed.

    - inplane smoothing
    """

    memb = smooth_channel_inplane(im_memb.ds[:], sigma, filter)
    im_memb_pp = write(memb, outstem, '_smooth', im_memb)

    return im_memb_pp



def perform_watershed(
    im_peaks,
    im_memb,
    im_dt=None,
    peaks_thr=1.0,
    memb_sigma=1.0,
    memb_filter='median',
    compactness=0.0,
    outstem='',
    save_steps=False,
    ):
    """Create segments by watershed in a two-step procedure.

    - masked watershed in distance transform to fatten the seeds
        and avoid local minima in the membrane image
    - unmasked watershed to cell boundaries
    """

    seeds = ndi.label(im_peaks.ds[:])[0]

    if im_dt is not None:
        seeds = watershed(-im_dt.ds[:], seeds, mask=im_dt.ds[:]>peaks_thr)
        if save_steps: write(seeds, outstem, '_edt', im_memb, imtype='Label')

    # TODO: try masked watershed, e.g.:
    # 1) simple dataset mask from channel-mean,
    # 2) dilated dapi mask for contraining cells
    ws = watershed(im_memb.ds[:], seeds, compactness=compactness)
    im_ws = write(ws, outstem, '_memb', im_memb, imtype='Label')

    return im_ws



def segmentation_postprocessing(
    im_dset_mask,
    im_ws,
    outstem='',
    save_steps=False,
    ):
    """Postprocess the segments.

    - delete labels outside of the dataset mask
    """

    ws_del = delete_labels_in_mask(im_ws.ds[:], ~im_dset_mask.ds[:], im_ws.maxlabel)
    im_ws_postproc = write(ws_del, outstem, '_memb_del', im_ws, imtype='Label')

    # NOTE: split_nucl_and_memb is done in export_regionprops for now
    # im_memb, im_nucl = split_nucl_and_memb(im_ws_postproc, outstem, save_steps)

    return im_ws_postproc


# TODO: unused?
def split_nucl_and_memb(
    im,
    outstem='',
    save_steps=False,
    ):

    data = im.slice_dataset()

    memb_mask = binary_dilation(find_boundaries(data))
    labels = np.copy(data)
    labels[~memb_mask] = 0
    im_memb = write(labels, outstem, '_memb', im, imtype='Label')

    labels = np.copy(data)
    labels[memb_mask] = 0
    im_nucl = write(labels, outstem, '_nucl', im, imtype='Label')

    return im_memb, im_nucl


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

    outstem = outstem or gen_outpath(ref_im, '')
    outpath = '{}{}'.format(outstem, postfix)

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


# TODO: unused?
def write_output(outpath, out, props):


    props['dtype'] = out.dtype
    mo = Image(outpath, **props)
    mo.create()
    mo.write(out)

    return mo


# TODO: unused?
def calculate_edt(im, outpath=''):
    """Calculate distance from mask."""

    mask = im.ds[:].astype('bool')
    abs_es = np.absolute(im.elsize)
    dt = distance_transform_edt(~mask, sampling=abs_es)

    # mask = im.ds[:].astype('uint32')
    # dt = edt.edt(mask, anisotropy=im.elsize, black_border=True, order='F', parallel=1)
    # TODO?: leverage parallel

    mo = write_output(outpath, dt, im.get_props())

    return mo, mask


def add_noise(data, sd=0.001):

    mask = data == 0
    data += np.random.normal(0, sd, (data.shape))
    data[mask] = 0

    return data


def find_local_maxima(data, size=[13, 13, 3], threshold=0.05, noise_sd=0.0):
    """Find peaks in image."""

    if threshold == -float('Inf'):
        threshold = img.min()

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


# TODO: unused?
def watershed_dog(im, markers, mask=None, inv=True, outpath=''):
    """Calculate watershed."""

    if inv:
    	data = -im.ds[:]
    else:
    	data = im.ds[:]

    ws = watershed(data, markers, mask=mask)

    mo = write_output(outpath, ws, im.get_props())

    return mo


def shift_channel(data, n_planes=0, zdim_idx=0):

    if zdim_idx == 0:
        data[n_planes:, :, :] = data[:-n_planes, :, :]
        data[:n_planes, :, :] = 0
    elif zdim_idx == 2:
        data[:, :, n_planes:] = data[:, :, :-n_planes]
        data[:, :, :n_planes] = 0

    return data


def smooth_channel_inplane(data, sigma=3, filter='median'):

    k = disk(sigma)
    data_smooth = np.zeros_like(data)
    for i, slc in enumerate(data):
        if filter == 'median':
            data_smooth[i, :, :] = median(slc, k)
        elif filter == 'gaussian':
            data_smooth[i, :, :] = gaussian(slc, sigma=sigma, preserve_range=True)

    return data_smooth


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


def split_segments(seg_path, ids='/segm/labels_memb_del_relabeled_fix', outputstem=''):

    labels = LabelImage(seg_path)
    labels.load(load_data=False)
    labels_ds = labels.slice_dataset()

    # nuclei
    outstem = '{}.h5{}'.format(outputstem, '/nucl/dapi')
    maskpath_sauvola = '{}_mask_sauvola'.format(outstem)
    maskpath_absmin = '{}_mask_absmin'.format(outstem)
    mask_nucl = nucl_mask(maskpath_sauvola, maskpath_absmin)
    write(mask_nucl, outstem, '_mask_nuclei', labels, imtype='Mask')

    # membranes  # TODO: may combine with planarity_mask to make it more data-informed
    outstem = '{}.h5{}'.format(outputstem, '/memb/boundary')
    mask_memb = memb_mask(labels_ds)
    write(mask_memb, outstem, '_mask', labels, imtype='Mask')

    outstem = '{}.h5{}'.format(outputstem, ids)
    for mask, pf in zip([mask_memb, mask_nucl], ['_memb', '_nucl']):
        labs = np.copy(labels_ds)
        labs[~mask] = 0
        write(labs, outstem, pf, labels, imtype='Label')


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
                return None

    centreslices = info_dict['centreslices']
    meds = info_dict['medians']
    vmax = info_dict['plotinfo']['vmax']

    aspects = ['equal', 'equal', 'equal']
    aspects = ['equal', 'auto', 'auto']
    for i, (dim, aspect) in enumerate(zip('zyx', aspects)):

        data_nucl = get_data('nucl/dapi_preprocess', 'chan/ch00', dimfac=3)
        data_memb = get_data('memb/mean_smooth', 'memb/mean', dimfac=5)
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
    figpath = '{}_{}-report.pdf'.format(
        info_dict['paths']['out_base'],
        report_type
        )
    f.suptitle(figtitle, fontsize=14, fontweight='bold')
    add_titles(axs, info_dict)
    f.savefig(figpath)

    info_dict.clear()


if __name__ == "__main__":
    main(sys.argv[1:])
