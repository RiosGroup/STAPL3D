#!/usr/bin/env python

"""Functions for generating reports.

"""

import sys
import argparse

import os
import numpy as np
import pickle

import h5py

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.color import label2rgb

from stapl3d import Image


def get_centreslices(info_dict, idss=[], ch=0, axlab='zyx'):
    """Get the centreslices for all dims and steps."""

    def extract(name, node):
        if isinstance(node, h5py.Dataset):
            idss.append(name)
        return None

    h5_path = info_dict['paths']['file']
    if not idss:
        with h5py.File(h5_path, 'r') as f:
            f.visititems(extract)

    centreslices = {ids: {dim: get_centreslice(h5_path, ids, dim, ch)
                          for dim in axlab}
                    for ids in idss}

    return centreslices


def get_centreslice(image_in, ids, dim='z', ch=0, tp=0):
    """Return an image's centreslice for a dimension."""

    if isinstance(image_in, Image):
        im = image_in
    else:
        im = Image(f'{image_in}/{ids}', permission='r')
        try:
            im.load(load_data=False)
        except KeyError:
            print('dataset {} not found'.format(ids))
            return None

    slcs = [slc for slc in im.slices]

    if len(im.dims) == 2:
        data = im.slice_dataset()
        return data

    if 'c' in im.axlab:
        ch_idx = im.axlab.index('c')
        im.slices[ch_idx] = slice(ch, ch + 1, 1)

    if 't' in im.axlab:
        tp_idx = im.axlab.index('t')
        im.slices[tp_idx] = slice(tp, tp + 1, 1)

    dim_idx = im.axlab.index(dim)
    cslc = int(im.dims[dim_idx] / 2)
    im.slices[dim_idx] = slice(cslc, cslc + 1)

    data = im.slice_dataset()

    im.slices = slcs

    return data



def gen_orthoplot(f, gs, size_xy=5, size_z=1):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_t = size_xy + size_z

    gs_sub = gs.subgridspec(size_t, size_t)

    # central: yx-image
    axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy]))
    # middle-bottom: zx-image
    axs.append(f.add_subplot(gs_sub[size_xy:, :size_xy], sharex=axs[0]))
    # right-middle: zy-image
    axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:], sharey=axs[0]))

    return axs


def gen_orthoplot_meh(f, gs, size_xy=5, size_z=1):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_t = size_xy + size_z

    gs_sub = gs.subgridspec(size_t, size_t)

    # full histogram
    ax = axdict['histogram']

    ax.hist(np.ravel(image), bins=256, log=logscale, color=[0, 0, 0])
    ax.set_xlim([0, dmax])


    # smoothed image histogram with thresholds
    ax = axdict['histogram smoothed image']
    ax.hist(np.ravel(image_smooth), bins=256, log=logscale, color=[0, 0, 0])
    linestyles = '--:'
    if info_dict['threshold_otsu'] is None:
        thresholds = thresholds[:2]
        linestyles = linestyles[:2]
        thrcolors = thrcolors[:2]
    labels = ['{:.5}'.format(float(thr)) for thr in thresholds]
    self._draw_thresholds(ax, thresholds, thrcolors, linestyles, labels)



def gen_orthoplot_with_colorbar(f, gs, cbar='vertical', idx=0, add_profile_insets=False):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_xy = 20
    size_z = 4
    size_c = 2
    size_t = size_xy + size_z + size_c

    # gs_sub = gs.subgridspec(size_xy + size_z + size_c, size_xy + size_z + size_c)

    if cbar == 'horizontal':
        gs_sub = gs.subgridspec(size_xy + size_z + size_c, size_xy + size_z)
        # central: yx-image
        axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy], aspect='equal'))
        # bottom: zx-image
        axs.append(f.add_subplot(gs_sub[size_xy:-size_c, :size_xy], sharex=axs[0], aspect='auto'))
        # right: zy-image
        axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:-size_c], sharey=axs[0], aspect='auto'))
        # bottom colorbar
        axs.append(f.add_subplot(gs_sub[-size_c:, :size_xy]))
    else:
        if not idx % 2:  # left side plots
            gs_sub = gs.subgridspec(size_t, size_t)
            axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy]))  # central: yx-image
            axs.append(f.add_subplot(gs_sub[size_xy:-size_c, :size_xy]))  # bottom: zx-image
            axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:-size_c]))  # right: zy-image
            axs.append(f.add_subplot(gs_sub[2:size_xy, -size_c+1:]))  # right colorbar
            # axs.append(f.add_subplot(gs_sub[-size_c+1:, 2:size_xy]))  # bottom colorbar
            if add_profile_insets:
                axs.append(f.add_subplot(gs_sub[:size_c+1, size_xy:size_xy+size_c+2])) # z
                axs.append(f.add_subplot(gs_sub[:size_xy, :size_c+1])) # y
                axs.append(f.add_subplot(gs_sub[:size_c+1, :size_xy])) # x
        else:  # right side plots
            gs_sub = gs.subgridspec(size_t, size_t)
            axs.append(f.add_subplot(gs_sub[:size_xy, size_c:size_xy+size_c]))  # central: yx-image
            axs.append(f.add_subplot(gs_sub[size_xy:-size_c, size_c:size_xy+size_c]))  # bottom: zx-image
            axs.append(f.add_subplot(gs_sub[:size_xy, -size_z:]))  # right: zy-image
            axs.append(f.add_subplot(gs_sub[2:size_xy, :size_c-1]))  # right colorbar
            # axs.append(f.add_subplot(gs_sub[-size_c+1:, size_c:size_xy]))  # bottom colorbar

    return axs


def gen_orthoplot_with_profiles(f, gs):
    """Create axes on a subgrid to fit three orthogonal projections."""

    axs = []
    size_xy = 20
    size_z = 4
    size_c = 2

    size_t = size_xy + size_z + size_c

    gs_sub = gs.subgridspec(size_t, size_t)

    axs.append(f.add_subplot(gs_sub[:size_xy, :size_xy]))  # central: yx-image
    axs.append(f.add_subplot(gs_sub[size_xy:-size_c, :size_xy]))  # bottom: zx-image
    axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:-size_c]))  # right: zy-image
    axs.append(f.add_subplot(gs_sub[2:size_xy, -size_c+1:]))  # right colorbar

    axs.append(f.add_subplot(gs_sub[:size_xy, :size_c]))
    axs.append(f.add_subplot(gs_sub[:size_c, :size_xy]))
    axs.append(f.add_subplot(gs_sub[:size_xy, size_xy:size_xy + size_c]))

    return axs
