#!/usr/bin/env python

"""Perform planarity estimation.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from glob import glob

from stapl3d import (
    get_n_workers,
    prep_outputdir,
    get_params,
    get_paths,
    Image,
    LabelImage,
    MaskImage,
    wmeMPI,
    get_image,
    split_filename,
    )

from stapl3d.channels import h5_nii_convert

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
    blocks=[],
    ACMEdir='',
    median_filter_par=0.5,
    membrane_filter_par=1.1,
    ):
    """Perform planarity estimation."""

    step_id = 'membrane_enhancement'

    dirs = get_params(dict(), parameter_file, 'dirtree')
    try:
        subdir = dirs['datadir'][step_id] or ''
    except KeyError:
        subdir = 'blocks'
    outputdir = prep_outputdir(outputdir, image_in, subdir)

    params = get_params(locals(), parameter_file, step_id)

    ipf = ''
    paths = get_paths(image_in)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)
    filepat = '{}_*{}.h5'.format(dataset, ipf)
    filepaths = glob(os.path.join(outputdir, filepat))
    filepaths.sort()
    if params['blocks']:
        filepaths = [filepaths[i] for i in params['blocks']]

    n_workers = get_n_workers(len(params['blocks']), params)

    ACMEdir = ACMEdir or os.environ.get('ACME')

    arglist = [
        (
            filepath,
            ACMEdir,
            params['median_filter_par'],
            params['membrane_filter_par'],
            outputdir,
        )
        for filepath in filepaths]

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(membrane_enhancement, arglist)


def membrane_enhancement(
    filepath,
    ACMEdir,
    median_filter_par=0.5,
    membrane_filter_par=1.1,
    outputdir='',
    ):
    """Perform membrane enhancement."""

    # Prepare the output.
    blockstem = os.path.splitext(filepath)[0]

    logging.basicConfig(filename='{}.log'.format(blockstem), level=logging.INFO)
    report = {'parameters': locals()}

    grp = 'memb'

    ids = 'mean'
    image_in = '{}.h5/{}/{}'.format(blockstem, grp, ids)
    image_out = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)
    h5_nii_convert(image_in, image_out, datatype='uint8')

    subprocess.call([
        os.path.join(ACMEdir, "cellPreprocess"),
        "{}_memb-mean.nii.gz".format(blockstem),
        "{}_memb-preprocess.nii.gz".format(blockstem),
        "0.5",
    ])

    subprocess.call([
        os.path.join(ACMEdir, "multiscalePlateMeasureImageFilter"),
        "{}_memb-preprocess.nii.gz".format(blockstem),
        "{}_memb-planarity.nii.gz".format(blockstem),
        "{}_memb-eigen.mha".format(blockstem),
        "1.1",
    ])

    """
    subprocess.call([
        os.path.join(ACMEdir, "membraneVotingField3D"),
        "{}_memb-planarity.nii.gz".format(blockstem),
        "{}_memb-eigen.mha".format(blockstem),
        "{}_memb-TV.mha".format(blockstem),
        "1.0",
    ])

    subprocess.call([
        os.path.join(ACMEdir, "membraneSegmentation"),
        "{}_memb-preprocess.nii.gz".format(blockstem),
        "{}_memb-TV.mha".format(blockstem),
        "{}_memb-segment.mha".format(blockstem),
        "1.0",
    ])
    """

    ids = 'preprocess'
    image_in = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)
    image_out = '{}.h5/{}/{}'.format(blockstem, grp, ids)
    h5_nii_convert(image_in, image_out)

    ids = 'planarity'
    image_in = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)
    image_out = '{}.h5/{}/{}'.format(blockstem, grp, ids)
    h5_nii_convert(image_in, image_out)


if __name__ == "__main__":
    main(sys.argv[1:])
