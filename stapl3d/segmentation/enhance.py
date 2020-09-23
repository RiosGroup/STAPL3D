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
    get_outputdir,
    get_blockfiles,
    get_params,
    get_n_workers,
    Image,
    LabelImage,
    MaskImage,
    wmeMPI,
    )

from stapl3d import h5_nii_convert

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
    ACMEdir='',
    median_filter_par=0.5,
    membrane_filter_par=1.1,
    ):
    """Perform planarity estimation."""

    step_id = 'membrane_enhancement'

    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, fallback='blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    ACMEdir = ACMEdir or os.environ.get('ACME')

    arglist = [
        (
            filepath,
            ACMEdir,
            params['median_filter_par'],
            params['membrane_filter_par'],
            outputdir,
        )
        for block_idx, filepath in zip(blocks, filepaths)]

    n_workers = get_n_workers(len(blocks), subparams)
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
        "{}".format(median_filter_par),
    ])

    subprocess.call([
        os.path.join(ACMEdir, "multiscalePlateMeasureImageFilter"),
        "{}_memb-preprocess.nii.gz".format(blockstem),
        "{}_memb-planarity.nii.gz".format(blockstem),
        "{}_memb-eigen.mha".format(blockstem),
        "{}".format(membrane_filter_par),
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
