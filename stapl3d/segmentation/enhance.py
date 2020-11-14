#!/usr/bin/env python

"""Perform planarity estimation.

"""

import os
import sys
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
    parse_args_common,
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
    """"Enhance the membrane with ACME.

    """

    step_ids = ['membrane_enhancement']
    fun_selector = {
        'estimate': estimate,
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
    step_id='membrane_enhancement',
    outputdir='',
    n_workers=0,
    blocks=[],  # FIXME: note that block selection implies selection of globbed files, not block idxs..
    ACMEdir='',
    median_filter_par=0.5,
    membrane_filter_par=1.1,
    ):
    """Perform planarity estimation."""

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
            step_id,
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
    step_id='membrane_enhancement',
    outputdir='',
    ):
    """Perform membrane enhancement."""

    # Prepare the output.
    blockstem = os.path.splitext(filepath)[0]

    logging.basicConfig(filename='{}.log'.format(blockstem), level=logging.INFO)
    report = {'parameters': locals()}

    grp = 'memb'; ids = 'mean'; # TODO: make flexible

    image_in = '{}.h5/{}/{}'.format(blockstem, grp, ids)
    image_out = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)
    h5_nii_convert(image_in, image_out, datatype='uint8')
    #h5_nii_convert(image_in, image_out)  # dtype_in='uint8'  (NOTE: used for mito: airyscan63x, FIXME: this was actually uint16 with low range)

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

    grp = 'memb'; ids = 'preprocess';  # TODO: make flexible
    image_in = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)
    image_out = '{}.h5/{}/{}'.format(blockstem, grp, ids)
    h5_nii_convert(image_in, image_out)

    grp = 'memb'; ids = 'planarity';  # TODO: make flexible
    image_in = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)
    image_out = '{}.h5/{}/{}'.format(blockstem, grp, ids)
    h5_nii_convert(image_in, image_out)


if __name__ == "__main__":
    main(sys.argv[1:])
