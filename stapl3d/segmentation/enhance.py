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
    h5_nii_convert,
    )

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
    ids_membrane='memb/mean',
    ods_preprocess='memb/preprocess',
    ods_planarity='memb/planarity',
    median_filter_par=0.5,
    membrane_filter_par=1.1,
    cleanup=True,
    ):
    """Perform planarity estimation."""

    outputdir = get_outputdir(image_in, parameter_file, outputdir,
                              step_id, 'blocks')

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    filepaths, blocks = get_blockfiles(image_in, outputdir, params['blocks'])

    ACMEdir = ACMEdir or os.environ.get('ACME')

    arglist = [
        (
            filepath,
            ACMEdir,
            params['ids_membrane'],
            params['ods_preprocess'],
            params['ods_planarity'],
            params['median_filter_par'],
            params['membrane_filter_par'],
            params['cleanup'],
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
    ids_membrane='memb/mean',
    ods_preprocess='memb/preprocess',
    ods_planarity='memb/planarity',
    median_filter_par=0.5,
    membrane_filter_par=1.1,
    cleanup=True,
    step_id='membrane_enhancement',
    outputdir='',
    ):
    """Perform membrane enhancement."""

    blockstem = os.path.splitext(filepath)[0]

    logging.basicConfig(filename='{}.log'.format(blockstem), level=logging.INFO)
    report = {'parameters': locals()}

    image_in = '{}.h5/{}'.format(blockstem, ids_membrane)
    niipath_memb = '{}_{}.nii.gz'.format(blockstem, ids_membrane.replace('/', '-'))
    h5path_prep = '{}.h5/{}'.format(blockstem, ods_preprocess)
    niipath_prep = '{}_{}.nii.gz'.format(blockstem, ods_preprocess.replace('/', '-'))
    h5path_plan = '{}.h5/{}'.format(blockstem, ods_planarity)
    niipath_plan = '{}_{}.nii.gz'.format(blockstem, ods_planarity.replace('/', '-'))

    h5_nii_convert(image_in, niipath_memb, datatype='uint8')

    subprocess.call([
        os.path.join(ACMEdir, "cellPreprocess"),
        niipath_memb, niipath_prep,
        "{}".format(median_filter_par),
    ])

    subprocess.call([
        os.path.join(ACMEdir, "multiscalePlateMeasureImageFilter"),
        niipath_prep, niipath_plan,
        "{}_eigen.mha".format(blockstem),
        "{}".format(membrane_filter_par),
    ])

    """
    subprocess.call([
        os.path.join(ACMEdir, "membraneVotingField3D"),
        niipath_plan,
        "{}_eigen.mha".format(blockstem),
        "{}_TV.mha".format(blockstem),
        "1.0",
    ])

    subprocess.call([
        os.path.join(ACMEdir, "membraneSegmentation"),
        niipath_prep,
        "{}_TV.mha".format(blockstem),
        "{}_segment.mha".format(blockstem),
        "1.0",
    ])
    """

    if cleanup:
        h5_nii_convert(niipath_prep, h5path_prep)
        h5_nii_convert(niipath_plan, h5path_plan)
        os.remove(niipath_memb)
        os.remove(niipath_prep)
        os.remove(niipath_plan)
        os.remove("{}_eigen.mha".format(blockstem))


if __name__ == "__main__":
    main(sys.argv[1:])
