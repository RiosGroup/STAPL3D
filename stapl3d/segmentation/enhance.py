#!/usr/bin/env python

"""Enhance the membrane with ACME.

    # TODO: reports
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

from stapl3d import parse_args, Stapl3r, Image, format_, h5_nii_convert
from stapl3d.blocks import Blocker

logger = logging.getLogger(__name__)


def main(argv):
    """Enhance the membrane with ACME."""

    steps = ['estimate']
    args = parse_args('membrane_enhancement', steps, *argv)

    enhancer = Enhancer(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        dataset=args.dataset,
        suffix=args.suffix,
        n_workers=args.n_workers,
    )

    for step in args.steps:
        enhancer._fun_selector[step]()


class Enhancer(Blocker):
    """Enhance the membrane with ACME."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Enhancer, self).__init__(
            image_in, parameter_file,
            module_id='membrane_enhancement',
            **kwargs,
            )

        self._fun_selector = {
            'estimate': self.estimate,
            }

        default_attr = {
            'step_id': 'membrane_enhancement',
            'blocks': [],
            'ACMEdir': '',
            'ids_membrane': 'memb/mean',
            'ods_preprocess': 'memb/preprocess',
            'ods_planarity': 'memb/planarity',
            'median_filter_par': 0.5,
            'membrane_filter_par': 1.1,
            'cleanup': True,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        self.set_directory(subdirectory='blocks')

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self.set_blocksize()
        self.set_blockmargin()
        self.set_blocks()
        self.set_blockfiles()

        self.ACMEdir = self.ACMEdir or os.environ.get('ACME')

        self._parsets = {
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('ACMEdir',
                         'ids_membrane', 'ods_preprocess', 'ods_planarity',
                         'median_filter_par', 'membrane_filter_par', 'cleanup'),
                'spar': ('n_workers', 'blocks', 'blockfiles'),
                },
            }

        # TODO: merge with parsets?
        self._partable = {
            'median_filter_par': 'Median filter parameter',
            'membrane_filter_par': 'Membrane filter parameter',
            }

    def estimate(self, **kwargs):
        """Perform planarity estimation.

        blocks=[],  # FIXME: note that block selection implies selection of globbed files, not block idxs..
        ACMEdir='',
        ids_membrane='memb/mean',
        ods_preprocess='memb/preprocess',
        ods_planarity='memb/planarity',
        median_filter_par=0.5,
        membrane_filter_par=1.1,
        cleanup=True,
        """

        self.set_parameters('estimate', kwargs)
        arglist = self._get_arglist(['blockfiles'])
        self.set_n_workers(len(arglist))
        self.dump_parameters(step=self.step)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, filepath):
        """Perform planarity estimation for a block."""

        blockstem = os.path.splitext(filepath)[0]

        image_in = '{}.h5/{}'.format(blockstem, self.ids_membrane)
        niipath_memb = '{}_{}.nii.gz'.format(blockstem, self.ids_membrane.replace('/', '-'))
        h5path_prep = '{}.h5/{}'.format(blockstem, self.ods_preprocess)
        niipath_prep = '{}_{}.nii.gz'.format(blockstem, self.ods_preprocess.replace('/', '-'))
        h5path_plan = '{}.h5/{}'.format(blockstem, self.ods_planarity)
        niipath_plan = '{}_{}.nii.gz'.format(blockstem, self.ods_planarity.replace('/', '-'))

        h5_nii_convert(image_in, niipath_memb, datatype='uint8')

        subprocess.call([
            os.path.join(self.ACMEdir, "cellPreprocess"),
            niipath_memb, niipath_prep,
            "{}".format(self.median_filter_par),
        ])

        subprocess.call([
            os.path.join(self.ACMEdir, "multiscalePlateMeasureImageFilter"),
            niipath_prep, niipath_plan,
            "{}_eigen.mha".format(blockstem),
            "{}".format(self.membrane_filter_par),
        ])

        """
        subprocess.call([
            os.path.join(self.ACMEdir, "membraneVotingField3D"),
            niipath_plan,
            "{}_eigen.mha".format(blockstem),
            "{}_TV.mha".format(blockstem),
            "1.0",
        ])

        subprocess.call([
            os.path.join(self.ACMEdir, "membraneSegmentation"),
            niipath_prep,
            "{}_TV.mha".format(blockstem),
            "{}_segment.mha".format(blockstem),
            "1.0",
        ])
        """

        if self.cleanup:
            h5_nii_convert(niipath_prep, h5path_prep)
            h5_nii_convert(niipath_plan, h5path_plan)
            os.remove(niipath_memb)
            os.remove(niipath_prep)
            os.remove(niipath_plan)
            os.remove("{}_eigen.mha".format(blockstem))


if __name__ == "__main__":
    main(sys.argv[1:])
