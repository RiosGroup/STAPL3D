#!/usr/bin/env python

"""Enhance the membrane with ACME.

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

from stapl3d import parse_args, Stapl3r, Image, h5_nii_convert
from stapl3d.blocks import Block3r

logger = logging.getLogger(__name__)


def main(argv):
    """Enhance the membrane with ACME."""

    steps = ['estimate']
    args = parse_args('membrane_enhancement', steps, *argv)

    enhanc3r = Enhanc3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        enhanc3r._fun_selector[step]()


class Enhanc3r(Block3r):
    """Enhance the membrane with ACME."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'membrane_enhancement'

        super(Enhanc3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'estimate': self.estimate,
            })

        self._parallelization.update({
            'estimate': ['blocks'],
            })

        self._parameter_sets.update({
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('ACMEdir',
                         'ids_membrane', 'ods_preprocess', 'ods_planarity',
                         'median_filter_par', 'membrane_filter_par', 'cleanup'),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        self._parameter_table.update({
            'median_filter_par': 'Median filter parameter',
            'membrane_filter_par': 'Membrane filter parameter',
            })

        default_attr = {
            'ACMEdir': '',
            'ids_membrane': 'memb/mean',
            'ods_preprocess': 'memb/ACME_preprocess',
            'ods_planarity': 'memb/ACME_planarity',
            'median_filter_par': 0.5,
            'membrane_filter_par': 1.1,
            'cleanup': True,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_enhancer()

        self._init_log()

        self._prep_blocks()

        self.ACMEdir = self.ACMEdir or os.environ.get('ACME')

        self._images = []
        self._labels = []

    def _init_paths_enhancer(self):

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        self._paths.update({
            'estimate': {
                'inputs': {
                    'blockfiles': blockfiles,
                    'membrane': f'{blockfiles}/{self.ids_membrane}',
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    'preprocess': f"{blockfiles}/{self.ods_preprocess}",
                    'planarity': f"{blockfiles}/{self.ods_planarity}",
                    # 'eigen': f"{blockfiles}/memb/ACME_eigen",
                    # 'tv': f"{blockfiles}/memb/ACME_tv",
                    # 'segment': f"{blockfiles}/memb/ACME_segment",
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

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

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, block_idx):
        """Perform planarity estimation for a block."""

        def h5path2nii(h5path):
            base, intpath = h5path.split('.h5/')
            ods = intpath.replace('/', '-')
            return f"{base}_{ods}.nii.gz"

        block = self._blocks[block_idx]
        inputs = self._prep_paths_blockfiles(self.inputs, block)
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        niipaths = {'membrane': h5path2nii(inputs['membrane'])}
        ACME_vols = ['preprocess', 'planarity', 'eigen', 'tv', 'segment']
        for vol in ACME_vols:
            if vol in outputs.keys():
                niipaths[vol] = h5path2nii(outputs[vol])
            else:
                niipaths[vol] = f"{niipaths['membrane']}.TMP{vol}.mha"
                # TODO: mha to h5 option

        h5_nii_convert(inputs['membrane'], niipaths['membrane'], datatype='uint8')

        idx = ACME_vols.index('preprocess')
        if any([vol in outputs.keys() for vol in ACME_vols[idx:]]):
            subprocess.call([
                os.path.join(self.ACMEdir, "cellPreprocess"),
                niipaths['membrane'], niipaths['preprocess'],
                f"{self.median_filter_par}",
            ])

        idx = ACME_vols.index('planarity')
        if any([vol in outputs.keys() for vol in ACME_vols[idx:]]):
            subprocess.call([
                os.path.join(self.ACMEdir, "multiscalePlateMeasureImageFilter"),
                niipaths['preprocess'], niipaths['planarity'], niipaths['eigen'],
                f"{self.membrane_filter_par}",
            ])

        idx = ACME_vols.index('tv')
        if any([vol in outputs.keys() for vol in ACME_vols[idx:]]):
            subprocess.call([
                os.path.join(self.ACMEdir, "membraneVotingField3D"),
                niipaths['planarity'], niipaths['eigen'], niipaths['tv'],
                "1.0",
            ])

        idx = ACME_vols.index('segment')
        if any([vol in outputs.keys() for vol in ACME_vols[idx:]]):
            subprocess.call([
                os.path.join(self.ACMEdir, "membraneSegmentation"),
                niipaths['preprocess'], niipaths['tv'], niipaths['segment'],
                "1.0",
            ])

        if self.cleanup:
            for vol in ['membrane'] + ACME_vols:
                if vol in outputs.keys():
                    h5_nii_convert(niipaths[vol], outputs[vol])
                try:
                    os.remove(niipaths[vol])
                except FileNotFoundError:
                    pass

    def view(self, input=[], images=[], labels=[], settings={}):

        images = images or [self.ids_membrane, self.ods_preprocess, self.ods_planarity]

        """

        if isinstance(input, str):
            super().view(input, images, labels, settings)
        elif type(input) == int or type(input) == float:
            filepath = self._abs(self.outputpaths['estimate']['blockfiles'].format(b=input))
            super().view(filepath, images, labels, settings)
        else:
            input = input or [0, 1]
            super().view_blocks(input, images, labels, settings)
        """

        if images is not None:
            images = images or self._images
        if labels is not None:
            labels = labels or self._labels

        if isinstance(input, (int, float)):
            input = self._blocks[input].path
        else:
            input = input or [0, 1]

        super().view(input, images, labels, settings)


if __name__ == "__main__":
    main(sys.argv[1:])
