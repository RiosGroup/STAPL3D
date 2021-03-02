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

from stapl3d import parse_args, Stapl3r, Image, h5_nii_convert
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
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        enhancer._fun_selector[step]()


class Enhancer(Blocker):
    """Enhance the membrane with ACME."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'membrane_enhancement'

        super(Enhancer, self).__init__(
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
            'median_filter_par': 0.5,
            'membrane_filter_par': 1.1,
            'cleanup': True,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        # image_in = self._paths['blockinfo']['inputs']['data']
        # block_formatter = self._paths['blockinfo']['outputs']['blockfiles']
        # self.set_blocksize(image_in)
        # self.set_blockmargin()
        # self.set_blocks(image_in, block_formatter)

        self.ACMEdir = self.ACMEdir or os.environ.get('ACME')

    def _init_paths(self):

        # FIXME: moduledir (=step_id?) can vary
        prev_path = {
            'moduledir': 'blocks', 'module_id': 'blocks',
            'step_id': 'blocks', 'step': 'split',
            'ioitem': 'outputs', 'output': 'blockfiles',
            }
        bpat = self._get_inpath(prev_path)
        bpat = self._build_path(
            moduledir=prev_path['moduledir'],
            prefixes=[self.prefix, prev_path['module_id']],
            suffixes=[{'b': 'p'}],
            ext='h5',
            )

        self._paths.update({
            'estimate': {
                'inputs': {
                    'membrane': f'{bpat}/memb/mean',
                    },
                'outputs': {
                    'preprocess': f"{bpat}/memb/ACME_preprocess",
                    'planarity': f"{bpat}/memb/ACME_planarity",
                    # 'eigen': f"{bpat}.h5/memb/ACME_eigen",
                    # 'tv': f"{bpat}.h5/memb/ACME_tv",
                    # 'segment': f"{bpat}.h5/memb/ACME_segment",
                    }
                },
            })

        for step in ['estimate']:  #self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

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

    def _estimate_block(self, block):
        """Perform planarity estimation for a block."""

        def h5path2nii(h5path):
            base, intpath = h5path.split('.h5/')
            ods = intpath.replace('/', '-')
            return f"{base}_{ods}.nii.gz"

        block = self._blocks[block]
        inputs = self._prep_paths(self.inputs, reps={'b': block.idx})
        outputs = self._prep_paths(self.outputs, reps={'b': block.idx})

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


if __name__ == "__main__":
    main(sys.argv[1:])
