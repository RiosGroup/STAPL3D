#!/usr/bin/env python

"""....
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

from stapl3d import parse_args, Stapl3r, Image
from stapl3d.blocks import Block3r

#from structure_tensor import eig_special_3d, structure_tensor_3d
#print(structure_tensor_3d)

logger = logging.getLogger(__name__)


def main(argv):
    """...."""

    steps = ['estimate']
    args = parse_args('structure_tensor', steps, *argv)

    structur3r = Structur3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        structur3r._fun_selector[step]()


class Structur3r(Block3r):
    """."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'structure_tensor'

        super(Structur3r, self).__init__(
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
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'ids_image': 'ch00',
            'ods_image': '',
            'sigma': 1.5,
            'rho': 5.5,
            'odss': ['ST', 'STval', 'STvec', 'STnrg'],
            'use_gpu': False,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_structurer()

        self._init_log()

        self._prep_blocks()

        self._images = []
        self._labels = []

    def _init_paths_structurer(self):

        bpat = self._l2p(['blocks', '{f}.h5'])

        self._paths.update({
            'estimate': {
                'inputs': {
                    'blockfiles': f'{bpat}',
                    },
                'outputs': {
                    'blockfiles': f'{bpat}',
                    'report': f'{bpat}'.replace('.h5', '.pdf'),
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def estimate(self, **kwargs):
        """...."""

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, block_idx):
        """...."""

        block = self._blocks[block_idx]

        inputs = self._prep_paths_blockfiles(self.inputs, block)
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        step = 'estimate'
        params = self._cfg[self.step_id][step]

        filepath = inputs['blockfiles']

        if self.use_gpu:
            print('Using GPU')
            import cupy as cp
            from structure_tensor.cp import eig_special_3d, structure_tensor_3d
        else:
            print('Using CPU')
            from structure_tensor import eig_special_3d, structure_tensor_3d

        data, props = load_data(filepath, self.ids_image)
        S = structure_tensor_3d(data.astype('float'), self.sigma, self.rho)
        val, vec = eig_special_3d(S.astype(float))

        out = {}
        ods_image = self.ods_image or self.ids_image
        ods_image += '_'
        if 'ST' in self.odss:
            out[f'{ods_image}ST'] = cp.asnumpy(S) if self.use_gpu else S
        if 'STval' in self.odss:
            out[f'{ods_image}STval'] = cp.asnumpy(val) if self.use_gpu else val
        if 'STvec' in self.odss:
            out[f'{ods_image}STvec'] = cp.asnumpy(vec) if self.use_gpu else vec
        if 'STnrg' in self.odss:
            out[f'{ods_image}STnrg'] = cp.asnumpy(cp.sum(val, axis=0)) if self.use_gpu else np.sum(val, axis=0)

        write_structure_tensor(filepath, out, props)


def load_data(filepath, ids):
    im = Image(filepath + f'/{ids}')
    im.load()
    props = im.get_props()
    data = im.slice_dataset()
    im.close()
    return data, props


def write_data(filepath, ods, props, data):
    im = Image(filepath + f'/{ods}', **props)
    im.create()
    im.write(data)
    im.close()


def expand_props(props, data):
    newprops = {k: v for k, v in props.items()}
    newprops['dtype'] = data.dtype
    newprops['shape'] = data.shape
    if data.ndim == 4:
        newprops['elsize'] = [1] + props['elsize']
        newprops['axlab'] = 'c' + props['axlab']
        newprops['chunks'] = [data.shape[0]] + list(props['chunks'])
    newprops['slices'] = None
    return newprops


def write_structure_tensor(filepath, odss, props):
    for ods, data in odss.items():
        newprops = expand_props(props, data)
        write_data(filepath, ods, newprops, data)


if __name__ == "__main__":
    main(sys.argv[1:])
