#!/usr/bin/env python

"""Stitch z-stacks with FiJi BigStitcher.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing
import subprocess

import yaml

import numpy as np

import xml.etree.ElementTree as ET

from stapl3d import parse_args, Stapl3r, Image
from stapl3d import imarisfiles
from stapl3d.preprocessing.shading import get_image_info

logger = logging.getLogger(__name__)


def main(argv):
    """Stitch z-stacks with FiJi BigStitcher."""

    def step_splitter(steptype='4step'):
        if steptype == '4step':
            stepmap = {
                'prep': [0],
                'load': [1, 2],
                'calc': [3, 4, 5],
                'fuse': [6],
                'postprocess': [7],
                }
            single_chan = ['calc', 'postprocess']
        elif steptype == '7step':
            stepmap = {
                'prep': [0],
                'define_dataset': [1],
                'load_tile_positions': [2],
                'calculate_shifts': [3],
                'filter_shifts': [4],
                'global_optimization': [5],
                'fuse_dataset': [6],
                'postprocess': [7],
                }
            single_chan = [
                'calculate_shifts',
                'filter_shifts',
                'global_optimization',
                'postprocess',
                ]
        return stepmap, single_chan

    stepmap, single_chan = step_splitter('4step')
    args = parse_args('stitching', list(stepmap.keys()), *argv)

    stitch3r = Stitch3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        stitch3r._fun_selector[step]()


class Stitch3r(Stapl3r):
    """Stitch z-stacks with FiJi BigStitcher.

    Methods
    ----------
    run
        Run all steps in the stitcher module.
    prep
        Read stack positions from metadata and write to BigStitcher format.
    load
        Create BigDataViewer dataset.
    calc
        Calculate pairwise shifts, filter and do global optimization.
    fuse
        Fuse stacks to volume.
    postprocess
        Merge the channels into symlinked file.

    Parameters
    ----------
    image_in : string
        Path to dataset.
    parameter_file : string
        Path to yaml parameter file.
    module_id : string
        Name of the STAPL3D module.
    step_id: string
        Identifier of the yaml parameterfile entry.
    directory : string
        Name of output subdirectory.
    prefix : string
        Output prefix.
    max_workers : int
        Maximal number of cores to use for processing.

    Attributes
    ----------
    stitch_steps : list of int []
        List of indices of stitching steps to perform:
        'prep': [0],
        'define_dataset': [1],
        'load_tile_positions': [2],
        'calculate_shifts': [3],
        'filter_shifts': [4],
        'global_optimization': [5],
        'fuse_dataset': [6],
        'postprocess': [7],
    channels : list of int []
        List of channel indices.
    stacks : list of int []
        List of stack indices.
    channel_ref : int 0
        Reference channel index.
    FIJI : string ''
        Path to Fiji executable.
    inputstem : string ''
        TODO
    z_shift : float 0
        Add a global shift in z.
    elsize : list of floats []
        Voxel sizes (zyxc).
    downsample : list of int [1, 2, 2]
        Downsampling factors for pairwise shift calculation (zyx).
    r : list of float [0.7, 1.0]
        Limits for correlation coefficients between pairs.
    max_shift : list of float [0, 0, 0]
        Limits for translation distance over (zyx).
    max_displacement : float 0
        Limit for total displacement.
    relative_shift : float 2.5
        Relative error threshold for global optimization.
    absolute_shift : float 3.5
        Absolute error threshold for global optimization.

    Examples
    --------
    >>> from stapl3d.preprocessing import stitching
    >>> stitch3r = stitching.Stitch3r(image_in, parameter_file, prefix=dataset)
    >>> stitch3r.FIJI = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
    >>> stitch3r.run()

    """

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Stitch3r, self).__init__(
            image_in, parameter_file,
            module_id='stitching',
            **kwargs,
            )

        self._stepmap = {
            'prep': [0],
            'load': [1, 2],
            'calc': [3, 4, 5],
            'fuse': [6],
            'postprocess': [7],
            }

        self._fun_selector = {
            'prep': self.prep,
            'load': self.load,
            'calc': self.calc,
            'fuse': self.fuse,
            'postprocess': self.postprocess,
            }

        self._parallelization = {
            'prep': [],
            'load': ['channels'],
            'calc': [],
            'fuse': ['channels'],
            'postprocess': [],
            }

        self._parameter_sets = {
            'prep': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            'load': {
                'fpar': self._FPAR_NAMES + ('FIJI',),
                'ppar': ('elsize',),
                'spar': ('_n_workers', 'stitch_steps', 'channels'),
                },
            'calc': {
                'fpar': self._FPAR_NAMES + ('FIJI',),
                'ppar': ('downsample', 'r', 'max_shift', 'max_displacement',
                         'relative_shift', 'absolute_shift'),
                'spar': ('_n_workers', 'stitch_steps'),
                },
            'fuse': {
                'fpar': self._FPAR_NAMES + ('FIJI',),
                'ppar': ('channel_ref', 'z_shift'),
                'spar': ('_n_workers', 'stitch_steps', 'channels'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'channels'),
                },
            }

        self._parameter_table = {}

        default_attr = {
            'stitch_steps': [],
            'channels': [],
            'stacks': [],
            'channel_ref': 0,
            'FIJI': '',
            'inputstem': '',
            'z_shift': 0,
            'elsize': [],
            'downsample': [1, 2, 2],
            'r': [0.7, 1.0],
            'max_shift': [0, 0, 0],
            'max_displacement': 0,
            'relative_shift': 2.5,
            'absolute_shift': 3.5,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        # Set derived parameter defaults.  # TODO: clear warning or error message
        self.FIJI = self.FIJI or os.environ.get('FIJI')
        iminfo = get_image_info(self.image_in)
        self.elsize = self.elsize or iminfo['elsize_zyxc'][:3]

    def _init_paths(self):

        # FIXME: moduledir (=step_id?) can vary
        prev_path = {
            'moduledir': 'shading', 'module_id': 'shading',
            'step_id': 'shading', 'step': 'apply',
            'ioitem': 'outputs', 'output': 'stacks',
            }
        spat = self._get_inpath(prev_path)
        if spat == 'default':
            spat = self._build_path(
                moduledir=prev_path['moduledir'],
                prefixes=[self.prefix, prev_path['module_id']],
                suffixes=[{'c': 'p', 's': 'p'}],
                ext='tif',
                )
        pat = self._suffix_formats['s']
        smat = spat.replace(pat, self._pat2mat(pat))

        tilefile = self._build_path(ext='conf')
        cmat = self._build_path(suffixes=[{'c': '?'}])
        cpat = self._build_filestem(suffixes=[{'c': 'p'}])
        xpat = self._build_path(suffixes=[{'c': 'p'}, 'stacks'], ext='xml')
        hpat = self._build_basename(suffixes=[{'c': 'p'}, 'stacks'], ext='h5')

        self._paths = {
            'prep': {
                'inputs': {
                    'data': self.image_in,
                    },
                'outputs': {
                    'tilefile': tilefile,
                    },
                },
            'load': {
                'inputs': {
                    'stacks': smat,
                    'tilefile': tilefile,
                    },
                'outputs': {
                    'channels': cpat,
                    },
                },
            'calc': {
                'inputs': {
                    'stacks': smat,
                    },
                'outputs': {
                    'channels': cpat,
                    },
                },
            'fuse': {
                'inputs': {
                    'stacks': smat,
                    },
                'outputs': {
                    'channels': cpat,
                    'xml': xpat,
                    'h5stem': hpat,
                    },
            },
            'postprocess': {
                'inputs': {
                    'channels': cmat,
                    },
                'outputs': {
                    'aggregate': self._build_path(ext='bdv'),
                    },
            },
        }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def prep(self, **kwargs):
        """Write tile position offsets."""

        self._prep_step('prep', kwargs)
        self.write_stack_offsets()

    def load(self, **kwargs):
        """Define dataset and load tile positions."""

        self.estimate(step='load', **kwargs)

    def calc(self, **kwargs):
        """Calculate the stitch."""

        self.estimate(step='calc', **kwargs)

    def fuse(self, **kwargs):
        """Fuse the tiles."""

        self.estimate(step='fuse', **kwargs)

    def postprocess(self, **kwargs):
        """Merge the channels in a symlinked file."""

        self._prep_step('postprocess', kwargs)
        self._link_channels()

    def estimate(self, step='', **kwargs):
        """Stitch dataset with Fiji BigStitcher."""

        self.stitch_steps = self._stepmap[step]
        arglist = self._prep_step(step, kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_channel, arglist)

    def _estimate_channel(self, channel):
        """Stitch dataset with Fiji BigStitcher."""

        inputs = self._prep_paths(self.inputs, reps={'c': channel}, abs=False)
        outputs = self._prep_paths(self.outputs, reps={'c': channel}, abs=False)

        macro_path = os.path.realpath(__file__).replace('.py', '.ijm')

        for stitch_step in self.stitch_steps:

            if stitch_step == 6:  # fuse
                self._adapt_xml(channel)

            args = [
                stitch_step,
                channel,
                inputs['stacks'],
                os.path.join(self.datadir, self.directory),
                outputs['channels'],
                inputs['tilefile'] if stitch_step == 2 else 'spam',
                ]
            args += list(self.elsize)
            args += list(self.downsample)
            args += list(self.r) + list(self.max_shift) + [self.max_displacement]
            args += [self.relative_shift, self.absolute_shift]

            cmdlist = [
                self.FIJI,
                '--headless', '--console', '-macro',
                macro_path,
                ' '.join([str(arg) for arg in args]),
                ]

            subprocess.call(cmdlist)

    def _get_arglist(self, parallelized_pars=[]):

        arglist = super()._get_arglist(parallelized_pars)
        if self.step=='calc':
            arglist = [(self.channel_ref,)]

        return arglist

    def _adapt_xml(self, channel, setups=[], replace=True):
        """Copy reference xml, replace filename and add affine mat for zshift."""

        def create_ViewTransform_element(
            name='Transform',
            affine='1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0',
            ):
            """Generate affine-transform xml element.

            <ViewTransform type="affine">
              <Name>Stitching Transform</Name>
              <affine>1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0</affine>
            </ViewTransform>
            """
            vt_new = ET.Element('ViewTransform', attrib={'type': 'affine'})
            b = ET.SubElement(vt_new, 'Name')
            b.text = name
            c = ET.SubElement(vt_new, 'affine')
            c.text = affine
            return vt_new

        outputs = self._prep_paths(self.outputs, reps={'c': channel})
        outputs_ref = self._prep_paths(self.outputs, reps={'c': self.channel_ref})
        outputs_h5 = self._prep_paths(self.outputs, reps={'c': channel}, abs=False)

        shutil.copy2(outputs['xml'], '{}.orig'.format(outputs['xml']))
        if channel != self.channel_ref:
            shutil.copy2(outputs_ref['xml'], outputs['xml'])

        tree = ET.parse(outputs['xml'])
        root = tree.getroot()

        if replace:
            h5 = root.find('SequenceDescription').find('ImageLoader').find('hdf5')
            h5.text = outputs_h5['h5stem']

        if self.z_shift:
            vrs = root.find('ViewRegistrations')
            affine = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 {:03f}'.format(self.z_shift)
            vt_new = create_ViewTransform_element(affine=affine)
            for vr in vrs.findall('ViewRegistration'):
                setup = vr.get('setup')
                if not setups:
                    vr.insert(0, vt_new)
                elif int(setup) in setups:
                    vr.insert(0, vt_new)

        tree.write(outputs['xml'], encoding='utf-8', xml_declaration=True)

    def write_stack_offsets(self):
        """Write tile offsets to file."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        # entry per channel X tile
        iminfo = get_image_info(inputs['data'])
        vo = np.tile(iminfo['stack_offsets'], [iminfo['nchannels'], 1])
        vo[:, 0] = list(range(0, vo.shape[0]))
        np.savetxt(outputs['tilefile'], vo,
                   fmt='%d;;(%10.5f, %10.5f, %10.5f)',
                   header='dim=3', comments='')

    def _link_channels(self):
        """Merge the channels in a symlinked file."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        # TODO: remove workaround: => directly to bdv
        base, ext = os.path.splitext(outputs['aggregate'])
        h5path = f'{base}.h5'
        imarisfiles.aggregate_bigstitcher(h5path, inputpat=inputs['channels'], ext='.h5')
        os.rename(h5path, outputs['aggregate'])
        # TODO: cat xmls


if __name__ == "__main__":
    main(sys.argv[1:])
