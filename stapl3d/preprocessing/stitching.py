#!/usr/bin/env python

"""Stitching helper functions.

    # TODO: passing stitching arguments from parfile to ijm
    # TODO: report
    # TODO: logging
    # TODO: cleanup shading stacks after converting to bdv
    # TODO: check if tileoffsets order needs to be switched after ch<=>tile order switch
    # TODO: check on data stitchability (i.e. mostly for ntiles=1)
    # TODO: put shading suffix in stitching intermediate filenames?
    # TODO: outstem not used for ijm

    # no good way to name fused if datset and suffix are unset

    # logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)
    # report = {'parameters': locals()}

FIXME: on calc:
java.lang.NumberFormatException: empty String
	at sun.misc.FloatingDecimal.readJavaFormatString(FloatingDecimal.java:1842)
	at sun.misc.FloatingDecimal.parseDouble(FloatingDecimal.java:110)
	at java.lang.Double.parseDouble(Double.java:538)
	at net.imagej.patcher.HeadlessGenericDialog.getMacroParameter(HeadlessGenericDialog.java:348)
	at net.imagej.patcher.HeadlessGenericDialog.addNumericField(HeadlessGenericDialog.java:128)
	at net.preibisch.stitcher.gui.popup.SimpleRemoveLinkPopup.filterPairwiseShifts(SimpleRemoveLinkPopup.java:78)
	at net.preibisch.stitcher.plugin.Filter_Pairwise_Shifts.run(Filter_Pairwise_Shifts.java:40)
	at ij.IJ.runUserPlugIn(IJ.java:230)
	at ij.IJ.runPlugIn(IJ.java:193)
	at ij.Executer.runCommand(Executer.java:137)
	at ij.Executer.run(Executer.java:66)
	at ij.IJ.run(IJ.java:312)
	at ij.IJ.run(IJ.java:323)
...

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

import czifile
from readlif.reader import LifFile

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

    stitcher = Stitcher(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        dataset=args.dataset,
        suffix=args.suffix,
        n_workers=args.n_workers,
    )

    for step in args.steps:
        stitcher._fun_selector[step]()


class Stitcher(Stapl3r):
    """Stitch z-stacks with FiJi BigStitcher."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Stitcher, self).__init__(
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

        default_attr = {
            'step_id': 'stitching',
            'stitch_steps': [],
            'channels': [],
            'channel_ref': 0,
            'FIJI': '',
            'inputstem': '',
            'tilefile': '',
            'z_shift': 0,
            'elsize': [],
            'downsample': [1, 2, 2],
            'r': [0.7, 1.0],
            'max_shift': [0, 0, 0],
            'max_displacement': 0,
            'relative_shift': 2.5,
            'absolute_shift': 3.5,
            'outputpath': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        # Set derived parameter defaults.
        # TODO: clear warning or error message
        self.FIJI = self.FIJI or os.environ.get('FIJI')
        iminfo = get_image_info(self.image_in)
        self.elsize = self.elsize or iminfo['elsize_zyxc'][:3]
        self.set_tilefile()
        self._set_inputstem('shading', 'apply')
        self._set_outputpath()

        self._parsets = {
            'prep': {
                'fpar': self._FPAR_NAMES + ('tilefile',),
                'ppar': (),
                'spar': ('n_workers',),
                },
            'load': {
                'fpar': self._FPAR_NAMES + ('FIJI', 'tilefile', 'inputstem'),
                'ppar': ('elsize',),
                'spar': ('n_workers', 'stitch_steps', 'channels'),
                },
            'calc': {
                'fpar': self._FPAR_NAMES + ('FIJI',),
                'ppar': ('downsample', 'r', 'max_shift', 'max_displacement',
                         'relative_shift', 'absolute_shift'),
                'spar': ('n_workers', 'stitch_steps'),
                },
            'fuse': {
                'fpar': self._FPAR_NAMES + ('FIJI',),
                'ppar': ('channel_ref', 'z_shift'),
                'spar': ('n_workers', 'stitch_steps', 'channels'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES + ('outputpath',),
                'ppar': (),
                'spar': ('n_workers', 'channels'),
                },
            }

        self._partable = {}

    def prep(self, **kwargs):
        """Write tile position offsets."""
        step = 'prep'
        self.stitch_steps = self._stepmap[step]
        self.estimate(step=step, **kwargs)

    def load(self, **kwargs):
        """Define dataset and load tile positions."""
        step = 'load'
        self.stitch_steps = self._stepmap[step]
        self.estimate(step=step, **kwargs)

    def calc(self, **kwargs):
        """Calculate the stitch."""
        step = 'calc'
        self.stitch_steps = self._stepmap[step]
        channels = list(self.channels)
        self.channels = [self.channel_ref]
        self.estimate(step=step, **kwargs)
        self.channels = list(channels)

    def fuse(self, **kwargs):
        """Fuse the tiles."""
        step = 'fuse'
        self.stitch_steps = self._stepmap[step]
        self.estimate(step=step, **kwargs)

    def postprocess(self, **kwargs):
        """Merge the channels in a symlinked file."""
        step = 'postprocess'
        self.stitch_steps = self._stepmap[step]
        self.n_workers = 1
        self.estimate(step=step, **kwargs)

    def estimate(self, step='', **kwargs):
        """Stitch dataset.

        # channels=[],
        # FIJI='',
        # inputstem='',
        # tilefile='',
        # channel_ref=0,
        # z_shift=0,
        # elsize=[],
        # downsample=[1, 2, 2],
        # r=[0.7, 1.0],
        # max_shift=[0, 0, 0],
        # max_displacement=0,
        # relative_shift=2.5,
        # absolute_shift=3.5,
        # ):
        """

        self.set_parameters(step, kwargs)
        arglist = self._get_arglist(['channels'])
        self.set_n_workers(len(arglist))
        pars = self.dump_parameters(step=self.step)
        self._logstep(pars)

        if self.step == 'prep':
            self.write_stack_offsets()
            return
        elif self.step == 'postprocess':
            self._link_channels()
            return

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            pool.starmap(self._estimate_channel, arglist)

    def _estimate_channel(self, channel):
        """Stitch dataset with fiji BigStitcher."""

        basename = self.format_()
        outstem = os.path.join(self.directory, basename)

        macro_path = os.path.realpath(__file__).replace('.py', '.ijm')

        # Inputs <inputstem><ch><stack>.tif
        # [TODO: generalize ch and stack]
        suffix_ch = 'C*' if channel < 0 else self._suffix_formats['c'].format(channel)
        suffix_st = self.format_([suffix_ch, 'S*'])

        if os.path.isdir(self.inputstem):
            tif_path = os.path.join(self.inputstem, '{}.tif'.format(suffix_st))
        else:
            tif_path = '{}_{}.tif'.format(self.inputstem, suffix_st)

        # Output
        suffix_ch = '' if channel < 0 else self._suffix_formats['c'].format(channel)
        basename = self.format_([self.dataset, self.suffix, suffix_ch])

        for stitch_step in self.stitch_steps:

            if stitch_step == 6:
                self._adapt_xml(channel)
                basename = self.format_([self.dataset, self.suffix, suffix_ch])

            args = [stitch_step, channel, tif_path, self.directory, basename, self.suffix, self.tilefile]
            args += self.elsize + self.downsample + self.r + self.max_shift
            args += [self.max_displacement, self.relative_shift, self.absolute_shift]
            args = self.format_([str(arg) for arg in args], delimiter=' ')
            cmdlist = [self.FIJI, '--headless', '--console', '-macro', macro_path, args]
            subprocess.call(cmdlist)

    def _get_arglist(self, parallelized_pars):

        arglist = super()._get_arglist(parallelized_pars)
        if self.step=='calc':
            arglist = [(self.channel_ref,)]

        return arglist

    def set_tilefile(self, suffix_tl='tiles', ext='.conf'):
        """Generate a filepath for a BigStitcher tile configuration file."""

        # TODO: option to provide INfilepath as argument
        if not self.tilefile:
            basename = self.format_([self.dataset, self.suffix, suffix_tl])
            filename =  '{}{}'.format(basename, ext)
            self.tilefile = os.path.join(self.directory, filename)

    def write_stack_offsets(self):
        """Write tile offsets to file."""

        # TODO: option to provide OUTfilepath as argument
        iminfo = get_image_info(self.image_in)

        if not self.tilefile:
            filestem, _ = os.path.splitext(self.image_in)
            self.tilefile = '{}_tiles.conf'.format(filestem)

        # entry per channel X tile
        vo = np.tile(iminfo['stack_offsets'], [iminfo['nchannels'], 1])
        vo[:, 0] = list(range(0, vo.shape[0]))
        np.savetxt(self.tilefile, vo,
                   fmt='%d;;(%10.5f, %10.5f, %10.5f)',
                   header='dim=3', comments='')

    def _adapt_xml(self, channel, setups=[], replace=True):
        """Copy reference xml, replace filename and add affine mat for zshift."""

        def create_ViewTransform_element(name='Transform', affine='1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'):
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

        ch_suf = self._suffix_formats['c'].format(self.channel_ref)
        basename = self.format_([self.dataset, self.suffix, ch_suf, 'stacks'])
        xml_ref = os.path.join(self.directory, '{}.xml'.format(basename))
        ch_suf = self._suffix_formats['c'].format(channel)
        basename = self.format_([self.dataset, self.suffix, ch_suf, 'stacks'])
        xml_path = os.path.join(self.directory, '{}.xml'.format(basename))

        shutil.copy2(xml_path, '{}.orig'.format(xml_path))
        if channel != self.channel_ref:
            shutil.copy2(xml_ref, xml_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        if replace:
            h5 = root.find('SequenceDescription').find('ImageLoader').find('hdf5')
            h5.text = '{}.h5'.format(basename)

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

        tree.write(xml_path, encoding='utf-8', xml_declaration=True)

    def _link_channels(self, inputstem='', channel_pat='', ext='.bdv'):
        """Merge the channels in a symlinked file."""

        # TODO: remove workaround: => directly to bdv
        base, ext = os.path.splitext(self.outputpath)
        outputpath = base + '.h5'

        inputstem = os.path.join(self.directory, self.format_())
        channel_pat = self._suffix_formats['c'].format(0).replace('0', '?')
        imarisfiles.aggregate_bigstitcher(outputpath,
            inputstem, channel_pat=channel_pat, ext='.h5')

        os.rename(outputpath, base + ext)

        # TODO: cat xmls

    def _set_outputpath(self, basename='stitched', ext='.bdv'):
        """Set the path to module output."""

        basename = self.format_() or basename
        outputfile = '{}{}'.format(basename, ext)
        self.outputpath = os.path.join(self.directory, outputfile)
        # self.outputpath = os.path.relpath(self.outputpath, start=self.directory)


if __name__ == "__main__":
    main(sys.argv[1:])
