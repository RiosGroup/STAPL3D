#!/usr/bin/env python

"""Stitching helper functions.

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

from stapl3d import (
    get_n_workers,
    get_outputdir,
    get_params,
    get_paths,
    Image,
)

from stapl3d.preprocessing import shading

logger = logging.getLogger(__name__)


def main(argv):
    """Generate a mask that covers the tissue."""

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
        '-p', '--parameter_file',
        required=True,
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
    channels=[],
    FIJI='',
    stitch_steps=[],
    channel_ref=0,
    z_shift=0,
    elsize=[],
    postfix='',
    ):
    """Stitch dataset."""

    step_id = 'stitching'

    stackdir = get_outputdir(image_in, '', '', 'stacks', 'stacks')
    outputdir = get_outputdir(image_in, parameter_file, outputdir, step_id, step_id)

    params = get_params(locals().copy(), parameter_file, step_id)
    subparams = get_params(locals().copy(), parameter_file, step_id, 'submit')

    iminfo = shading.get_image_info(image_in)
    subparams['channels'] = subparams['channels'] or iminfo['channels']
    params['elsize'] = params['elsize'] or iminfo['elsize_zyxc'][:3]

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    fstem = '{}{}'.format(cfg['dataset']['name'], cfg['shading']['params']['postfix'])
    inputstem = os.path.join(stackdir, fstem)
    postfix = cfg['shading']['params']['postfix'] + params['postfix']

    FIJI = FIJI or os.environ.get('FIJI')

    arglist = [
        (
            image_in,
            inputstem,
            ch,
            FIJI,
            params['stitch_steps'],
            params['channel_ref'],
            params['z_shift'],
            params['elsize'],
            postfix,
            outputdir,
        )
        for ch in subparams['channels']]

    n_workers = get_n_workers(subparams['channels'], subparams)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(estimate_channel, arglist)


def estimate_channel(
    image_in,
    inputstem,
    channel,
    FIJI='',
    stitch_steps=[],
    channel_ref=0,
    z_shift=0,
    elsize=[],
    postfix='',
    outputdir='',
    ):

    # Prepare the output.
    step_id = 'stitching'
    postfix = postfix or '_{}'.format(step_id)

    outputdir = get_outputdir(image_in, '', outputdir, step_id, step_id)

    paths = get_paths(image_in)
    datadir, filename = os.path.split(paths['base'])
    dataset, ext = os.path.splitext(filename)

    filestem = '{}{}'.format(dataset, postfix)
    outputstem = os.path.join(outputdir, filestem)
    outputpat = '{}.h5/{}'.format(outputstem, '{}')

    logging.basicConfig(filename='{}.log'.format(outputstem), level=logging.INFO)
    report = {'parameters': locals()}

    if not elsize:
        try:
            iminfo = get_image_info(image_in)
            elsize = iminfo['elsize_zyxc'][:3]
        except:
            print('WARNING: could not determine elsize: please check in/output')
            elsize = [1.0, 1.0, 1.0]

    FIJI = FIJI or os.environ.get('FIJI')

    macro_path = os.path.realpath(__file__).replace('.py', '.ijm')

    for stitch_step in stitch_steps:

        if stitch_step == 6:
            filestem = os.path.join(outputdir, dataset)
            adapt_xml(filestem, channel, channel_ref, z_shift)

        macro_args = '{} {} {} {} {} {} {} {} {}'.format(
            str(stitch_step),
            inputstem,
            outputdir,
            dataset,
            str(channel),
            postfix,
            elsize[0], elsize[1], elsize[2],
            )
        cmdlist = [FIJI, '--headless', '--console', '-macro',
                   "{}".format(macro_path), macro_args]
        subprocess.call(cmdlist)


def adapt_xml(filestem, channel, channel_ref, zshift=0, setups=[], replace=True):

    xml_ref = '{}_ch{:02d}.xml'.format(filestem, channel_ref)
    xml_path = '{}_ch{:02d}.xml'.format(filestem, channel)

    shutil.copy2(xml_path, '{}.orig'.format(xml_path))
    if channel != channel_ref:
        shutil.copy2(xml_ref, xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if replace:
        h5 = root.find('SequenceDescription').find('ImageLoader').find('hdf5')
        h5.text = '{}_ch{:02d}.h5'.format(os.path.basename(filestem), channel)

    if zshift:
        vrs = root.find('ViewRegistrations')
        affine = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 {:03f}'.format(zshift)
        vt_new = create_ViewTransform_element(affine=affine)
        for vr in vrs.findall('ViewRegistration'):
            setup = vr.get('setup')
            if not setups:
                vr.insert(0, vt_new)
            elif int(setup) in setups:
                vr.insert(0, vt_new)

    tree.write(xml_path)


def create_ViewTransform_element(name='Transform', affine='1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'):
    """
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


def write_stack_offsets(image_in, conffile=''):

    iminfo = shading.get_image_info(image_in)

    if not conffile:  # TODO: default to stackdir...
        filestem, _ = os.path.splitext(image_in)
        conffile = '{}_tileoffsets_chxx.conf'.format(filestem)

    # entry per channel X tile
    vo = np.tile(iminfo['stack_offsets'], [iminfo['nchannels'], 1])
    vo[:, 0] = list(range(0, vo.shape[0]))
    np.savetxt(conffile, vo,
               fmt='%d;;(%10.5f, %10.5f, %10.5f)',
               header='dim=3', comments='')


def bdv_load_elsize(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    item = root.find('./SequenceDescription/ViewSetups/ViewSetup/voxelSize/size')
    elsize = [float(e) for e in item.text.split()]


if __name__ == "__main__":
    main(sys.argv[1:])
