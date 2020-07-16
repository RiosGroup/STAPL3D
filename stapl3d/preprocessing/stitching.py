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

import xml.etree.ElementTree as ET

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
    ):
    """Stitch dataset."""

    step_id = 'stitching'


def adapt_xml(filestem, channel, channel_ref, zshift=0, setups=[]):

    xml_ref = '{}_ch{:02d}.xml'.format(filestem, channel_ref)
    xml_path = '{}_ch{:02d}.xml'.format(filestem, channel)
    h5_filename = '{}_ch{:02d}.h5'.format(os.path.basename(filestem), channel)

    shutil.copy2(xml_ref, '{}.orig'.format(xml_path))

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if replacement:
        sd = root.find('SequenceDescription')
        il = sd.find('ImageLoader')
        h5 = il.find('hdf5')
        h5.text = h5_filename

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


if __name__ == "__main__":
    main(sys.argv[1:])
