#!/usr/bin/env python

"""Co-acquisition image registration.

"""

import os
import sys
import argparse
import logging
import pickle
import shutil
import multiprocessing

import yaml

from glob import glob

import numpy as np
import SimpleITK as sitk  # NOTE: this needs to have SimpleElastix

from stapl3d import parse_args, Stapl3r, Image, format_, wmeMPI
from stapl3d import Image

logger = logging.getLogger(__name__)


def main(argv):
    """Co-acquisition image registration."""

    steps = ['estimate', 'apply', 'postprocess']
    args = parse_args('registration', steps, *argv)

    registrat3r = Registrat3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        registrator._fun_selector[step]()


class Registrat3r(Stapl3r):
    """Co-acquisition image registration."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        super(Registrat3r, self).__init__(
            image_in, parameter_file,
            module_id='registration',
            **kwargs,
            )

        self._fun_selector = {
            'estimate': self.estimate,
            'apply': self.apply,
            'postprocess': self.postprocess,
            }

        self._parallelization = {
            'estimate': ['filepaths'],
            'apply': ['filepaths'],
            'postprocess': [],
            }

        self._parameter_sets = {
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'filepaths'),
                },
            'apply': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'filepaths'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            }

        self._parameter_table = {
            }

        default_attr = {
            'filepat': '*_LR.czi',
            'LR_suffix': 'LR',
            'HR_suffix': 'HR',
            'channel': 0,
            'timepoint': 0,
            'filepaths': [],
            'centrepoint': {},
            'margin': {'y': 20, 'x': 20},
            'tasks': 1,
            'method': 'affine',
            'target_voxelsize': {},
            'volumes': [],
            '_empty_slice_volume': '',
            '_empty_slice_threshold': 1000,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._images = ['raw_nucl']
        self._labels = ['label_cell']

    def _init_paths(self):

        self.set_filepaths()

        ext = os.path.splitext(os.path.split(self.filepaths[0])[1])[1]
        lr = '{f}' + f'{self.LR_suffix}{ext}'
        hr = '{f}' + f'{self.HR_suffix}{ext}'

        self._paths = {
            'estimate': {
                'inputs': {
                    'lowres': lr,
                    'highres': hr,
                    },
                'outputs': {
                    'elastix': '{f}_elastix.txt',
                    'transformix': '{f}_transformix.txt',
                    },
                },
            'apply': {
                'inputs': {
                    'lowres': lr,
                    'highres': hr,
                    'moving': '',
                    'transformix': '{f}_transformix.txt',
                    },
                'outputs': {
                    'reg': '{f}.h5',
                    },
            },
            'postprocess': {
                'inputs': {
                    },
                'outputs': {
                    },
                },
        }

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def estimate(self, **kwargs):
        """Co-acquisition image registration."""

        arglist = self._prep_step('estimate', kwargs)
        # NOTE: ITK is already multithreaded => n_workers = 1
        self.n_threads = min(self.tasks, multiprocessing.cpu_count())
        self._n_workers = 1

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_stack, arglist)

    def _estimate_stack(self, filepath):

        filestem = os.path.basename(filepath).split(self.LR_suffix)[0]

        inputs = self._prep_paths(self.inputs, reps={'f': filestem})
        outputs = self._prep_paths(self.outputs, reps={'f': filestem})

        ch = self.channel
        tp = self.timepoint

        fixed = load_itk_image(inputs['highres'], ch=ch, tp=tp)
        slc = self._slc_lowres(inputs)
        moving = load_itk_image(inputs['lowres'], ch=ch, tp=tp, slc=slc)

        parmap, eif = self._elastix_register(fixed, moving, self.method)

        sitk.WriteParameterFile(parmap, outputs['elastix'])

        tpMap = eif.GetTransformParameterMap()
        tpMap[0]['FinalBSplineInterpolationOrder'] = '1',
        tpMap[0]['ResultImagePixelType'] = "uint16",
        sitk.WriteParameterFile(tpMap[0], outputs['transformix'])

    def _elastix_register(self, fixed, moving, parpath='affine'):

        if parpath in ['rigid', 'affine', 'bspline']:
            parmap = sitk.GetDefaultParameterMap(parpath)
        else:
            parmap = sitk.ReadParameterFile(parpath)

        parmap['AutomaticTransformInitialization'] = "true",
        parmap['AutomaticTransformInitializationMethod'] = "GeometricalCenter",

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)
        elastixImageFilter.SetParameterMap(parmap)
        elastixImageFilter.Execute()

        return parmap, elastixImageFilter

    def apply(self, **kwargs):
        """Co-acquisition image registration."""

        arglist = self._prep_step('apply', kwargs)
        # NOTE: ITK is already multithreaded => n_workers = 1
        #self.n_threads = min(self.tasks, multiprocessing.cpu_count())
        #self._n_workers = 1
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._apply_transform, arglist)

    def _apply_transform(self, filepath):

        filestem = os.path.basename(filepath).split(self.LR_suffix)[0]

        inputs = self._prep_paths(self.inputs, reps={'f': filestem})
        outputs = self._prep_paths(self.outputs, reps={'f': filestem})

        # Target voxelsize and shape
        lr_elsize, lr_shape = get_shapes(inputs['lowres'])
        hr_elsize, hr_shape = get_shapes(inputs['highres'])
        tgt_elsize = {**lr_elsize, **self.target_voxelsize}
        tgt_shape = {d: int( (hr_shape[d] * hr_elsize[d]) / tgt_elsize[d] )
                     for d in 'xyz'}

        # Read transform.
        tpMap = sitk.ReadParameterFile(inputs['transformix'])
        tpMap['Size'] = tuple([str(tgt_shape[d]) for d in 'xyz'])
        tpMap['Spacing'] = tuple([str(tgt_elsize[d]) for d in 'xyz'])
        affine = tpMap['TransformParameters']
        unit = ('1', '0', '0', '0', '1', '0', '0', '0', '1', '0', '0', '0')

        # Reorder to process _empty_slice_volume first.
        a = self.volumes.pop(self._empty_slice_volume, {})
        b = {self._empty_slice_volume: a} if a else {}
        volumes = {**b, **self.volumes}

        # Perform transformation.
        for ods, vol in volumes.items():

            order = 1 if vol['type'] == 'raw' else 0
            dtype = 'uint16' if vol['type'] == 'raw' else 'uint32'
            tpars = affine if vol['resolution'] == 'lowres' else unit
            slic3 = self._slc_lowres(inputs) if vol['resolution'] == 'lowres' else {}

            if 'moving' in vol.keys():
                inputs = {**inputs, **{'moving': vol['moving']}}
            inputs = self._prep_paths(inputs, reps={'f': filestem})
            moving = get_moving(inputs, vol, slc=slic3)

            tpMap['FinalBSplineInterpolationOrder'] = str(order),
            tpMap['TransformParameters'] = tpars

            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(tpMap)
            transformixImageFilter.SetMovingImage(moving)
            transformixImageFilter.Execute()
            data = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

            # Discard empty region.  # FIXME: assuming zyx
            if self._empty_slice_volume:
                if ods == self._empty_slice_volume:
                    idxs = self._identify_empty_slices(data)
                data = self._remove_empty_slices(data, idxs, axis=0)

            # Write output.  # TODO?: axis order options
            ods = vol['ods'] if 'ods' in vol.keys() else ods
            outpath = '{}/{}'.format(outputs['reg'], ods)
            props = {
                'elsize': [tgt_elsize[d] for d in 'zyx'],
                'chunks': [tgt_shape['z'], 64, 64],
                'shape': data.shape,
                'dtype': dtype,
                'axlab': 'zyx',
            }

            mo = Image(outpath, **props)
            mo.create()
            mo.write(data.astype(dtype))
            mo.close()

    def postprocess(self, **kwargs):
        """Co-acquisition image registration."""

        arglist = self._prep_step('postprocess', kwargs)
        # TODO

    def set_filepaths(self):
        """Set the filepaths by globbing the directory."""

        # directory = os.path.abspath(self.directory)
        directory = os.path.abspath(os.path.dirname(self.image_in))
        self.filepaths = sorted(glob(os.path.join(directory, self.filepat)))

    def _slc_lowres(self, inputs):
        """Get a slice for the lowres cutout matching the highres zstack."""

        hr_elsize, hr_shape = get_shapes(inputs['highres'])
        lr_elsize, lr_shape = get_shapes(inputs['lowres'])

        centrepoint_lr_def = {d: lrs / 2 for d, lrs in lr_shape.items()}
        cp = {**centrepoint_lr_def, **self.centrepoint}

        slc = {}
        dims = {**self.centrepoint, **self.margin}.keys()
        for d in dims:
            halfwidth = (hr_elsize[d] * hr_shape[d]) / 2
            extent = halfwidth / lr_elsize[d] + self.margin[d]
            slc_start = max(0, int(cp[d] - extent))
            slc_end = min(lr_shape[d], int(cp[d] + extent))
            slc[d] = slice(slc_start, slc_end)

        return slc

    def _identify_empty_slices(self, data, ods='raw_nucl'):
        """Find empty slices."""

        nz = np.count_nonzero(np.reshape(data==0, [data.shape[0], -1]), axis=1)
        # FIXME: generalize (automate threshold)
        idxs = np.nonzero(nz > self._empty_slice_threshold)

        return idxs

    def _remove_empty_slices(self, data, idxs=[], axis=0):
        """Discard empty region."""
        if self._empty_slice_volume:
            data = np.delete(data, idxs, axis=axis)
        return data


def load_itk_image(filepath, ch=0, tp=0, slc={}):
    """Read a 3D volume."""

    if '.nii' in filepath:  # NOTE: assuming 3D nifti here
        itk_im = sitk.ReadImage(filepath)
        return itk_im

    if '.czi' in filepath:
        im = Image_from_czi(filepath, stack=0)
    else:
        im = Image(filepath)
        im.load()

    if 'c' in im.axlab:
        im.slices[im.axlab.index('c')] = slice(ch, ch + 1, 1)
    if 't' in im.axlab:
        im.slices[im.axlab.index('t')] = slice(tp, tp + 1, 1)
    for k, v in slc.items():
        im.slices[im.axlab.index(k)] = v
    data = im.slice_dataset()
    if data.dtype == 'bool':
        data = data.astype('uint8')
    itk_im = sitk.GetImageFromArray(data)
    spacing = np.array(im.elsize[:data.ndim][::-1], dtype='float')
    itk_im.SetSpacing(spacing)
    im.close()

    return itk_im


def get_shapes(filepath):
    im = Image(filepath)
    im.load()
    elsize = {d: im.elsize[im.axlab.index(d)] for d in 'zyx'}
    shape = {d: im.dims[im.axlab.index(d)] for d in 'zyx'}
    im.close()
    return elsize, shape


def get_moving(inputs, vol, slc={}):

    if 'moving' in vol.keys():
        movingfile = inputs['moving']
        if 'ids' in vol.keys():
            ids = vol['ids']
            movingfile = f'{movingfile}/{ids}'
    else:
        movingfile = inputs[vol['resolution']]

    ch = vol['channel'] if 'channel' in vol.keys() else 0
    tp = vol['timepoint'] if 'timepoint' in vol.keys() else 0

    moving = load_itk_image(movingfile, ch=ch, tp=tp, slc=slc)

    return moving


def Image_from_czi(filepath, stack=0):
    # TODO: integrate into Image class
    from stapl3d.preprocessing.shading import read_zstack, get_image_info
    iminfo = get_image_info(filepath)
    data = np.transpose(np.squeeze(read_zstack(filepath, stack)), [1,2,3,0])
    props = {
        'axlab': 'zyxc',
        'shape': iminfo['dims_zyxc'],
        'elsize': iminfo['elsize_zyxc'],
        'dtype': iminfo['dtype'],
        }
    im = Image('', **props)
    im.create()
    im.write(data)
    return im
