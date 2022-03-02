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
import subprocess
import tempfile

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
            'centrepoints': {},
            'margin': {'y': 20, 'x': 20},
            'tasks': 1,
            'methods': {'affine': ''},
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

        init_suffix = ''
        for method, parpath in self.methods.items():

            parmap, eif = self._elastix_register(filestem, fixed, moving,
                                                 method, parpath,
                                                 init_suffix)
            tpMap = eif.GetTransformParameterMap()
            tpMap[0]['FinalBSplineInterpolationOrder'] = '1',
            tpMap[0]['ResultImagePixelType'] = "uint16",

            k = 'elastix'
            outpath = outputs[k].replace(f'_{k}.txt', f'_{k}_{method}.txt')
            sitk.WriteParameterFile(parmap, outpath)

            k = 'transformix'
            outpath = outputs[k].replace(f'_{k}.txt', f'_{k}_{method}.txt')
            sitk.WriteParameterFile(tpMap[0], outpath)

            init_suffix = f'{method}'

    def _elastix_register(self, filestem, fixed, moving,
                          method='affine', parpath='',
                          init_suffix='', uselandmarks=False):

        elastixImageFilter = self._get_filter()

        if init_suffix:
            init_path = f"{filestem}_transformix_{init_suffix}.txt"
            elastixImageFilter.SetInitialTransformParameterFileName(init_path)

        if not parpath:  # ['rigid', 'affine', 'bspline']:
            parmap = sitk.GetDefaultParameterMap(method)
        else:
            parmap = sitk.ReadParameterFile(parpath)

        parmap['AutomaticTransformInitialization'] = "true",
        parmap['AutomaticTransformInitializationMethod'] = "GeometricalCenter",

        if uselandmarks:
            elastixImageFilter.SetFixedPointSetFileName(f"{filestem}_fixed.txt")
            elastixImageFilter.SetMovingPointSetFileName(f"{filestem}_moving.txt")

        self._run_filter(elastixImageFilter, fixed, moving, parmap)

        return parmap, elastixImageFilter

    def _get_filter(self):
        """Initiate an empty elastix filter."""

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOn()
        elastixImageFilter.LogToConsoleOn()

        return elastixImageFilter

    def _run_filter(self, elastixImageFilter, fixed, moving, parmap):
        """Execute a filter after setting the images and parameters."""

        elastixImageFilter.SetParameterMap(parmap)
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)
        elastixImageFilter.Execute()

    def _write_filter(self, parmap, IF, filestem, suffix):
        """Write filter result and parameters to file."""

        sitk.WriteParameterFile(parmap, f'{filestem}_elastix_{suffix}.txt')
        transformParameterMap = IF.GetTransformParameterMap()
        #transformParameterMap[0]['FinalBSplineInterpolationOrder'] = '1',
        #transformParameterMap[0]['ResultImagePixelType'] = "uint16",
        tfpath = f'{filestem}_transformix_{suffix}.txt'
        sitk.WriteParameterFile(transformParameterMap[0], tfpath)
        result = IF.GetResultImage()
        sitk.WriteImage(result, f'{filestem}_{suffix}.nii.gz')

        return result

    def apply(self, **kwargs):
        """Co-acquisition image registration."""

        if 'bspline' in self.methods.keys():
            self._reg_dir = tempfile.mkdtemp(prefix='reg_', dir=self.directory)

        arglist = self._prep_step('apply', kwargs)
        # NOTE: ITK is already multithreaded => n_workers = 1
        #self.n_threads = min(self.tasks, multiprocessing.cpu_count())
        #self._n_workers = 1
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._apply_transform, arglist)

        if 'bspline' in self.methods.keys():
            shutil.rmtree(self._reg_dir)

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

        # Reorder volumes to process _empty_slice_volume first.
        a = self.volumes.pop(self._empty_slice_volume, {})
        b = {self._empty_slice_volume: a} if a else {}
        volumes = {**b, **self.volumes}

        # Perform transformation.
        for ods, vol in volumes.items():

            # Read transform.
            k = 'transformix'
            if 'method' in vol.keys():
                method = vol['method']
            elif vol['resolution'] == 'lowres':
                method = list(self.methods.keys())[-1]  # typically deformable
            else:
                method = list(self.methods.keys())[0]  # typically affine
            parpath = inputs[k].replace(f'_{k}.txt', f'_{k}_{method}.txt')
            tpMap = sitk.ReadParameterFile(parpath)

            # Pick dtype
            dtype = 'uint16' if vol['type'] == 'raw' else 'uint32'
            dtype = 'float'  # TODO: only float if deformable reg? TODO: test float64?
            tpMap['ResultImagePixelType'] = dtype,

            # Switch parameters
            transform = tpMap['TransformParameters']
            unit = ('1', '0', '0', '0', '1', '0', '0', '0', '1', '0', '0', '0')
            tpars = transform if vol['resolution'] == 'lowres' else unit
            tpMap['TransformParameters'] = tpars

            # Set size and spacing
            tpMap['Size'] = tuple([str(tgt_shape[d]) for d in 'xyz'])
            tpMap['Spacing'] = tuple([str(tgt_elsize[d]) for d in 'xyz'])

            # Set order
            order = 1 if vol['type'] == 'raw' else 0
            tpMap['FinalBSplineInterpolationOrder'] = str(order),

            # Get image
            slic3 = self._slc_lowres(inputs) if vol['resolution'] == 'lowres' else {}
            if 'moving' in vol.keys():
                inputs = {**inputs, **{'moving': vol['moving']}}
            inputs = self._prep_paths(inputs, reps={'f': filestem})
            moving = get_moving(inputs, vol, slc=slic3)

            if 'bspline' in self.methods.keys():
                ads = ods.replace('/', '-')
                data = self._apply_transform_with_binary(moving, tpMap, f'{filestem}_{ads}')
            else:
                data = self._apply_transform_with_python(moving, tpMap)

            dtype = data.dtype if vol['type'] == 'raw' else 'uint32'

            # Discard empty region.  # FIXME: assuming zyx with axis=0
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

    def _apply_transform_with_python(self, moving, tpMap):

        transformixImageFilter = sitk.TransformixImageFilter()

        transformixImageFilter.SetTransformParameterMap(tpMap)
        transformixImageFilter.SetMovingImage(moving)
        transformixImageFilter.Execute()

        data = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

        return data

    def _apply_transform_with_binary(self, moving, tpMap, chanstem):

        chandir = os.path.join(self._reg_dir, chanstem)
        os.makedirs(chandir, exist_ok=True)

        # Write the adapted parameter file.
        fpath_moving_par = os.path.join(chandir, f'{chanstem}.txt')
        sitk.WriteParameterFile(tpMap, fpath_moving_par)

        # Convert to nifti for transformix input
        fpath_moving_nii = os.path.join(chandir, f'{chanstem}.nii.gz')
        sitk.WriteImage(moving, fpath_moving_nii)

        # Run transformix
        cmdlist = [
            'transformix',
            '-in',  fpath_moving_nii,
            '-out', chandir,
             '-tp', fpath_moving_par,
        ]
        subprocess.call(cmdlist)

        # Read nifti from disk.
        im = Image(os.path.join(chandir, 'result.nii'), permission='r')
        im.load()
        data = im.slice_dataset().transpose()
        im.close()

        print(f'{chanstem} done')

        return data

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

        name = inputs['lowres'].replace(f'{self.LR_suffix}.czi', '')
        if name in self.centrepoints.keys():
            cp = self.centrepoints[name]
        else:
            cp = self.centrepoint

        centrepoint_lr_def = {d: lrs / 2 for d, lrs in lr_shape.items()}
        cp = {**centrepoint_lr_def, **cp}

        slc, slc_f, pad = {}, {}, {}
        dims = {**cp, **self.margin}.keys()
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
    #itk_im = sitk.GetImageFromArray(data.astype('float32'))
    spacing = np.array(im.elsize[:data.ndim][::-1], dtype='float')
    itk_im.SetSpacing(spacing)
    #itk_im.SetOrigin(get_origin_ims(filepath))

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


def get_origin_ims(filepath):
    """Read the origin in world coordinates from an Imarisfile."""

    def att2str(att):
        return ''.join([t.decode('utf-8') for t in att])

    import h5py
    f = h5py.File(filepath, 'r')
    im_info = f['/DataSetInfo/Image']
    origin = [float(att2str(im_info.attrs[f'ExtMin{i}'])) for i in range(3)]
    f.close()

    return origin
