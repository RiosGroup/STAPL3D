#!/usr/bin/env python

"""Co-acquisition image registration.

    FIXME: only works in conda base environment (with stapl3d installed, via conda.pth).
"""

""" Example LSD coregistration pipeline.

# Imports and function definitions
import os
import numpy as np
from stapl3d import Image
from stapl3d.preprocessing import registration

# Path definitions.

projdir = 'F:\\mkleinnijenhuis\\1.projects\\LSD'
dataset = '20210705_Exp29_OBS_DMG_Insert4_Day15'
datadir = os.path.join(projdir, dataset)
filestem = os.path.join(datadir, dataset)
os.chdir(datadir)

prefix_fixed = '20210705_Exp29_OBS_DMG_Insert4_Day15_live2-Stitched'
fpath_fixed = os.path.join(datadir, f'{prefix_fixed}.ims')
slcs_fixed = {}
ch_fixed = 0  # DMG cells
tp_fixed = -1   # the last timepoint
th_fixed = {'z': 0, 'y': 0, 'x': 0}
pad_fixed = {'z': (20, 20), 'y': (200, 200), 'x':(200, 200)}
translate_fixed = np.array([0, 0, 0])

prefix_moving = '20210804_Exp29_OBS_DMG_Insert4_Day15_LSR3D-Stitched'
fpath_moving = os.path.join(datadir, f'{prefix_moving}.ims')
slcs_moving = {}
ch_moving = 1  # DMG cells
tp_moving = 0  # the only timepoint
th_moving = {'z': 0, 'y': np.pi, 'x': 0}
pad_moving = {'z': (0, 0), 'y': (0, 0), 'x':(0, 0)}
translate_moving = np.array([0, -212, 20]) * np.array([0.332, 0.332, 1.209]) # itk_props_fixed['spacing']

# NOTE: no wobble here

attenuation_parameters = {
    'slope': 5,
    'width': 30,
    'reverse': True,
    'plot_curve': True,
}

im, data, mask = registration.read_image(fpath_fixed, ch=ch_fixed, tp=tp_fixed, slc=slcs_fixed, pad=pad_fixed)
itk_props = registration.im_to_itk_props(im, th_fixed, pad_fixed)
registration.write_3d_as_nii(f'{filestem}_fixed.nii.gz', data, itk_props)
registration.write_3d_as_nii(f'{filestem}_fixed_mask.nii.gz', mask, itk_props)

im, data, mask = registration.read_image(fpath_moving, ch=ch_moving, tp=tp_moving, slc=slcs_moving, pad=pad_moving)
itk_props = registration.im_to_itk_props(im, th_moving, pad_moving)
itk_props['origin'] += translate_moving
# NOTE: no wobble here

data = registration.modulate_z_intensities(im, data, **attenuation_parameters)

registration.write_3d_as_nii(f'{filestem}_moving.nii.gz', data, itk_props)
registration.write_3d_as_nii(f'{filestem}_moving_mask.nii.gz', mask, itk_props)

registration.register(filestem, ['rigid', 'deformable'])

append_to_ims = os.path.join(datadir, f'{prefix_fixed}_final-tp.ims')
registration.apply_registration(fpath_fixed, fpath_moving, filestem, 'deformable', th_moving, pad_moving, translate_moving, append_to_ims)
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

from scipy.special import expit

from skimage.io import imread, imsave
from skimage.filters import gaussian, median
from skimage.transform import resize
from skimage.morphology import disk

import SimpleITK as sitk  # NOTE: this needs to have SimpleElastix

from stapl3d import parse_args, Stapl3r, Image, transpose_props

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
        registrat3r._fun_selector[step]()


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
        slc, pad = self._slc_lowres(inputs)
        moving = load_itk_image(inputs['lowres'], ch=ch, tp=tp, slc=slc, padding=pad)

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
            if not os.path.exist(init_suffix):
                init_suffix = f"{filestem}_transformix_{init_suffix}.txt"
            elastixImageFilter.SetInitialTransformParameterFileName(init_suffix)

#        if not parpath:
        if parpath in ['rigid', 'affine', 'bspline']:
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
            if 'transform' in vol.keys():
                if vol['transform'] == 'unit':
                    tpars = unit

            tpMap['TransformParameters'] = tpars

            # Set size and spacing
            tpMap['Size'] = tuple([str(tgt_shape[d]) for d in 'xyz'])
            tpMap['Spacing'] = tuple([str(tgt_elsize[d]) for d in 'xyz'])

            # Set order
            order = 1 if vol['type'] == 'raw' else 0
            tpMap['FinalBSplineInterpolationOrder'] = str(order),

            # Get image
            if vol['resolution'] == 'lowres':
                slic3, pad = self._slc_lowres(inputs)
            else:
                slic3, pad = {}, {}
            if 'moving' in vol.keys():
                inputs = {**inputs, **{'moving': vol['moving']}}
            inputs = self._prep_paths(inputs, reps={'f': filestem})
            moving = get_moving(inputs, vol, slc=slic3, padding=pad)

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

            slc_start_f = int(cp[d] - extent)
            slc_end_f = int(cp[d] + extent)

            pad_lower = max(0, -slc_start_f)
            pad_upper = max(0, -lr_shape[d] + slc_end_f)
            pad[d] = [pad_lower, pad_upper]

        return slc, pad

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


def load_itk_image(filepath, rl=0, ch=0, tp=0, slc={}, padding={}, flip=''):
    """Read a 3D volume and return in ITK format."""

    if '.nii' in filepath:  # NOTE: assuming 3D nifti here
        itk_im = sitk.ReadImage(filepath)
        return itk_im

    im, data = read_image(filepath, rl, ch, tp, slc)

    pad_width = tuple(padding[al] for al in im.axlab if al in padding.keys())
    data = pad_and_flip(data, pad_width, [im.axlab.index(al) for al in flip])

    itk_im = data_to_itk(data, elsize=im.elsize, origin=get_origin_ims(filepath), dtype='float32')

    return itk_im


def read_image(filepath, rl=0, ch=0, tp=0, slc={}, pad={}):
    """Read a 3D volume."""

    im = Image(filepath, reslev=rl)
    im.load()

    if 't' in im.axlab:
        if tp == -1:
            tp = im.dims[im.axlab.index('t')] - 1
        im.slices[im.axlab.index('t')] = slice(tp, tp + 1, 1)

    if 'c' in im.axlab and ch != -1:
        im.slices[im.axlab.index('c')] = slice(ch, ch + 1, 1)

    for k, v in slc.items():  # spatial slices
        im.slices[im.axlab.index(k)] = v

    data = im.slice_dataset()
    if data.dtype == 'bool':
        data = data.astype('uint8')

    if ch == -1:
        data = np.mean(data, axis=im.axlab.index('c'))

    mask = np.ones(data.shape, dtype='uint8')

    pad_width = tuple(pad[al] for al in im.axlab if al in pad.keys())
    data = np.pad(data, pad_width)
    mask = np.pad(mask, pad_width)

    im.close()

    return im, data, mask


def padded_mask(shape, pad):

    mask = np.ones(shape, dtype='uint8')
    pad_width = tuple(pad[al] for al in im.axlab if al in pad.keys())

    return np.pad(mask, pad_width)


def pad_and_flip(data, pad_width=((0, 0), (0, 0), (0, 0)), flip=[]):

    # NOTE: in the end we want this handled by an initial transform setting
    # NB: origin changed through flip and pad; is origin used??
    # alignment should be reasonable: keeping origin to default 0, 0, 0 for now

    data = np.pad(data, pad_width)

    for axis in flip:
        data = np.flip(data, axis)

    return data


def data_to_itk_old(data, elsize=[1, 1, 1], origin=(0, 0, 0), dtype='float32'):
    """Numpy array (zyx) to ITK image (xyz)."""

    itk_im = sitk.GetImageFromArray(data.astype(dtype))
    itk_im.SetSpacing(np.array(elsize[:data.ndim][::-1], dtype='float'))
    itk_im.SetOrigin(origin)

    return itk_im


def write_3d_as_nii_old(outpath, data, elsize=[1, 1, 1], origin=(0, 0, 0), dtype='float32'):
    itk_im = data_to_itk(data, elsize, origin, dtype)
    sitk.WriteImage(itk_im, outpath)


def get_rot_from_axis_angles(th_x, th_y, th_z, tol=0):

    # TODO: make sure to get the order right in the rotation
    Rz, Ry, Rx = np.eye(4), np.eye(4), np.eye(4)

    Rz[:3, :3] = np.array([
            [np.cos(th_z), -np.sin(th_z), 0],
            [np.sin(th_z),  np.cos(th_z), 0],
            [0,                        0, 1],
        ])
    Ry[:3, :3] = np.array([
            [np.cos(th_y),  0, np.sin(th_y)],
            [0,             1,            0],
            [-np.sin(th_y), 0, np.cos(th_y)],
            ])
    Rx[:3, :3] = np.array([
            [1,            0,             0],
            [0, np.cos(th_x), -np.sin(th_x)],
            [0, np.sin(th_x),  np.cos(th_x)],
            ])

    R = Rz @ Ry @ Rx

    R[abs(R) < tol] = 0

    return R


def get_affine(th, elsize, translation_vx):
    """"""

    T = np.eye(4)
    for i in range(3):
        T[i, 3] = translation_vx[i] * elsize[i]

    Z = np.eye(4)
    Z[:3, :3] = np.diag(elsize)

    R = get_rot_from_axis_angles(th[0], th[1], th[2], tol=1e-14)

    S = np.eye(4)  # TODO: implement shear??

    return T @ R @ Z @ S


def im_to_itk_props(im, th={'z': 0, 'y': 0, 'x': 0}, pad={'z': (0, 0), 'y': (0, 0), 'x':(0, 0)}):

    elsize_xyz = [im.elsize[im.axlab.index(al)] for al in 'xyz']

    spacing = np.array(elsize_xyz, dtype='float')

    extent = np.array([im.dims[im.axlab.index(al)] for al in 'xyz']) * spacing

    # TODO: make sure to get the order right in the rotation
    rotation_moving = get_rot_from_axis_angles(th['x'], th['y'], th['z'], tol=1e-12)
    direction = rotation_moving[:3, :3].ravel()

    origin = np.array([-pad['x'][0], -pad['y'][0], -pad['z'][0]]) * spacing

    # FIXME!!: tie the translation to the rotation in the right way
    if th['y'] == 0:
        origin = np.array([0, 0, 0])
    elif th['y'] == np.pi / 2:
        origin = np.array([extent[2], 0, 0])
    elif th['y'] == np.pi:
        origin = np.array([extent[0], 0, extent[2]])
    elif th['y'] == -np.pi / 2:
        origin = np.array([extent[2], 0, extent[0]])

    origin = origin.astype('float')

    return {'spacing': spacing, 'origin': origin , 'direction': direction}


def data_to_itk(data, spacing=[1, 1, 1], origin=(0, 0, 0), direction=[1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='float32'):
    """Numpy array (zyx) to ITK image (xyz)."""

    itk_im = sitk.GetImageFromArray(data.astype(dtype))
    itk_im.SetSpacing(np.array(spacing, dtype='float'))
    itk_im.SetOrigin(np.array(origin, dtype='float'))
    itk_im.SetDirection(np.array(direction, dtype='float'))

    return itk_im


def write_3d_as_nii(outpath, data, itk_props={}):

    default_props = {
        'spacing': [1, 1, 1],
        'origin': (0, 0, 0),
        'direction': [1, 0, 0, 0, 1, 0, 0, 0, 1],
        'dtype': 'float32',
        }
    itk_props ={**default_props, **itk_props}
    itk_im = data_to_itk(data,
        itk_props['spacing'],
        itk_props['origin'],
        itk_props['direction'],
        itk_props['dtype'],
        )
    sitk.WriteImage(itk_im, outpath)


def landmarks_to_labels(pts_filepath, shape, elsize, expand=0):

    labels = np.zeros(shape, dtype='uint16')
    pts = np.loadtxt(pts_filepath, delimiter=' ', skiprows=3, dtype='uint16')
    for i, pt in enumerate(pts):
        labels[pt[2], pt[1], pt[0]] = i+1

    if expand:
        from stapl3d.segmentation.segment import expand_labels
        labels = expand_labels(labels, elsize, expand)

    return labels


def write_pointsfile(filepath, points, pointstype):

    pointstypes = {
        'point': {'dtype': float, 'format': '%f'},
        'index': {'dtype': int  , 'format': '%d'},
    }

    formatstring = pointstypes[pointstype]['format']
    with open(filepath, 'w') as f:
        f.write(f'{pointstype}\n{len(points)}\n\n')
        b = '\n'.join(' '.join(formatstring %x for x in y) for y in points)
        f.write(b)


def convert_pointsfile(fpath_points_in, fpath_image_in, padding=[0, 0, 0], fpath_points_out=''):

    pointstypes = {
        'point': {'dtype': float, 'format': '%d'},
        'index': {'dtype': int  , 'format': '%f'},
    }

    # Read points.
    with open(fpath_points_in) as f:
        pointstype = f.readline().strip()
    datatype = pointstypes[pointstype]['dtype']
    points_in = np.loadtxt(fpath_points_in, dtype=datatype, delimiter=' ', skiprows=3)

    # Get image parameters.
    from stapl3d import Image
    im = Image(fpath_image_in)
    im.load()
    offset = im.ims_load_extent('Min')[::-1]
    elsize = im.ims_load_elsize()[:3][::-1]
    im.close()

    # Convert points.
    if pointstype == 'point':
        points = (points_in - np.array(offset)) / np.array(elsize) + np.array(padding)
    elif pointstype == 'index':
        points = (points_in - np.array(padding)) * np.array(elsize) + np.array(offset)

    pointstype = 'index' if pointstype == 'point' else 'point'
    datatype = pointstypes[pointstype]['dtype']
    points = points.astype(datatype)

    if fpath_points_out:
        write_pointsfile(fpath_points_out, points, pointstype)

    return points


def unpad(data, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return data[tuple(slices)]


def estimate_wobble(
        filepath, resolution_level=2, channel=-1, flip='',
        sigma=[15, 15, 15], tissue_threshold=600, filepath_mask='',
        filters={'gaussian': 10}, filepath_tif=''):

        tissue_mask = wobble_mask(filepath, resolution_level, channel, flip, sigma, tissue_threshold, filepath_mask)
        distance = wobble_distance(tissue_mask, filters, filepath_tif)

        return tissue_mask, distance


def wobble_mask(filepath, resolution_level=2, channel=-1, flip='', sigma=[15, 15, 15], tissue_threshold=600, filepath_out=''):

    im, data = read_image(filepath, resolution_level, channel)
    for axis_label in flip:
        data = np.flip(data, axis=im.axlab.index(axis_label))

    if any(sigma):
        data = gaussian(data, sigma)

    tissue_mask = data > tissue_threshold

    if filepath_out:
        write_3d_as_nii(filepath_out, tissue_mask.astype('uint8'), im.elsize[:3])

    return tissue_mask


def wobble_distance(filepath, filters={'gaussian': 10}, filepath_tif=''):

    im, data = read_image(filepath_mask, resolution_level, channel)
    for axis_label in flip:
        data = np.flip(data, axis=im.axlab.index(axis_label))

    distance = np.argmax(tissue_mask, axis=0)

    for filtertype, filterpar in filters.items():
        if filtertype == 'gaussian':
            distance = gaussian(distance.astype(float), filterpar)
        if filtertype == 'median':
            distance = median(distance, footprint=disk(filterpar))

    dims_yx = [im.dims[im.axlab.index(al)] for al in 'yx']
    distance = resize(distance, dims_yx, preserve_range=True)

    if filepath_tif:
        imsave(filepath_tif, distance.astype('int'))

    return distance


def wobble(data, im, filepath_tif, direction='bottom'):

    dims = im.dims
    axlab = im.axlab
    dims_yx = [im.dims[im.axlab.index(al)] for al in 'yx']
    distance = resize(imread(filepath_tif), dims_yx, preserve_range=True).astype('int')
    # distance = imread(filepath_tif)

    data_shifted = np.zeros_like(data)

    for x in range(data.shape[axlab.index('x')]):
        for y in range(data.shape[axlab.index('y')]):

            if direction == 'top':
                z_dim = dims[axlab.index('z')]
                d = data[:z_dim - distance[y][x], y, x]
                data_shifted[distance[y][x]:, y, x] = d
            elif direction == 'bottom':
                d = data[distance[y][x]:, y, x]
                data_shifted[:d.shape[0], y, x] = d

    return data_shifted


def offset_data(data, offset=0):

    data_offset = np.zeros_like(data)
    data_offset[offset:, :, :] = data[:-offset, :, :]

    return data_offset


def modulate_z_intensities(im, data, slope, width, reverse=False, offset=0, plot_curve=False):

    z_dim = data.shape[im.axlab.index('z')]
    z_fac = np.zeros(z_dim)
    x = np.linspace(slope, -slope, width)
    y = expit(x)

    #yp = y[width/2-z_dim/2:width/2+z_dim/2]
    #yp.shape
    z_fac[:y.shape[0]] = y
    z_fac = np.roll(z_fac, offset)
    z_fac[:offset] = 1

    if reverse:
        z_fac = z_fac[::-1]

    tileshape = [data.shape[im.axlab.index('x')], data.shape[im.axlab.index('y')], 1]
    z = np.tile(z_fac, tileshape).transpose()

    if plot_curve:
        import matplotlib.pyplot as plt
        plt.plot(list(range(z_dim)), z_fac)
        plt.grid()
        plt.xlabel('slices')
        plt.title('simulated attenuation')
        plt.show()

    return np.multiply(data, z)


def get_filter():
    """Initiate an empty elastix filter."""

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.LogToConsoleOn()

    return elastixImageFilter


def run_filter(elastixImageFilter, fixed, moving, parmap):
    """Execute a filter after setting the images and parameters."""

    elastixImageFilter.SetParameterMap(parmap)
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.Execute()

    return elastixImageFilter


def write_filter(parmap, IF, filestem, suffix):
    """Write filter result and parameters to file."""

    sitk.WriteParameterFile(parmap, f'{filestem}_elastix_{suffix}.txt')
    transformParameterMap = IF.GetTransformParameterMap()
    #transformParameterMap[0]['FinalBSplineInterpolationOrder'] = '1',
    #transformParameterMap[0]['ResultImagePixelType'] = "uint16",
    sitk.WriteParameterFile(transformParameterMap[0], f'{filestem}_transformix_{suffix}.txt')
    result = IF.GetResultImage()
    sitk.WriteImage(result, f'{filestem}_{suffix}.nii.gz')

    return result


def register_tmp(filestem, suffix, parpath, uselandmarks=False, init1_suffix=''):
    """Register images."""

    fixed = sitk.ReadImage(f'{filestem}_fixed.nii.gz')
    moving = sitk.ReadImage(f'{filestem}_moving.nii.gz')

    elastixImageFilter = get_filter()

    if init_suffix:
        elastixImageFilter.SetInitialTransformParameterFileName(f"{filestem}_transformix_{init_suffix}.txt")

    if parpath in ['rigid', 'affine', 'bspline']:
        parmap = sitk.GetDefaultParameterMap(parpath)
    else:
        parmap = sitk.ReadParameterFile(parpath)

    elastixImageFilter.SetFixedPointSetFileName(f"{filestem}_fixed.txt")
    elastixImageFilter.SetMovingPointSetFileName(f"{filestem}_moving.txt")

    fixed_mask = sitk.ReadImage(f'{filestem}_fixed_mask.nii.gz')
    fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
    elastixImageFilter.SetFixedMask(fixed_mask)
    moving_mask = sitk.ReadImage(f'{filestem}_moving_mask.nii.gz')
    moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
    elastixImageFilter.SetMovingMask(moving_mask)

    elastixImageFilter = run_filter(elastixImageFilter, fixed, moving, parmap)

    result = write_filter(parmap, elastixImageFilter, filestem, suffix)

    return elastixImageFilter, parmap, result


def register(filestem, steps, prev_step=''):
    """Register images."""

    for curr_step in steps:
        run_registration(
            f'{filestem}_fixed.nii.gz',
            f'{filestem}_moving.nii.gz',
            filestem,
            curr_step,
            f'{filestem}_{curr_step}.txt',
            not prev_step,
            prev_step,
            f'{filestem}_fixed_mask.nii.gz',
            f'{filestem}_moving_mask.nii.gz',
            )
        prev_step = curr_step


def run_registration(fixed, moving, filestem, suffix, parpath, uselandmarks=False, init_suffix='', fixed_mask='', moving_mask=''):
    """Register images."""

    if isinstance(fixed, str):
        fixed = sitk.ReadImage(fixed)
    if isinstance(moving, str):
        moving = sitk.ReadImage(moving)

    elastixImageFilter = get_filter()

    if init_suffix:
        elastixImageFilter.SetInitialTransformParameterFileName(f"{filestem}_transformix_{init_suffix}.txt")

    if parpath in ['rigid', 'affine', 'bspline']:
        parmap = sitk.GetDefaultParameterMap(parpath)
    else:
        parmap = sitk.ReadParameterFile(parpath)

    if uselandmarks:
        elastixImageFilter.SetFixedPointSetFileName(f"{filestem}_fixed.txt")
        elastixImageFilter.SetMovingPointSetFileName(f"{filestem}_moving.txt")

    if fixed_mask:
        if isinstance(fixed_mask, str):
            fixed_mask = sitk.ReadImage(fixed_mask)
            fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
        elastixImageFilter.SetFixedMask(fixed_mask)
    if moving_mask:
        if isinstance(moving_mask, str):
            moving_mask = sitk.ReadImage(moving_mask)
            moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
        elastixImageFilter.SetMovingMask(moving_mask)

    elastixImageFilter = run_filter(elastixImageFilter, fixed, moving, parmap)

    result = write_filter(parmap, elastixImageFilter, filestem, suffix)

    return elastixImageFilter, parmap, result



def apply_registration(fpath_fixed, fpath_moving, filestem, suffix='deformable', th={}, pad={}, translate=[0, 0, 0], append_to_ims='', pad_fixed={}):
    """..."""

    prefix_moving = os.path.splitext(fpath_moving)[0]
    datadir = os.path.dirname(fpath_moving)

    # Get the number of channels and timepoints in the moving image.
    im = Image(fpath_moving)
    im.load()
    nchannels = im.dims[3]
    ntimepoints = im.dims[4]
    im.close()
    nchannels, ntimepoints

    # Load final-timepoint Imaris file for props and appending channels.
    # NOTE: extract in Imaris from live-imaging file (Edit => Crop Time)
    fpath_live = append_to_ims or fpath_fixed
    mo = Image(fpath_live, permission='r+')
    mo.load()
    ch_offset = mo.dims[3]
    props = mo.get_props2()
    props = mo.squeeze_props(props, 4)  # squeeze props for czyx output
    mo.close()

    # Path to the transformix config file
    parpath = f'{filestem}_transformix_{suffix}.txt'

    # Use the transformix binary to generate output
    datas = []
    for ch in range(nchannels):

        for tp in range(ntimepoints):

            # Define output path.
            chanstem = f'{prefix_moving}_ch{ch:02d}_tp{tp:03d}'
            chandir = os.path.join(datadir, 'reg', chanstem)
            os.makedirs(chandir, exist_ok=True)
            fpath_moving_nii = os.path.join(datadir, 'reg', f'{chanstem}.nii.gz')

            # Convert to nifti for transformix input.
            im, data, _ = read_image(fpath_moving, ch=ch, tp=tp, slc={}, pad=pad)
            itk_props = im_to_itk_props(im, th, pad)
            itk_props['origin'] += np.array(translate)
            # NOTE: no wobble here
            write_3d_as_nii(fpath_moving_nii, data, itk_props)

            # Run transformix.
            cmdlist = [
                'transformix',
                '-in',  fpath_moving_nii,
                '-out', chandir,
                 '-tp', parpath,
            ]
            subprocess.call(cmdlist)
            print(f'channel={ch}, timepoint={tp} done')

            # Load result from disk for concatenated volume.
            im = Image(os.path.join(chandir, 'result.nii'), permission='r')
            im.load()
            data = np.clip(im.slice_dataset().transpose(), 0, 65535)
            datas.append(data)
            im.close()

            # Write to new Imaris channel.
            mo.create()
            mo.close()
            mo.load()
            mo.slices[3] = slice(ch_offset + ch, ch_offset + ch + 1)
            mo.write(unpad(data, {al: pad_fixed[al] for al in 'zyx'}.values()).astype('uint16'))
            mo.close()

            # Remove intermediates.
            os.remove(fpath_moving_nii)
            shutil.rmtree(chandir)

    # Write to 4D czyx .h5 file.
    datas = np.stack(datas)
    props['shape'] = props['dims'] = datas.shape
    props = transpose_props(props, 'czyx')
    props['shape'] = props['dims']
    props['slices'] = None
    del props['path']
    del props['permission']
    mo = Image(f'{prefix_moving}_padded.h5/reg', permission='r+', **props)
    mo.create()
    mo.write(datas)
    mo.close()


def get_shapes(filepath):
    im = Image(filepath)
    im.load()
    elsize = {d: im.elsize[im.axlab.index(d)] for d in 'zyx'}
    shape = {d: im.dims[im.axlab.index(d)] for d in 'zyx'}
    im.close()
    return elsize, shape


def get_moving(inputs, vol, slc={}, padding={}):

    if 'moving' in vol.keys():
        movingfile = inputs['moving']
        if 'ids' in vol.keys():
            ids = vol['ids']
            movingfile = f'{movingfile}/{ids}'
    else:
        movingfile = inputs[vol['resolution']]

    ch = vol['channel'] if 'channel' in vol.keys() else 0
    tp = vol['timepoint'] if 'timepoint' in vol.keys() else 0

    moving = load_itk_image(movingfile, ch=ch, tp=tp, slc=slc, padding=padding)

    return moving


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
