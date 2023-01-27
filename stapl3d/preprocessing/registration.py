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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from scipy.special import expit

from skimage.io import imread, imsave
from skimage.filters import gaussian, median, threshold_otsu
from skimage.transform import resize
from skimage.morphology import disk

import SimpleITK as sitk  # NOTE: this needs to have SimpleElastix

from stapl3d import parse_args, Stapl3r, Image, transpose_props

logger = logging.getLogger(__name__)


def main(argv):
    """Co-acquisition image registration."""

    steps = ['estimate', 'apply', 'postprocess']
    args = parse_args('registration_coacq', steps, *argv)

    registrat3r = Registrat3r_CoAcq(
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
    """Image coregistration.
    
    A base class for coregistration in STAPL3D.
    Refer to Registrat3r_LSD and Registrat3r_COACQ for specific applications.
    """

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'registration'

        super(Registrat3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector = {}

        self._parallelization = {}

        self._parameter_sets = {}

        self._parameter_table = {}

        default_attr = {}
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._images = []
        self._labels = []

    def _init_paths(self):

        self._paths = {}

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

    def convert_pointsfile(self,
        fpath_points_in, fpath_image_in,
        padding=[0, 0, 0], fpath_points_out='',
        ):
        """Convert between ITK points formats: world <=> voxel."""

        pointstypes = {
            'point': {'dtype': float, 'format': '%d'},
            'index': {'dtype': int  , 'format': '%f'},
        }

        # Read points.
        with open(fpath_points_in) as f:
            pointstype = f.readline().strip()
        datatype = pointstypes[pointstype]['dtype']
        points_in = np.loadtxt(
            fpath_points_in,
            dtype=datatype,
            delimiter=' ',
            skiprows=3,
            )

        # Get image parameters.
        im = Image(fpath_image_in)
        im.load()
        offset = im.ims_load_extent('Min')[::-1]
        elsize = im.ims_load_elsize()[:3][::-1]
        im.close()

        # Convert points.
        if pointstype == 'point':
            points = points_in - np.array(offset)
            points /= np.array(elsize)
            points += np.array(padding)
        elif pointstype == 'index':
            points = points_in - np.array(padding)
            points *= np.array(elsize)
            points += np.array(offset)

        pointstype = 'index' if pointstype == 'point' else 'point'
        datatype = pointstypes[pointstype]['dtype']
        points = points.astype(datatype)

        if fpath_points_out:
           self.write_pointsfile(fpath_points_out, points, pointstype)

        return points

    def write_pointsfile(self, filepath, points, pointstype):
        """Write ITK pointsfile to disk."""

        pointstypes = {
            'point': {'dtype': float, 'format': '%f'},
            'index': {'dtype': int  , 'format': '%d'},
        }

        formatstring = pointstypes[pointstype]['format']
        with open(filepath, 'w') as f:
            f.write(f'{pointstype}\n{len(points)}\n\n')
            b = '\n'.join(' '.join(formatstring %x for x in y) for y in points)
            f.write(b)

    def read_image(self, filepath, slc={},
                   rl=0, channel=0, timepoint=0,
                   transform={},
                   ):
        """Read a 3D volume."""

        im = Image(filepath, reslev=rl, permission='r')
        im.load()

        tp = timepoint
        ch = channel
        pad = transform['pad']

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

        itk_props = self.im_to_itk_props(im, **transform)

        return im, data, mask, itk_props

    def im_to_itk_props(self, im,
                        theta={'z': 0, 'y': 0, 'x': 0},
                        pad={'z': (0, 0), 'y': (0, 0), 'x':(0, 0)},
                        translate={'z': 0, 'y': 0, 'x': 0},
                        ):
        """Convert STAPL3D Image class attributes to ITK form."""

        elsize_xyz = [im.elsize[im.axlab.index(al)] for al in 'xyz']
        spacing = np.array(elsize_xyz, dtype='float')

        # TODO: make sure to get the order right in the rotation
        rotation_moving = self.axis_angles_to_matrix(**theta)
        direction = rotation_moving[:3, :3].ravel()

        origin = np.array([-pad[al][0] for al in 'xyz']) * spacing

        # FIXME!!: tie the translation to the rotation in the right way
        extent = np.array([im.dims[im.axlab.index(al)] for al in 'xyz']) * spacing
        if theta['y'] == 0:
            origin = np.array([0, 0, 0])
        elif theta['y'] == np.pi / 2:
            origin = np.array([extent[2], 0, 0])
        elif theta['y'] == np.pi:
            origin = np.array([extent[0], 0, extent[2]])
        elif theta['y'] == -np.pi / 2:
            origin = np.array([extent[2], 0, extent[0]])

        origin += np.array([translate[al] for al in 'xyz'])

        origin = origin.astype('float')

        return {'spacing': spacing, 'origin': origin , 'direction': direction}

    def axis_angles_to_matrix(self, x=0, y=0, z=0, tol=1e-12):
        """Convert axis-angle rpresentation to rotation matrix."""

        # TODO: make sure to get the order right in the rotation
        Rz, Ry, Rx = np.eye(4), np.eye(4), np.eye(4)

        Rz[:3, :3] = np.array([
                [np.cos(z), -np.sin(z), 0],
                [np.sin(z),  np.cos(z), 0],
                [        0,          0, 1],
            ])
        Ry[:3, :3] = np.array([
                [ np.cos(y), 0, np.sin(y)],
                [         0, 1,         0],
                [-np.sin(y), 0, np.cos(y)],
                ])
        Rx[:3, :3] = np.array([
                [1,         0,          0],
                [0, np.cos(x), -np.sin(x)],
                [0, np.sin(x),  np.cos(x)],
                ])

        R = Rz @ Ry @ Rx

        R[abs(R) < tol] = 0

        return R

    def write_3d_as_nii(self, outpath, data, itk_props={}):
        """Write a 3D volume to nifti format."""

        default_props = {
            'spacing': [1, 1, 1],
            'origin': (0, 0, 0),
            'direction': [1, 0, 0, 0, 1, 0, 0, 0, 1],
            'dtype': 'float32',
            }
        itk_props ={**default_props, **itk_props}
        itk_im = self.data_to_itk(data,
            itk_props['spacing'],
            itk_props['origin'],
            itk_props['direction'],
            itk_props['dtype'],
            )
        sitk.WriteImage(itk_im, outpath)

    def data_to_itk(self, data, spacing=[1, 1, 1], origin=(0, 0, 0),
                    direction=[1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='float32',
                    ):
        """Numpy array (zyx) to ITK image (xyz)."""

        itk_im = sitk.GetImageFromArray(data.astype(dtype))
        itk_im.SetSpacing(np.array(spacing, dtype='float'))
        itk_im.SetOrigin(np.array(origin, dtype='float'))
        itk_im.SetDirection(np.array(direction, dtype='float'))

        return itk_im

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

        return elastixImageFilter

    def _write_filter(self, parmap, IF, filepaths):
        """Write filter result and parameters to file."""

        if filepaths['elastix']:
            sitk.WriteParameterFile(parmap, filepaths['elastix'])

        if filepaths['transformix']:
            tpMap = IF.GetTransformParameterMap()
            #tpMap[0]['FinalBSplineInterpolationOrder'] = '1',
            #tpMap[0]['ResultImagePixelType'] = "uint16",
            sitk.WriteParameterFile(tpMap[0], filepaths['transformix'])

        result = IF.GetResultImage()

        if filepaths['result']:
            sitk.WriteImage(result, filepaths['result'])

        return result

    def _get_parmap(self, parpath):
        """Get a default or specific parameter map."""

        if parpath in ['rigid', 'affine', 'bspline']:
            parmap = sitk.GetDefaultParameterMap(parpath)
        else:
            parmap = sitk.ReadParameterFile(parpath)

        return parmap


class Registrat3r_LSD(Registrat3r):
    """mLSR3D to Live image coregistration."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'registration_lsd'

        super(Registrat3r_LSD, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'prepare': self.prepare,
            'estimate': self.estimate,
            'apply': self.apply,
            'postprocess': self.postprocess,
            })

        self._parallelization.update({
            'prepare': [],  # ['_datasets'],
            'estimate': [],
            'apply': ['channels', 'timepoints'],
            'postprocess': [],
            })

        self._parameter_sets.update({
            'prepare': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            'apply': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'channels', 'timepoints'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        self._parameter_table.update({
            })

        prep_dict = {
            'convert_points': False,
            'image': {
                'channel': 0,
                'timepoint': -1,
                'transform': {
                    'pad': {'z': (0, 0), 'y': (0, 0), 'x':(0, 0)},
                    'theta': {'z': 0, 'y': 0, 'x': 0},
                    'translate': {'z': 0, 'y': 0, 'x': 0},
                    },
                },
            'unwobble': {},
            'modulate_z': {},
            }
        default_attr = {
            'prep_live': {k: v for k, v in prep_dict.items()},
            'prep_mLSR3D': {k: v for k, v in prep_dict.items()},
            'registration_steps': {'rigid': {}, 'deformable': {}},
            'channels': [],
            'timepoints': [],
            '_suffixes': {'live': 'fixed', 'mLSR3D': 'moving'},
#            '_fixed_image': 'live',
#            '_datasets': ['live', 'mLSR3D'],
            '_registration_step': '',
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths_lsd()

        self._init_log()

        self._images = []
        self._labels = []

    def _init_paths_lsd(self):

        regpat = os.path.join('reg', 'ch{c:02d}_tp{t:03d}')

        self._paths.update({
            'prepare': {
                'inputs': {
                    'data': '',
                    'live': '',
                    'mLSR3D': '',
                    'points_live': '',
                    'points_mLSR3D': '',
                    },
                'outputs': {
                    'filestem': '',
                    'fixed': '',
                    'moving': '',
                    'mask_fixed': '',
                    'mask_moving': '',
                    'points_fixed': '',
                    'points_moving': '',
                    },
                },
            'estimate': {
                'inputs': {
                    'filestem': '',
                    'fixed': '',
                    'moving': '',
                    'mask_fixed': '',
                    'mask_moving': '',
                    'points_fixed': '',
                    'points_moving': '',
                    'parameters_rigid': '',
                    'parameters_deformable': '',
                    },
                'outputs': {
                    'rigid': '',
                    'deformable': '',
                    'elastix_rigid': '',
                    'elastix_deformable': '',
                    'transformix_rigid': '',
                    'transformix_deformable': '',
                    },
                },
            'apply': {
                'inputs': {
                    'live': '',
                    'mLSR3D': '',
                    'filestem': '',
                    'parpath': '',
                    },
                'outputs': {
                    'filestem': '',
                    'regdir': regpat,
                    'append_to_ims': '',
                    },
            },
            'postprocess': {
                'inputs': {
                    'regdir': regpat,
                    },
                'outputs': {
                    'h5': '',
                    'ims': '',
                    },
                },
        })

        for step in self._fun_selector.keys():
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs')
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs')

        self.inputpaths['prepare']['data'] = self.inputpaths['prepare']['mLSR3D']
        self.inputpaths['apply']['data'] = self.inputpaths['prepare']['mLSR3D']
        # FIXME: this is a temp workaround for poor implementation of _get_arglist

    def prepare(self, **kwargs):
        """Prepare datasets for mLSR3D to Live registration."""

        _ = self._prep_step('prepare', kwargs)
        arglist = (('live',), ('mLSR3D',))  # TODO: HPC parallization
        self._set_n_workers(len(arglist))
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._prepare_dataset, arglist)

    def _prepare_dataset(self, dset):
        """Prepare dataset for mLSR3D to Live registration.

        Optional steps:
        1. Convert landmark file to voxel coordinates.
        2. Unwobble dataset.
        3. Modulate intensities over Z.
        """

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        pars = {'live': self.prep_live, 'mLSR3D': self.prep_mLSR3D}[dset]

        filepath_dset = inputs[dset]
        pointspath_um =     inputs[f'points_{dset}'] or filepath_dset.replace('.ims', '.txt')

        suffix = self._suffixes[dset]
        outstem =         outputs['filestem']         or os.path.splitext(inputs['live'])[0]
        points_path_vx  = outputs[f'points_{suffix}'] or f'{outstem}_{suffix}.txt'
        nifti_path_data = outputs[suffix]             or f'{outstem}_{suffix}.nii.gz'
        nifti_path_mask = outputs[f'mask_{suffix}']   or f'{outstem}_{suffix}_mask.nii.gz'

        im, data, mask, itk = self.read_image(filepath_dset, **pars['image'])

        # 1. Convert landmark files (from world to voxel coordinates).
        if pars['convert_pointsfile']:
            pad = pars['image']['transform']['pad']
            pad_lower = [pad[al][0] for al in 'xyz']
            _ = self.convert_pointsfile(
                pointspath_um,
                filepath_dset,
                pad_lower,
                points_path_vx,
                )

        # 2. Unwobble.
        if pars['unwobble']:

            # Generate mask.
            tissue_mask = self.unwobble_mask(
                im, data, pars['image'],
                **pars['unwobble']['tissue_mask'],
                )

            if 'map_z_distance' in pars['unwobble'].keys():
                # Calculate distance to top/bottom.
                distance = self.unwobble_distance(
                    im, data, pars['image'],
                    tissue_mask,
                    pars['unwobble']['direction'],
                    pars['unwobble']['map_z_distance']['filters'],
                    )

            if 'shift_data' in pars['unwobble'].keys():
                # Shift data columns by distance in z.
                data = self.unwobble_data(
                    im, data, pars['image'],
                    distance,
                    pars['unwobble']['direction'],
                    )

            if 'shift_points' in pars['unwobble'].keys():
                # Shift data colums by distance in z.
                self.unwobble_points(
                    points_path_vx,
                    distance,
                    pars['unwobble']['direction'],
                    points_path_vx,
                    )

        # 3. Modulate intensities over Z.
        if pars['modulate_z']:
            data = self.modulate_z(
                im, data,
                **pars['modulate_z'],
                )

        self.write_3d_as_nii(nifti_path_data, data, itk)
        self.write_3d_as_nii(nifti_path_mask, mask, itk)

    def unwobble_mask(
        self,
        im, data, impars,
        read_mask=False,
        reslev=4,
        sigma={'z': 5, 'y': 20, 'x':20},
        threshold=0,
        ):
        """"""

        # TODO: to outputpaths
        filepath_dset = im.path
        data_path_lr = filepath_dset.replace('.ims', f'_rl{reslev}.nii.gz')
        filt_path_lr = filepath_dset.replace('.ims', f'_rl{reslev}_smooth.nii.gz')
        mask_path_lr = filepath_dset.replace('.ims', f'_rl{reslev}_tissuemask.nii.gz')

        if read_mask:

            # Read wobblemask.
            im_lr = Image(mask_path_lr, permission='r')
            im_lr.load()
            tissue_mask = im_lr.slice_dataset()
            tissue_mask = tissue_mask.transpose()  # nii-xyz to zyx
            im_lr.close()

            return tissue_mask

        # Write nifti at low resolution for generating a tissue mask.
        im_lr, data_lr, _, itk_lr = self.read_image(
            filepath_dset, rl=reslev, **impars,
            )
        self.write_3d_as_nii(data_path_lr, data_lr, itk_lr)

        # Smooth volume.
        elsize = dict(zip(im_lr.axlab, im_lr.elsize))
        sigma_vx = [max(1, int(sigma[al] / es))
                    for al, es in elsize.items() if al in 'zyx']
        data_smooth = gaussian(data_lr, sigma=sigma_vx, preserve_range=True)
        self.write_3d_as_nii(filt_path_lr, data_smooth, itk_lr)

        # Threshold smoothed volume
        if not threshold:
            threshold = threshold_otsu(data_smooth)
        tissue_mask = data_smooth > threshold
        self.write_3d_as_nii(mask_path_lr, tissue_mask.astype('uint8'), itk_lr)

        return tissue_mask

    def unwobble_distance(
        self,
        im, data, impars,
        tissue_mask,
        direction='bottom',
        filters={'gaussian': 10},
        ):
        """Calculate the distance in z from mask to the volume top or bottom."""

        filepath_dset = im.path
        filepath_tif = filepath_dset.replace('.ims', f'_distance_{direction}.tif')

        dims_yx = [im.dims[im.axlab.index(al)] for al in 'yx']

        if direction == 'top':
            tissue_mask = np.flip(tissue_mask, axis=0)

        distance = np.argmax(tissue_mask, axis=0)

        for filtertype, filterpar in filters.items():
            if filtertype == 'gaussian':
                distance = gaussian(distance.astype(float), filterpar)
            if filtertype == 'median':
                distance = median(distance, footprint=disk(filterpar))

        distance = resize(distance, dims_yx, preserve_range=True)

        if filepath_tif:
            filepath_png = filepath_tif.replace('.tif', '.png')
            imsave(filepath_tif, distance.astype('int'))
            imsave(filepath_png, distance.astype('int'))

        return distance.astype('int')

    def unwobble_data(
        self,
        im, data, impars,
        distance,
        direction='bottom',
        ):
        """Shift each z-column by a distance."""

        axlab = im.axlab
        dims = im.dims

        data_shifted = np.zeros_like(data)

        for x in range(data.shape[axlab.index('x')]):
            for y in range(data.shape[axlab.index('y')]):

                if direction == 'top':
                    z_dim = dims[axlab.index('z')]  # NOTE: why not from shape?
                    d = data[:z_dim - distance[y][x], y, x]
                    data_shifted[distance[y][x]:, y, x] = d

                elif direction == 'bottom':
                    d = data[distance[y][x]:, y, x]
                    data_shifted[:d.shape[0], y, x] = d

        return data_shifted

    def unwobble_points(self, pointsfile_in, distance, direction, pointsfile_out):
        """Shift each point by a distance in z."""

        points_in = np.loadtxt(pointsfile_in, dtype='int', delimiter=' ', skiprows=3)

        for pt in points_in:
            if direction == 'top':
                pt[2] += distance[pt[1], pt[0]]
            elif direction == 'bottom':
                pt[2] -= distance[pt[1], pt[0]]

        self.write_pointsfile(pointsfile_out, points_in, 'index')

    def modulate_z(self, im, data,
                   slope=5, width=30, reverse=False,
                   offset=0, plot_curve=False,
                   ):
        """Modulate the intensities in the volume with a logistic over Z."""

        # Define logistic function.
        x = np.linspace(slope, -slope, width)
        y = expit(x)

        # Place it at offset in a vector of len(z_dim), padded with 1's.
        z_dim = data.shape[im.axlab.index('z')]
        z_fac = np.zeros(z_dim)
        z_fac[:y.shape[0]] = y
        z_fac = np.roll(z_fac, offset)
        z_fac[:offset] = 1

        # Flip it.
        if reverse:
            z_fac = z_fac[::-1]

        # Plot it.
        if plot_curve:
            plt.plot(list(range(z_dim)), z_fac)
            plt.grid()
            plt.xlabel('slices')
            plt.title('simulated attenuation')
            plt.show()

        # Multiply with profile.
        tileshape = [data.shape[im.axlab.index(al)] for al in 'xy'] + [1]
        z = np.tile(z_fac, tileshape).transpose()

        return np.multiply(data, z)

    def estimate(self, **kwargs):
        """mLSR3D to Live registration."""

        _ = self._prep_step('estimate', kwargs)
        self._register()

    def _register(self, prev_step=''):
        """Register images."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        filepaths = self._get_registration_filepaths(inputs, outputs)

        for curr_step, overrides in self.registration_steps.items():
            self.run_registration(
                filepaths,
                curr_step,
                not prev_step,
                prev_step,
                overrides,
                )
            prev_step = curr_step

    def _get_registration_filepaths(self, inputs, outputs):
        """Gather all filepaths used in the regsitration proper.
        
        Entries specified in parameterfile or defaulting to {filestem}_....<ext>
        """

        filestem = os.path.splitext(self.inputpaths['prepare']['live'])[0]
        filestem = inputs['filestem'] or filestem

        filepaths = {}

        intypes = {'': 'nii.gz', 'mask': 'nii.gz', 'points': 'txt'}
        outtypes = {'elastix': 'txt', 'transformix': 'txt', '': 'nii.gz'}

        # Set paths to input files, masks and landmarks for fixed and moving.
        for intype, ext in intypes.items():
            for _, suffix in self._suffixes.items():
                key = self.format_([intype, suffix])
                filename = "_".join([filestem, suffix])
                filepaths[key] = inputs[key] or f'{filename}.{ext}'
                """
                # FIXME: null spec does not function. => None was already replace by '' in self._merge_paths
                if inputs[key] is None:
                    filepaths[key] = ''
                elif not inputs[key]:
                    filepaths[key] = f'{filename}.{ext}'
                else:
                    filepaths[key] = inputs[key]
                """

        for step, _ in self.registration_steps.items():

            # Set paths to input parameters for each step.
            key = f'parameters_{step}'
            filename = "_".join([filestem, key])
            filepaths[key] = inputs[key] or f'{filename}.txt'

            # Set paths to output parameters and nifti's for each step.
            for outtype, ext in outtypes.items():
                key = self.format_([outtype, step])
                filename = "_".join([filestem, key])
                filepaths[key] = outputs[key] or f'{filename}.{ext}'
                """
                # FIXME: null spec does not function. => None was already replace by '' in self._merge_paths
                if outputs[key] is None:
                    filepaths[key] = ''
                elif not outputs[key]:
                    filepaths[key] = f'{filename}.{ext}'
                else:
                    filepaths[key] = outputs[key]
                """

        return filepaths

    def run_registration(self, filepaths, curr_step, uselandmarks=False,
                         init_suffix='', parmap_overrides={}):
        """Register images."""

        # Read the images.
        fixed = sitk.ReadImage(filepaths['fixed'])
        moving = sitk.ReadImage(filepaths['moving'])

        # Get a basic ImageFilter object.
        IF = self._get_filter()

        # Read and adapt the parameter map.
        key = f'parameters_{curr_step}'
        parmap = self._get_parmap(filepaths[key])
        for k, v in parmap_overrides.items():
            if isinstance(v, list):
                parmap[k] = tuple([str(e) for e in v])
            else:
                parmap[k] = str(v),

        # Set the initial tranform on concatenated transforms.
        if init_suffix:
            key = f'transformix_{init_suffix}'
            IF.SetInitialTransformParameterFileName(filepaths[key])
            parmap['InitialTransformParametersFileName'] = filepaths[key],
            # FIXME This may reset at: tpMap = IF.GetTransformParameterMap()???!!!

        # Set landmark files.
        if uselandmarks:
            IF.SetFixedPointSetFileName(filepaths['points_fixed'])
            IF.SetMovingPointSetFileName(filepaths['points_moving'])

        # Read and set the masks.
        if filepaths['mask_fixed']:
            if isinstance(filepaths['mask_fixed'], str):
                fixed_mask = sitk.ReadImage(filepaths['mask_fixed'])
                fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
            IF.SetFixedMask(fixed_mask)
        if filepaths['mask_moving']:
            if isinstance(filepaths['mask_moving'], str):
                moving_mask = sitk.ReadImage(filepaths['mask_moving'])
                moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
            IF.SetMovingMask(moving_mask)

        # Run registration
        IF = self._run_filter(IF, fixed, moving, parmap)

        # Write the results.
        outpaths = {
            'result': filepaths[curr_step],
            'elastix': filepaths[f'elastix_{curr_step}'],
            'transformix': filepaths[f'transformix_{curr_step}'],
        }
        result = self._write_filter(parmap, IF, outpaths)

        return IF, parmap, result

    def apply(self, **kwargs):
        """Apply mLSR3D to Live registration parameters."""

        arglist = self._prep_step('apply', kwargs)
        # NOTE: ITK (transformix) is already multithreaded => n_workers = 1
        # self.n_threads = 1  # min(self.tasks, multiprocessing.cpu_count())
        self._n_workers = 1

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._apply_channel_timepoint, arglist)

    def _apply_channel_timepoint(self, ch=0, tp=0):
        """Apply mLSR3D to Live registration parameters to 3D dataset."""

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs, reps={'c': ch, 't': tp})
        os.makedirs(outputs['regdir'], exist_ok=True)

        #filestem = inputs['filestem']
        fpath_moving = inputs['mLSR3D'] or self.inputpaths['prepare']['mLSR3D']
        fpath_fixed = inputs['live'] or self.inputpaths['prepare']['live']
        filestem = inputs['filestem'] or os.path.splitext(fpath_fixed)[0]
        suffix = self._registration_step or list(self.registration_steps.keys())[-1]
        parpath = "TransformParameters.0.txt"  # f'{filestem}_transformix_{suffix}.txt
        # FIXME: write this file to {filestem}_transformix_{}.txt

        _, data, _, itk_props = self.read_image(
            fpath_moving,
            **self.prep_mLSR3D['image'],
            )

        self._apply_with_binary(outputs['regdir'], parpath, data, itk_props)
        # self._apply_with_python(outputs['regdir'], parpath, data, itk_props)  # FIXME

        print(f'channel={ch}, timepoint={tp} done')

    def _apply_with_binary(self, regdir, parpath, data, itk_props):

        # Convert 3D input volume to nifti for transformix binary.
        fpath_moving_nii = os.path.join(regdir, 'tmp.nii.gz')
        self.write_3d_as_nii(fpath_moving_nii, data, itk_props)

        # Run transformix.
        cmdlist = [
            'transformix',
            '-in', fpath_moving_nii,
            '-out', regdir,
            '-tp', parpath,
            #'-threads', f'{self.n_threads:d}',
        ]
        subprocess.call(cmdlist)

    def _apply_with_python(self, regdir, parpath, data, itk_props):

        moving = self.data_to_itk(data, **itk_props)

        tpMap = sitk.ReadParameterFile(parpath)
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(tpMap)
        transformixImageFilter.SetMovingImage(moving)
        transformixImageFilter.Execute()
        data = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

        fpath_result = os.path.join(regdir, 'result.nii.gz')
        self.write_3d_as_nii(fpath_result, data, itk_props)

    def postprocess(self, **kwargs):
        """Gather outputs of mLSR3D to Live registration."""

        arglist = self._prep_step('postprocess', kwargs)

        outputs = self._prep_paths(self.outputs)

        fpath_fixed = self.inputpaths['prepare']['live']  # FIXME: get it from the right location
        fpath_moving = self.inputpaths['prepare']['mLSR3D']  # FIXME: get it from the right location

        filestem = fpath_fixed.replace('.ims', '')
        output_h5 = outputs['h5'] or f'{filestem}_padded.h5/reg'
        output_ims = outputs['ims'] or f'{filestem}_reg.ims'

        # Get the number of channels and timepoints in the moving image.
        im = Image(fpath_moving)
        im.load()
        nchannels, ntimepoints = im.dims[3], im.dims[4]
        im.close()

        channels = self.channels or list(range(nchannels))
        timepoints = self.timepoints or list(range(ntimepoints))

        datas = []
        for tp in timepoints:
            data_tp = []
            for ch in channels:

                # Get the path to the single-channel result nifti
                inputs = self._prep_paths(self.inputs, reps={'c': ch, 't': tp})
                fpath_result_nii = os.path.join(inputs['regdir'], 'result.nii')
                # FIXME: this is for binary transformix

                # Load result from disk for concatenated volume.
                im = Image(fpath_result_nii, permission='r')
                im.load()
                data = np.clip(im.slice_dataset().transpose(), 0, 65535)
                # FIXME: assuming uint16, xyz
                im.close()

                data_tp.append(data)

                # Remove intermediates.
                #os.remove(fpath_moving_nii)
                #shutil.rmtree(inputs['regdir'])

            data_tp = np.stack(data_tp, axis=3)
            datas.append(data_tp)

        datas = np.stack(datas, axis=4)  # zyxct

        if output_h5:
            self._write_h5(outputs['h5'], fpath_fixed, datas)
        if output_ims:
            self._write_ims(outputs['ims'], fpath_fixed, datas)

    def _write_h5(self, filepath_out, filepath_ref, datas, axislabels_out='czyx'):
        """Write registration results to hdf5."""

        # Get output props
        im = Image(filepath_ref, permission='r')
        im.load()
        props = im.get_props()

        # zyxct to czyx  # TODO: other options
        if 't' not in axislabels_out:
            props = im.squeeze_props(props, 4)
        props['shape'] = datas.shape[:len(axislabels_out)]
        props = transpose_props(props, axislabels_out)

        data = np.moveaxis(datas[:, :, :, :, 0], 3, 0)

        props['slices'] = None
        im.close()

        # Write to hdf5 file.
        mo = Image(filepath_out, permission='r+', **props)
        mo.create()
        mo.write(data)
        mo.close()

    def _write_ims(self, filepath_out, filepath_ims, datas):
        """Write registration results to Imaris."""

        from stapl3d import imarisfiles

        # Create empty ref and copy it to outfile.
        suffix_ref = '_ref'
        imarisfiles.create_ref(filepath_ims, suffix_ref)
        filepath_ref = filepath_ims.replace('.ims', f'{suffix_ref}.ims')
        shutil.copy2(filepath_ref, filepath_out)

        mo = Image(filepath_out, permission='r+')
        mo.load()

        # First write the final-timepoint channels of live imaging
        im = Image(filepath_ims, permission='r')
        im.load()
        im.slices[4] = slice(im.dims[4] - 1, im.dims[4])
        for ch in range(im.dims[3]):
            im.slices[3] = slice(ch, ch + 1)
            if ch != 0:
                mo.create()
            mo.write(im.slice_dataset())

        # Add the coregistration results.
        pad_fixed = self.prep_mLSR3D['image']['transform']['pad']
        pad_width = {al: pad_fixed[al] for al in 'zyx'}.values()
        for ch_reg in range(datas.shape[3]):
            mo.create()
            data = unpad(datas[:, :, :, ch_reg, 0], pad_width).astype('uint16')
            mo.write(data)
        mo.close()

        # Set vizualization attributes.
        for ch in range(im.dims[3]):
            ch_dict = imarisfiles.ims_get_attr(filepath_ims, ch)
            imarisfiles.ims_set_attr(filepath_out, ch, ch_dict)

        ch_dict = imarisfiles.ims_get_attr(filepath_ims, ch=self.prep_live['image']['channel'])
        for ch_reg in range(datas.shape[3]):
            imarisfiles.ims_set_attr(filepath_out, ch + ch_reg, ch_dict)

    def load_ims(self, filepath):
        # Load final-timepoint Imaris file for props and appending channels.
        # NOTE: extract in Imaris from live-imaging file (Edit => Crop Time)
        mo = Image(filepath, permission='r+')
        mo.load()
        ch_offset = mo.dims[3]
        props = mo.get_props2()
        props = mo.squeeze_props(props, 4)  # squeeze props for czyx output
        mo.close()
        return ch_offset, props, mo

    def view(self, inputs, channel_axis=None):

        import napari

        viewer = napari.Viewer()
        for inputfile in inputs:
            im = Image(inputfile)
            im.load()
            data = im.slice_dataset()
            im.close
            viewer.add_image(data, channel_axis=channel_axis)

class Registrat3r_CoAcq(Registrat3r):
    """Co-acquisition image registration."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'registration_coacq'

        super(Registrat3r_CoAcq, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'estimate': self.estimate,
            'apply': self.apply,
            'postprocess': self.postprocess,
            })

        self._parallelization.update({
            'estimate': ['filepaths'],
            'apply': ['filepaths'],
            'postprocess': [],
            })

        self._parameter_sets.update({
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
            })

        self._parameter_table.update({
            })

        default_attr = {
            'filepat': '*_LR.czi',
            'LR_suffix': 'LR',
            'HR_suffix': 'HR',
            'channel': 0,
            'timepoint': 0,
            'filepaths': [],
            'centrepoint': {},
            'centrepoints': {},
            'margin': {'z': 0, 'y': 20, 'x': 20},
            'tasks': 1,
            'methods': {'affine': 'affine'},
            'target_voxelsize': {},
            'volumes': [],
            '_empty_slice_volume': '',
            '_empty_slice_threshold': 1000,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths_coacq()

        self._init_log()

        self._images = ['raw_nucl']
        self._labels = ['label_cell']

    def _init_paths_coacq(self):

        self.set_filepaths()

        ext = os.path.splitext(os.path.split(self.filepaths[0])[1])[1]
        lr = '{f}' + f'{self.LR_suffix}{ext}'
        hr = '{f}' + f'{self.HR_suffix}{ext}'

        self._paths.update({
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
        })

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

        tf = {
            'theta': {'z': 0, 'y': 0, 'x': 0},
            'pad': {'z': [0, 0], 'y': [0, 0], 'x': [0, 0]},
            'translate': {'z': 0, 'y': 0, 'x': 0},
        }
        _, data, _, itk_props = self.read_image(
            inputs['highres'],
            channel=ch, timepoint=tp, transform=tf,
            )  # FIXME: pad has to exist in transform ... make a default
        fixed = self.data_to_itk(data, **itk_props)

        slc, pad = self._slc_lowres(inputs)

        tf = {
            'theta': {'z': 0, 'y': 0, 'x': 0},
            'pad': pad,
            'translate': {'z': 0, 'y': 0, 'x': 0},
        }
        _, data, _, itk_props = self.read_image(
            inputs['lowres'],
            channel=ch, timepoint=tp, slc=slc, transform=tf,
            )
        moving = self.data_to_itk(data, **itk_props)

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

        _ = self._run_filter(elastixImageFilter, fixed, moving, parmap)

        return parmap, elastixImageFilter

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
        lr_elsize, lr_shape = self._get_shapes(inputs['lowres'])
        hr_elsize, hr_shape = self._get_shapes(inputs['highres'])
        tgt_elsize = {**lr_elsize, **self.target_voxelsize}
        tgt_shape = {d: int( (hr_shape[d] * hr_elsize[d]) / tgt_elsize[d] )
                     for d in 'xyz'}

        # Reorder volumes to process _empty_slice_volume first.
        # # confused here: skipping for now
        # FIXME: volumes are lists in the parfiles, but dicts here
        a = self.volumes.pop(self._empty_slice_volume, {})
        b = {self._empty_slice_volume: a} if a else {}
        volumes = {**b, **self.volumes}
        #volumes = {k, v for k, v in self.volumes.items()}

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

        hr_elsize, hr_shape = self._get_shapes(inputs['highres'])
        lr_elsize, lr_shape = self._get_shapes(inputs['lowres'])

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

    def _get_shapes(self, filepath):
        """Return zyx voxel and matrix size."""

        im = Image(filepath)
        im.load()
        elsize = {d: im.elsize[im.axlab.index(d)] for d in 'zyx'}
        shape = {d: im.dims[im.axlab.index(d)] for d in 'zyx'}
        im.close()

        return elsize, shape


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

    if parpath in ['rigid', 'affine', 'bspline']:
        parmap = sitk.GetDefaultParameterMap(parpath)
    else:
        parmap = sitk.ReadParameterFile(parpath)

    if init_suffix:
        elastixImageFilter.SetInitialTransformParameterFileName(f"{filestem}_transformix_{init_suffix}.txt")
        parmap['InitialTransformParametersFileName'] = f"{filestem}_transformix_{init_suffix}.txt",

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
