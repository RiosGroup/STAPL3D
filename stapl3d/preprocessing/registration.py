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
            'filepat': '*_25x.czi',
            'filepaths': [],
            'tasks': 1,
            'method': 'affine',
            'centrepoint': [],
            '_margins': [20, 20],
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

    def _init_paths(self):

        self.set_filepaths()

        stem = self._build_path()
        #spat = self._build_path(suffixes=[{'s': 'p'}])
        #smat = self._build_path(suffixes=[{'s': '?'}])
        fpat = self._build_path(suffixes=[{'f': 'p'}])

        self._paths = {
            'estimate': {
                'inputs': {
                    'lowres': '{f}_25x_{s}.czi',
                    'highres': '{f}_63x_{s}.czi',
                    },
                'outputs': {
                    'elastix': f'{fpat}_params_elastix.txt',
                    'transformix': f'{fpat}_params_transformix.txt',
                    },
                },
            'apply': {
                'inputs': {
                    },
                'outputs': {
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
        """Co-acquisition image registration.

        """

        arglist = self._prep_step('estimate', kwargs)

        # NOTE: ITK is already multithreaded => n_workers = 1
        # TODO: check if the case for SimpleElastix
        self.n_threads = min(self.tasks, multiprocessing.cpu_count())
        self._n_workers = 1

        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_stack, arglist)

    def _estimate_stack(self, filepath):

        #filestem = os.path.splitext(os.path.basename(filepath))[0]
        filestem = os.path.basename(filepath).split('_S')[0]
        print(filestem)
        inputs = self._prep_paths(self.inputs, reps={'f': filestem})
        print(inputs)
        outputs = self._prep_paths(self.outputs, reps={'f': filestem})
        print(outputs)

        fixed = load_itk_image(inputs['highres'])
        slc_yx = find_lowres_slice_yx(inputs['lowres'], inputs['highres'], self.centrepoint, self._margins)
        moving = load_itk_image(inputs['lowres'], slc=slc_yx)

        parmap, eif = self._elastix_register(fixed, moving, self.method)

        sitk.WriteParameterFile(parmap, outputs['elastix'])

        transformParameterMap = eif.GetTransformParameterMap()
        transformParameterMap[0]['FinalBSplineInterpolationOrder'] = '1',
        transformParameterMap[0]['ResultImagePixelType'] = "uint16",
        sitk.WriteParameterFile(transformParameterMap[0], outputs['transformix'])

        # fixme: these inputs are the czi, we need segmentation h5's for labels
        #fname = '{}{}.h5/{}'.format(fstem, stack_pf, ids)
        #movingfile = os.path.join(blockdir, fname)
        vols = [
            {'type': 'raw',   'filepath': inputs['lowres'],  'ids': 'chan/ch00',           'ods': 'raw_nucl'},
            {'type': 'label', 'filepath': inputs['highres'], 'ids': 'segm/labels_curated', 'ods': 'label_cell_curated'},
            ]
        self._elastix_transform(outputs['transformix'], inputs['highres'], vols, slc_xy)

    def _elastix_register(self, fixed, moving, method):

        parmap = sitk.GetDefaultParameterMap(self.method)
        parmap['AutomaticTransformInitialization'] = "true",
        parmap['AutomaticTransformInitializationMethod'] = "GeometricalCenter",
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)
        elastixImageFilter.SetParameterMap(parmap)
        elastixImageFilter.Execute()

        return parmap, elastixImageFilter

    def _elastix_transform(self, transformfile, fixedfile, vols, slc_xy):

        transformParameterMap = sitk.ReadParameterFile(transformfile)
        affine = transformParameterMap['TransformParameters']

        im = Image(fixedfile)
        im.load()
        props = im.get_props()
        props['slices'] = None
        im.close()

        tgt_elsize = [im.elsize[im.axlab.index(dim)] for dim in 'xyz']
        tgt_shape =  [im.dims[im.axlab.index(dim)] for dim in 'xyz']
        if self.tgt_elsize:  # xyz  # self.tgt_elsize = [0.3321, 0.3321, 1.2048]
            elsize = [self.tgt_elsize[dim] for dim in 'xyz']
            tgt_shape = [int(ssh/(te/se)) for se, te, ssh in zip(tgt_elsize, self.tgt_elsize, tgt_shape)]

        transformParameterMap['Size'] = tuple([str(s) for s in tgt_shape])
        transformParameterMap['Spacing'] = tuple([str(s) for s in tgt_elsize])

        for vol in vols:

            ids, ods, movingfile, voltype = vol['ids'], vol['ods'], vols['filepath'], vols['type']
            if voltype == 'raw':  # lowres data in 25x space
                tp = affine
                dtype = 'uint16'  # TODO: generalize
                order = 1
                slc = slc_yx
            elif voltype == 'label':  # highres labels in 63x space
                tp = ('1', '0', '0', '0', '1', '0', '0', '0', '1', '0', '0', '0')
                dtype = 'uint32'
                order = 0
                slc = {}

            moving = load_itk_image(movingfile, slc=slc)
            transformParameterMap['FinalBSplineInterpolationOrder'] = str(order),
            transformParameterMap['TransformParameters'] = tp

            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(transformParameterMap)
            transformixImageFilter.SetMovingImage(moving)
            transformixImageFilter.Execute()
            data = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

            # filter empty slices???
            if ods == 'raw_nucl' or ods == 'chan/ch00':
                nz = np.count_nonzero(np.reshape(data==0, [data.shape[0], -1]), axis=1)
                idxs = np.nonzero(nz > 1000)
            data = np.delete(data, idxs, axis=0)

            props['elsize'] = tgt_elsize[::-1]
            props['shape'] = data.shape
            props['chunks'] = [tgt_shape[2], 64, 64]
            props['dtype'] = dtype

            fname = '{}_zstack{}.h5/{}'.format(dataset, stack_idx, ods)
            fpath = os.path.join(projectdir, fname)
            mo = Image(fpath, **props)
            mo.create()
            mo.write(data.astype(dtype))
            mo.close()


    def apply(self, **kwargs):
        """Co-acquisition image registration.

        """

        arglist = self._prep_step('apply', kwargs)

    def postprocess(self, **kwargs):
        """Co-acquisition image registration.

        """

        arglist = self._prep_step('postprocess', kwargs)


    # def elastix_apply(self, fixed, moving, method):
    #
    #     # write all transformed channels to imaris file
    #     transformixImageFilter = sitk.TransformixImageFilter()
    #     transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    #
    #     fname_out_fixed = '{}{}'.format(fstem_fixed, stack_pf)
    #     fname_out_moving = '{}'.format(fstem_moving)
    #     fname_out = "{}_{}_{}.ims".format(fname_out_fixed, fname_out_moving, method)
    #     fpath_out = os.path.join(ddir_out, fname_out)
    #
    #     shutil.copy2(fpath_fixed, fpath_out)
    #     mo = Image(fpath_out, dtype='uint16', permission='r+')
    #     mo.load(load_data=False)
    #     #for ch in range(0, im_moving.dims[im_moving.axlab.index('c')]):
    #     for ch in range(0, nchannels):
    #         print(ch)
    #         slc_c = {'c': slice(ch, ch + 1)}
    #         mo.slices[mo.axlab.index('c')] = slc_c['c']
    #         moving = load_itk_image(fpath_moving, slc={**slc_yx, **slc_c})
    #         transformixImageFilter.SetMovingImage(moving)
    #         transformixImageFilter.Execute()
    #         data = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())
    #         mo.write(data.astype('uint16'))
    #     mo.close()
    #     #im_moving.close()

    def set_filepaths(self):
        """Set the filepaths by globbing the directory."""

        # directory = os.path.abspath(self.directory)
        directory = os.path.abspath(os.path.dirname(self.image_in))
        self.filepaths = sorted(glob(os.path.join(directory, self.filepat)))







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
    itk_im = sitk.GetImageFromArray(data)
    spacing = np.array(im.elsize[:data.ndim][::-1], dtype='float')
    itk_im.SetSpacing(spacing)
    im.close()

    return itk_im


def find_lowres_slice_yx(input_lr, input_hr, lr_centrepoint_yx=[], margins=[20, 20]):
    """Get a slice for the lowres cutout matching the highres zstack."""

    def get_shapes(filepath):
        if 'czi' in filepath:
            from stapl3d.preprocessing.shading import get_image_info
            iminfo = get_image_info(filepath)
            elsize = iminfo['elsize_zyxc'][1:3]
            shape = [iminfo['ncols'], iminfo['nrows']]
        else:
            im = Image(filepath)
            im.load()
            elsize = [im.elsize[im.axlab.index(d)] for d in 'yx']
            shape = [im.dims[im.axlab.index(d)] for d in 'yx']
            im.close()
        return elsize, shape

    hr_elsize, hr_shape = get_shapes(input_hr)
    lr_elsize, lr_shape = get_shapes(input_lr)

    lr_blocksize_yx = [(hr_shape[i] * hr_elsize[i]) / lr_elsize[i] + 2 * margins[i] for i in [0, 1]]

    if not lr_centrepoint_yx:
        lr_centrepoint_yx = [lr_shape[0] / 2, lr_shape[1] / 2]

    slc_yx = {k: slice(int(cp - bs / 2), int(cp + bs / 2))
              for k, cp, bs in zip('yx', lr_centrepoint_yx, lr_blocksize_yx)}

    return slc_yx


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
