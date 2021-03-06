#!/usr/bin/env python

"""Calculate regionprops of segments.

"""

import sys
import argparse

# conda install cython
# conda install pytest
# conda install pandas
# pip install ~/workspace/scikit-image/  # scikit-image==0.16.dev0

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import distance

from stapl3d import Image, LabelImage, wmeMPI
from stapl3d.channels import get_bias_field_block

from skimage.measure import regionprops, regionprops_table
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries

from stapl3d.segmentation.segment import extract_segments


def main(argv):
    """Calculate regionprops of segments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '--seg_paths',
        nargs='*',
        help='paths to label volumes (xyz)',
        )
    parser.add_argument(
        '--seg_names',
        nargs='*',
        help='names for (sub)segmentations',
        )
    parser.add_argument(
        '--data_path',
        nargs='*',
        help='paths to data channels',
        )
    parser.add_argument(
        '--data_names',
        nargs='*',
        help='names for channels',
        )
    parser.add_argument(
        '--aux_data_path',
        help='path to auxilliary data file (zyxc)',
        )
    parser.add_argument(
        '--downsample_factors',
        nargs='*',
        type=int,
        default=[],
        help='the downsample factors applied to the aux_data_path image'
        )
    parser.add_argument(
        '--csv_path',
        default='',
        help='path to output csv file',
        )
    parser.add_argument(
        '-s', '--blocksize',
        required=True,
        nargs='*',
        type=int,
        default=[],
        help='size of the datablock'
        )
    parser.add_argument(
        '-m', '--blockmargin',
        nargs='*',
        type=int,
        default=[],
        help='the datablock overlap used'
        )
    parser.add_argument(
        '--blockrange',
        nargs=2,
        type=int,
        default=[],
        help='a range of blocks to process'
        )
    parser.add_argument(
        '--channels',
        nargs='*',
        type=int,
        default=[],
        help='a list of channel indices to extract intensity features for'
        )
    parser.add_argument(
        '-f', '--filter_borderlabels',
        action='store_true',
        help='save intermediate results'
        )
    parser.add_argument(
        '--min_labelsize',
        type=int,
        default=0,
        help='minimum labelsize in voxels',
        )
    parser.add_argument(
        '--split_features',
        action='store_true',
        help='save intermediate results'
        )
    parser.add_argument(
        '--fset_morph',
        default=['label'],
        help='morphology feature set',
        )
    parser.add_argument(
        '--fset_intens',
        default=['mean_intensity'],
        help='intensity feature set',
        )
    parser.add_argument(
        '--fset_addit',
        default=['com_z', 'com_y', 'com_x'],
        help='auxilliary feature set',
        )

    args = parser.parse_args()

    export_regionprops(
        args.seg_paths,
        args.seg_names,
        args.data_path,
        args.data_names,
        args.aux_data_path,
        args.downsample_factors,
        args.csv_path,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.channels,
        args.filter_borderlabels,
        args.min_labelsize,
        args.split_features,
        args.fset_morph,
        args.fset_intens,
        args.fset_addit,
        )


def export_regionprops(
        seg_paths,
        seg_names=['full', 'memb', 'nucl'],
        data_paths=[],
        data_names=[],
        aux_data_path=[],
        downsample_factors=[1, 1, 1],
        outputstem='',
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        channels=[],
        filter_borderlabels=False,
        min_labelsize=0,
        split_features=False,
        fset_morph=['label'],
        fset_intens=['mean_intensity'],
        fset_addit=['com_z', 'com_y', 'com_x'],
        ):

    # load the segments: ['full'] or ['full', 'memb', 'nucl']
    label_ims = {}
    pfs = seg_names[:len(seg_paths)]
    for pf, seg_path in zip(pfs, seg_paths):
        im = LabelImage(seg_path, permission='r')
        im.load(load_data=False)
        label_ims[pf] = im
    comps = label_ims['full'].split_path()

    # prepare parallel processing
    mpi_label = wmeMPI(usempi=False)
    blocksize = blocksize or label_ims['full'].dims
    mpi_label.set_blocks(label_ims['full'], blocksize, blockmargin, blockrange)
    mpi_label.scatter_series()

    # load the data
    data_ims = {}
    mpi_data = wmeMPI(usempi=False)
    for i, data_path in enumerate(data_paths):
        pf = 'im{:02d}'.format(i)
        data = Image(data_path, permission='r')
        data.load(load_data=False)
        ch_idx = data.axlab.index('c')
        # FIXME channels for multiple data_paths
        chs = channels or [ch for ch in range(data.dims[ch_idx])]
        names = [data_names.pop(0) for _ in range(len(chs))]
        data_ims[pf] = {'im': data, 'ch': chs, 'names': names}
        """ TODO
        try:
            mpi_data.blocks = [
                {'id': split_filename(comps['file'])[0]['postfix'],
                 'slices': dset_name2slices(comps['file'], axlab=data.axlab, shape=data.dims),
                 'path': '',},
                ]
        except:
        """
    mpi_data.set_blocks(data, blocksize, blockmargin, blockrange)

    border_labelset = set([])
#    if filter_borderlabels:
#        outstem = outputstem or label_ims['full'].split_path()['base']
#        outstem += '_dataset'
#        border_labelset |= filter_borders(label_ims['full'], outstem)

    dfs = []
    for i in mpi_label.series:
        print('processing block {:03d} with id: {}'.format(i, mpi_label.blocks[i]['id']))
        dfs.append(process_block(
            mpi_label.blocks[i],
            mpi_data.blocks[i],
            label_ims,
            split_features,
            data_ims,
            min_labelsize,
            channels,
            filter_borderlabels,
            fset_morph,
            fset_intens,
            fset_addit,
            border_labelset,
            outputstem,
            aux_data_path,
            downsample_factors,
            )
        )

    return dfs


def process_block(
        block_label,
        block_data,
        label_ims,
        split_features,
        data_ims,
        min_labelsize,
        channels,
        filter_borderlabels=False,
        fset_morph=['label'],
        fset_intens=['mean_intensity'],
        fset_addit=['com_z', 'com_y', 'com_x'],
        border_labelset=set([]),
        outputstem='',
        aux_data_path='',
        downsample_factors=[1, 1, 1],
        ):

    morph, intens, add = get_feature_set(fset_morph, fset_intens, fset_addit)

    all_regions = {}
    for pf, label_im in label_ims.items():
        label_im.slices = block_label['slices'][:3]
        all_regions[pf] = label_im.slice_dataset().astype('int')

    all_data = {}
    for dpf, datadict in data_ims.items():
        data = datadict['im']
        data.slices = block_data['slices']
        for ch, name in zip(datadict['ch'], datadict['names']):
            data.slices[data.axlab.index('c')] = slice(ch, ch + 1, 1)
            ch_data = data.slice_dataset()
            all_data[name] = ch_data

    outstem = outputstem or label_ims['full'].split_path()['base']
    outstem += '_{}'.format(block_label['id'])

    if filter_borderlabels:
        border_labelset |= filter_borders(label_ims['full'], outstem)

    if min_labelsize:
        all_regions = filter_size(all_regions, min_labelsize, outstem)

    for pf, regions in all_regions.items():

        try:
            rpt = regionprops_table(regions, properties=morph)
        except IndexError:
            print('IndexError on MORPH {}: empty labelset'.format(block_label['id']))
            df = get_empty_dataframe(morph, add, intens, channels)
        except ValueError:
            print('ValueError on MORPH {}'.format(block_label['id']))
            df = get_empty_dataframe(morph, add, intens, channels)
        else:

            df = pd.DataFrame(rpt)

            origin = [block_data['slices'][i].start for i in [0, 1, 2]]  # in full dataset voxels
            df = add_features(df, aux_data_path, origin, downsample_factors)

            for cpf, ch_data in all_data.items():
                df_int = get_intensity_df_data(regions, intens, ch_data, cpf)
                df = pd.concat([df, df_int], axis=1)

        outstem = outputstem or label_im.split_path()['base']
        outstem += '_{}'.format(block_label['id'])
        csv_path = "{}_features_{}.csv".format(outstem, pf)
        df.to_csv(csv_path)

    # TODO: close images

    return df


def filter_borders(label_im, outstem):

    labelset = find_border_labels(label_im)

    strpat = 'found {:12d} border labels in {}'
    print(strpat.format(len(labelset), outstem))

    write_labelset(labelset, outstem, pf='borderlabels')

    return labelset


def filter_size(all_regions, min_labelsize, outstem=''):

    pf = 'nucl' if 'nucl' in all_regions.keys() else 'full'
    rp = regionprops(all_regions[pf])
    small_labels = [prop.label for prop in rp if prop.area < min_labelsize]

    strpat = 'found {:12d} small labels in {}'
    print(strpat.format(len(small_labels), outstem))

    write_labelset(set(small_labels), outstem, pf='smalllabels')

    maxlabel = np.amax(all_regions['full'])
    fw = np.zeros(maxlabel + 1, dtype='bool')
    fw[small_labels] = True
    sl_mask = np.array(fw)[all_regions['full']]
    for pf in all_regions.keys():
        all_regions[pf][sl_mask] = 0

    return all_regions


def write_labelset(labelset, outstem, pf):

    ppath = "{}_{}.pickle".format(outstem, pf)
    with open(ppath, 'wb') as f:
        pickle.dump(labelset, f)


def get_nuclearmask(block_label):

    maskfile_compound = False # FIXME: as argument
    if maskfile_compound:
        maskpath = os.path.join(datadir, '{}_bfc_nucl-dapi_mask_sauvola.ims'.format(dataset))
        mask_sauvola_im = MaskImage(maskpath_sauvola, permission='r')
        mask_sauvola_im.load(load_data=False)
        mask_sauvola_im.slices[:3] = block_label['slices'][:3]
        mask_sauvola_im.slices[3] = slice(0, 1, None)

    labelfile_blocks = False # FIXME: as argument
    from stapl3d import MaskImage
    #datadir = '/hpc/pmc_rios/Kidney/190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    datadir = 'G:\\mkleinnijenhuis\\PMCdata\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    if labelfile_blocks:
        block_label['id'] = '02496-03904_12736-14144_00000-00106'
        #block_label['id'] = '03776-05184_12736-14144_00000-00106'
        blockdir = os.path.join(datadir, 'blocks_1280')
        maskpath_sauvola = os.path.join(blockdir, '{}_{}.h5/nucl/dapi_mask_sauvola'.format(dataset, block_label['id']))
        mask_sauvola_im = MaskImage(maskpath_sauvola, permission='r')
        mask_sauvola_im.load(load_data=False)
        maskpath_absmin = os.path.join(blockdir, '{}_{}.h5/nucl/dapi_mask_absmin'.format(dataset, block_label['id']))
        mask_absmin_im = MaskImage(maskpath_absmin, permission='r')
        mask_absmin_im.load(load_data=False)
    elif maskfile_compund:
        maskpath_sauvola = os.path.join(datadir, '{}_bfc_nucl-dapi_mask_sauvola.ims'.format(dataset))
        mask_sauvola_im = MaskImage(maskpath_sauvola, permission='r')
        mask_sauvola_im.load(load_data=False)
        mask_sauvola_im.slices[:3] = block_label['slices'][:3]
        mask_sauvola_im.slices[3] = slice(0, 1, None)
        maskpath_absmin = os.path.join(datadir, '{}_bfc_nucl-dapi_mask_absmin.ims'.format(dataset))
        mask_absmin_im = MaskImage(maskpath_absmin, permission='r')
        mask_absmin_im.load(load_data=False)
        mask_absmin_im.slices[:3] = block_label['slices'][:3]
        mask_absmin_im.slices[3] = slice(0, 1, None)

    mask_sauvola = mask_sauvola_im.slice_dataset().astype('bool')
    mask_absmin = mask_absmin_im.slice_dataset().astype('bool')
    mask = mask_absmin & mask_sauvola

    return mask


def add_features(df, image_in='', origin=[0, 0, 0], dsfacs=[1, 16, 16]):

    if 'centroid-0' in df.columns:

        cens = ['centroid-{}'.format(i) for i in [0, 1, 2]]
        coms = ['com_{}'.format(d) for d in 'zyx']

        df[coms] = df[cens] + origin

        if image_in:

            dt_im = Image(image_in, permission='r')
            dt_im.load(load_data=False)
            data = dt_im.slice_dataset()
            dt_im.close()

            ds_centroid = np.array(df[coms] / dsfacs, dtype='int')
            ds_centroid = [data[p[0], p[1], p[2]] for p in ds_centroid]
            df['dist_to_edge'] = np.array(ds_centroid)

    if 'inertia_tensor_eigvals-0' in df.columns:
        ites = ['inertia_tensor_eigvals-{}'.format(i) for i in [0, 1, 2]]
        eigvals = np.clip(np.array(df[ites]), 0, np.inf)
        df['fractional_anisotropy'] = fractional_anisotropy(eigvals)
        df['major_axis_length'] = get_ellips_axis_lengths(eigvals[:, 0])
        df['minor_axis_length'] = get_ellips_axis_lengths(eigvals[:, -1])

    # TODO: range, variance, ...

    return df


def get_intensity_df_data(regions, rp_props_int, ch_data, cpf):

    try:
        rpt = regionprops_table(regions, ch_data, properties=rp_props_int)
    except ValueError:
        print('got ValueError on INT {}'.format(cpf))
        cols = ['{}_{}'.format(cpf, col)
                for col in get_column_names(rp_props_int)]
        df_int = pd.DataFrame(columns=cols)
    else:
        df_int = pd.DataFrame(rpt)
        df_int.columns = ['{}_{}'.format(cpf, col)
                          for col in get_column_names(rp_props_int)]

    return df_int


def get_intensity_df(regions, rp_props_int, data, ch, bf=None):

    data.slices[data.axlab.index('c')] = slice(ch, ch + 1, 1)
    ch_data = data.slice_dataset()

    if bf is not None:
        bias = get_bias_field_block(bf, data.slices, ch_data.shape)
        bias = np.reshape(bias, ch_data.shape)
        ch_data /= bias
        ch_data = np.nan_to_num(ch_data, copy=False)

    try:
        rpt = regionprops_table(regions, ch_data, properties=rp_props_int)
    except ValueError:
        print('got ValueError on INT {}'.format(ch))
        cols = ['ch{:02d}_{}'.format(ch, col)
                for col in get_column_names(rp_props_int)]
        df_int = pd.DataFrame(columns=cols)
    else:
        df_int = pd.DataFrame(rpt)
        df_int.columns = ['ch{:02d}_{}'.format(ch, col)
                          for col in get_column_names(rp_props_int)]

    return df_int


def split_filename(filename, blockoffset=[0, 0, 0]):
    """Extract the data indices from the filename."""

    datadir, tail = os.path.split(filename)
    fname = os.path.splitext(tail)[0]
    parts = re.findall('([0-9]{5}-[0-9]{5})', fname)
    id_string = '_'.join(parts)
    dset_name = fname.split(id_string)[0][:-1]

    x = int(parts[-3].split("-")[0]) - blockoffset[0]
    X = int(parts[-3].split("-")[1]) - blockoffset[0]
    y = int(parts[-2].split("-")[0]) - blockoffset[1]
    Y = int(parts[-2].split("-")[1]) - blockoffset[1]
    z = int(parts[-1].split("-")[0]) - blockoffset[2]
    Z = int(parts[-1].split("-")[1]) - blockoffset[2]

    dset_info = {'datadir': datadir, 'base': dset_name,
                 'nzfills': len(parts[1].split("-")[0]),
                 'postfix': id_string,
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}

    return dset_info, x, X, y, Y, z, Z


def dset_name2slices(dset_name, blockoffset=[0, 0, 0], axlab='xyz', shape=[]):
    """Get slices from data indices in a filename."""

    _, x, X, y, Y, z, Z = split_filename(dset_name, blockoffset)
    slicedict = {'x': slice(x, X, 1),
                 'y': slice(y, Y, 1),
                 'z': slice(z, Z, 1)}
    for dim in ['c', 't']:
        if dim in axlab:
            upper = shape[axlab.index(dim)]
            slicedict[dim] = slice(0, upper, 1)

    slices = [slicedict[dim] for dim in axlab]

    return slices


def get_column_names(rp_props):

    cols = []
    for i, it in enumerate(rp_props):
        if 'centroid' in it or 'eigvals' in it:
            cols += ['{}-{}'.format(it, dim)
                     for dim in [0, 1, 2]]
        elif 'moments' in it:
            # FIXME: need only half of the diagonal matrix
            cols += ['{}-{}-{}-{}'.format(it, dim1, dim2, dim3)
                     for dim1 in [0, 1, 2, 3]
                     for dim2 in [0, 1, 2, 3]
                     for dim3 in [0, 1, 2, 3]]
        else:
            cols += [it]

    return cols


def get_empty_dataframe(rp_props_morph, rp_props_add, rp_props_int, channels):

    cols_morph = get_column_names(rp_props_morph)

    cols_int = get_column_names(rp_props_int)
    cols = ['ch{:02d}_{}'.format(ch, col) for ch in channels for col in cols_int]

    df = pd.DataFrame(columns=cols_morph + rp_props_add + cols)

    return df


def get_feature_set(fset_morph='minimal', fset_intens='minimal', aux_data_path=''):

    # TODO: self-defined features
    """
    # eccentricity not implemented
    # orientation not implemented
    # perimeter not implemented
    # moments_hu not implemented
    # weighted_moments_hu not implemented
    # max_intensity needs aux data volume
    # mean_intensity needs aux data volume
    # min_intensity needs aux data volume
    # weighted_centroid needs aux data volume
    # weighted_moments needs aux data volume
    # weighted_moments_central needs aux data volume
    # weighted_moments_normalized needs aux data volume
    """

    # morphological features
    # TODO: file bug report on minor/major_axis_length
    # gives this ValueError:
    # https://github.com/scikit-image/scikit-image/issues/2625
    fsets_morph ={
        'none': (
            'label',
            ),
        'minimal': (
            'label',
            'area',
            'centroid'
            ),
        'medium': (
            'label',
            'area',
            'centroid',
            'bbox',
            'equivalent_diameter',
            'extent',
            'euler_number',
            'inertia_tensor_eigvals',
            ),
        'maximal': (
            'label',
            'area',
            'bbox',
            'centroid',
            'equivalent_diameter',
            'extent',
            'euler_number',
            # 'convex_area',
            # 'solidity',
            'moments',
            'moments_central',
            'moments_normalized',
            # 'orientation',
            'inertia_tensor_eigvals',
            # 'major_axis_length',
            # 'minor_axis_length',
            ),
        }

    # intensity features
    fsets_intens ={
        'none': (),
        'minimal': (
            'mean_intensity',
            ),
        'medium': (
            'mean_intensity',
            'weighted_centroid',
            ),
        'maximal': (
            'min_intensity',
            'mean_intensity',
            'median_intensity',
            'variance_intensity',
            'max_intensity',
            'weighted_centroid',
            # FIXME: OverflowError: Python int too large to convert to C long
            # 'weighted_moments',
            # 'weighted_moments_central',
            # 'weighted_moments_normalized',
            ),
        }

    # FIXME: convex hull morph features often fail
    # FIXME: intensity weighted fail

    try:
        morph = fsets_morph[fset_morph]
    except (KeyError, TypeError):
        morph = fset_morph

    try:
        intens = fsets_intens[fset_intens]
    except (KeyError, TypeError):
        intens = fset_intens

    try:
        addit = get_additional_columns(aux_data_path, fset_morph)
    except (KeyError, TypeError):
        addit = fset_addit

    return morph, intens, addit


def get_additional_columns(aux_data_path='', fset_morph='minimal'):

    cols_add = ['com_z', 'com_y', 'com_x']
    if aux_data_path:
        cols_add += ['dist_to_edge']
    if fset_morph == 'maximal':
        cols_add += ['fractional_anisotropy',
                     'major_axis_length', 'minor_axis_length']

    return cols_add


def find_border_labels(labels):

    border_labelset = set([])

    fullslices = [slc for slc in labels.slices]

    for dim in [0, 1, 2]:
        for s in [0, -1]:

            if s:
                fullstop = fullslices[dim].stop
                labels.slices[dim] = slice(fullstop - 1, fullstop, None)
            else:
                fullstart = fullslices[dim].start
                labels.slices[dim] = slice(fullstart, fullstart + 1, None)

            labeldata = labels.slice_dataset()
            border_labelset |= set(np.unique(labeldata))

            labels.slices = [slc for slc in fullslices]

    border_labelset -= set([0])

    return border_labelset


def split_nucl_and_memb_data(labeldata, nuclearmask=None):

    labeldata_memb = np.copy(labeldata)
    labeldata_nucl = np.copy(labeldata)

    memb_mask = find_boundaries(labeldata)
    for i, slc in enumerate(memb_mask):
        memb_mask[i, :, :] = binary_dilation(slc)
    labeldata_memb[~memb_mask] = 0

    if nuclearmask is None:
        nuclearmask = ~memb_mask

    labeldata_nucl[~nuclearmask] = 0

#     print('mask_nucl0_sum', np.sum(~memb_mask))
#     print('mask_nucl1_sum', np.sum(nuclearmask))
#     print('mask_memb_sum', np.sum(memb_mask))
#     print('label_full_sum', np.sum(labeldata.astype('bool')))
#     print('label_memb_sum', np.sum(labeldata_memb.astype('bool')))
#     print('label_nucl_sum', np.sum(labeldata_nucl.astype('bool')))

    return labeldata_memb, labeldata_nucl


def split_nucl_and_memb(labels, outpat, nuclearmask=None):

    labeldata = labels.slice_dataset()

    labeldata, labeldata_nucl = split_nucl_and_memb_data(labeldata, nuclearmask)

    pf = '_memb'
    outpath = outpat.format(pf)
    im_memb = LabelImage(outpath, **props)
    im_memb.create()
    im_memb.write(labeldata)

    pf = '_nucl'
    outpath = outpat.format(pf)
    im_nucl = LabelImage(outpath, **props)
    im_nucl.create()
    im_nucl.write(labeldata_nucl)

    return im_memb, im_nucl


def small_label_mask(labeldata, maxlabel, min_labelsize=5):
    """
    NOTE:
    - need to take out very small labels (<5) for convex hull
    - even without these labels, convex hull often fails
    - removed features dependent on convex hull from the feature sets for now
    """

    rp = regionprops(labeldata)

    small_labels = [prop.label for prop in rp if prop.area < min_labelsize]
    fw = np.zeros(maxlabel + 1, dtype='bool')
    fw[small_labels] = True
    smalllabelmask = np.array(fw)[labeldata]

    return smalllabelmask


def label_selection_mask(labeldata, filestem):

    import scanpy as sc

    filename = '{}_adata.h5ad'.format(filestem)
    adata = sc.read(filename)
    labelset = set(adata.obs['label'].astype('uint32'))

    ulabels = np.load('{}_ulabels.npy'.format(filestem))
    ulabels = np.delete(ulabels, 0)
    maxlabel = np.amax(ulabels)
    fw = np.zeros(maxlabel + 1, dtype='bool')
    for label in labelset:
        fw[label] = True
    mask = np.array(fw)[labeldata]

    return mask


def select_features(dfs, feat_select, min_size=0, split_features=False):

    df1 = dfs['full'][feat_select['morphs']]

    key = 'memb' if split_features else 'full'
    df2 = pd.DataFrame(index=df1.index)
    df2[feat_select['membrane']] = dfs[key][feat_select['membrane']]

    key = 'nucl' if split_features else 'full'
    df3 = pd.DataFrame(index=df1.index)
    df3[feat_select['nuclear']] = dfs[key][feat_select['nuclear']]

    # for rows with small nuclei: replace intensity features with the value for the full segment
    if min_size:
        key = 'nucl' if split_features else 'full'  # THIS doesnt achive anything
        dfx = pd.DataFrame(index=df1.index)
        dfx[feat_select['morphs']] = dfs[key][feat_select['morphs']]

        small = dfx['area'].isna() | dfx['area'] < min_size
        df2[small] = dfs['full'].loc[small][feat_select['membrane']]
        df3[small] = dfs['full'].loc[small][feat_select['nuclear']]

    # create some compound features
    df4 = pd.DataFrame(index=df1.index)
    comcols = ['centroid-0', 'centroid-1', 'centroid-2']
    dfa = dfs['full'][comcols]
    dfb = dfs['full'][comcols]
    dfb[dfb.index.isin(dfs['nucl'].index)] = dfs['nucl'][comcols]
    comcols = ['ch00_weighted_centroid-0', 'ch00_weighted_centroid-1', 'ch00_weighted_centroid-2']
    dfc = dfs['full'][comcols]
    #df4['dist_c'] = np.sqrt((np.square(np.array(dfa)-np.array(dfb)).sum(axis=1)))
    #df4['dist_w'] = np.sqrt((np.square(np.array(dfa)-np.array(dfc)).sum(axis=1)))
    #df4['dist_cn'] = df4['dist_c'] / dfs['full']['major_axis_length']
    dist_w = np.sqrt((np.square(np.array(dfa)-np.array(dfc)).sum(axis=1)))
    df4['polarity'] = dist_w / dfs['full']['major_axis_length']

    df = pd.concat([df1, df2, df3, df4], axis=1)

    return df


def rename_columns(df, pairs={}, metrics=['mean']):

    var_names = list(df.columns)
    if not pairs:
        markers = ['DAPI', 'KI67', 'PAX8', 'NCAM1',
                   'SIX2', 'CADH1', 'CADH6', 'FACTIN']
        pairs = {
            'ch{:02d}_{}_intensity'.format(i, metric): '{}_{}'.format(marker, metric)
            for i, marker in enumerate(markers)
            for metric in metrics
            }
        pairs['area'] = 'volume'
    all_var_names = [pairs[featname] if featname in pairs.keys() else featname
                     for featname in var_names]
    df.columns = all_var_names

    return df


def get_feature_names(fset_morph='', fset_intens='', metrics=['mean']):

    feat_names = {}
    if fset_morph == 'minimal':
        feat_names['morphs'] = [
            'area',
            'com_z', 'com_y', 'com_x',
        ]
    elif fset_morph == 'maximal':
        feat_names['morphs'] = [
            'area',
            'com_z', 'com_y', 'com_x',
            'equivalent_diameter',
            'extent',
            'fractional_anisotropy',
            'major_axis_length', 'minor_axis_length',
        ]
    else:
        feat_names['morphs'] = fset_morph

    # TODO: generalize with arguments
    d = {'markers': list(range(8)),
         'membrane': [3, 5, 6, 7],
         'nuclear': [0, 1, 2, 4],
         }
    # metrics = ['mean', 'median', 'variance', 'min', 'max']
    marker_ids = ['ch{:02d}'.format(i) for i in d['markers']]
    # marker_ids[1] = 'ch01p'
    for k, v in d.items():
        feat_names[k] = [
            '{}_{}_intensity'.format(marker_ids[i], metric)
            for i in v for metric in metrics
            ]

    feat_names['spatial'] = [
        'dist_to_edge', 'polarity',
        ]

    return feat_names


def postprocess_features(
        seg_paths,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        csv_dir='',
        csv_stem='',
        feat_pf='_features',
        segm_pfs=['full', 'memb', 'nucl'],
        ext='csv',
        min_size_nucl=50,
        save_border_labels=False,
        split_features=False,
        fset_morph='minimal',
        fset_intens='minimal',
        ):

    labels = LabelImage(seg_paths[0], permission='r')
    labels.load(load_data=False)
    comps = labels.split_path()

    csv_dir = csv_dir or comps['dir']

    mpi = wmeMPI(usempi=False)
    mpi.set_blocks(labels, blocksize, blockmargin, blockrange)
    mpi.scatter_series()

    li = []
    for i in mpi.series:
        print('processing block {:03d}'.format(i))
        block = mpi.blocks[i]

        # read the csv's
        filestem = '{}_{}{}'.format(csv_stem, block['id'], feat_pf)
        dfs = {}
        for segm_pf in segm_pfs:
            filename = '{}_{}.{}'.format(filestem, segm_pf, ext)
            filepath = os.path.join(csv_dir, filename)
            dfs[segm_pf] = pd.read_csv(filepath, index_col='label', header=0)

        if len(dfs['full'].index) == 0:
            continue

        # select features
        # metrics=['mean', 'median', 'variance', 'min', 'max']
        metrics=['mean']
        feat_names = get_feature_names(fset_morph, fset_intens, metrics)
        df = select_features(dfs, feat_names, min_size_nucl, split_features)
        #df = rename_columns(df, metrics=metrics)

        # label filtering: only select full segments
        filestem = '{}_{}'.format(csv_stem, block['id'])
        sl_path = os.path.join(csv_dir, '{}_smalllabels.pickle'.format(filestem))
        with open(sl_path, 'rb') as f:
            slabels = pickle.load(f)
        bl_path =  os.path.join(csv_dir, '{}_borderlabels.pickle'.format(filestem))
        if save_border_labels:
            labels.slices = block['slices']
            blabels = find_border_labels(labels)
            with open(bl_path, 'wb') as f:
                pickle.dump(blabels, f)
        else:
            with open(bl_path, 'rb') as f:
                blabels = pickle.load(f)

        blabels -= slabels
        df = df.drop(labels=blabels)

        li.append(df)

    combined_df = pd.concat(li, keys=mpi.series)
    combined_df.index.names = ['block', 'label']
    combined_df.reset_index(inplace=True)

    combined_df.drop_duplicates(subset='label', inplace=True)

    outputpath = os.path.join(csv_dir, '{}{}.csv'.format(csv_stem, feat_pf))
    combined_df.to_csv(outputpath, index=True, encoding='utf-8-sig')

    return combined_df


def fractional_anisotropy(eigvals):
    lm = np.sum(eigvals, axis=1) / 3
    fa = np.sqrt(3/2) * np.sqrt(np.sum( (eigvals.T - lm) ** 2, axis=0 ) ) / np.sqrt(np.sum(eigvals**2, axis=1))
    return fa


def get_ellips_axis_lengths(l):
    return 4 * np.sqrt(l)


#def polarity(com, wcom):
#    return distance.cdist(com, wcom)
    # return np.linalg.norm(com-wcom)


if __name__ == "__main__":
    main(sys.argv[1:])
