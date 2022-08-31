#!/usr/bin/env python

"""Calculate features of segments.

"""

import os
import sys
import logging
import pickle
import shutil
import multiprocessing

import re
import glob
import yaml

import numpy as np

import pandas as pd

from skimage.measure import regionprops, regionprops_table

from stapl3d import parse_args, Stapl3r, Image, LabelImage
from stapl3d.blocks import Block3r

logger = logging.getLogger(__name__)


def main(argv):
    """Calculate features of segments."""

    steps = ['estimate', 'postprocess']
    args = parse_args('features', steps, *argv)

    featur3r = Featur3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        featur3r._fun_selector[step]()


class Featur3r(Block3r):
    """Calculate features of segments."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'features'

        super(Featur3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'estimate': self.estimate,
            'postprocess': self.postprocess,
            })

        self._parallelization.update({
            'estimate': ['blocks'],
            'postprocess': [],
            })

        self._parameter_sets.update({
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            'postprocess': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers',),
                },
            })

        # TODO
        self._parameter_table.update({
            })

        default_attr = {
            'blocks': [],
            'compartments': [],
            'channels': {},
            'filter_borderlabels': True,
            'morphological_features': 'minimal',
            'intensity_features': 'minimal',
            'spatial_features': [],
            'compound_features': [],
            'downsample_factors': [1, 1, 1],
            'morphological_feature_selection': {},
            'intensity_feature_selection': {},
            'intensity_channel_selection': {},
            'spatial_feature_selection': {},
            'compound_feature_selection': [],
            'thresholds': {},
            '_coord_prefix': 'dataset_',
            '_additional_features': [],
            '_additional_feature_funs': {},
            '_additional_feature_selection': {},
            # '_anisotropy_correction': False,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_featurer()

        self._init_log()

        self._prep_blocks()

        self.set_feature_set()

        if not self.channels:
            self.channels = {ch: f'ch{ch:02d}' for ch in range(self.fullsize['c'])}

    def _init_paths_featurer(self):

        # TODO
        prev_path = {
            'moduledir': 'segmentation', 'module_id': 'segmentation',
            'step_id': 'segmentation', 'step': 'estimate',
            'ioitem': 'outputs', 'output': 'blockfiles',
            }
        bpat = self._get_inpath(prev_path)
        if bpat == 'default':
            os.makedirs('blocks', exist_ok=True)
            bpat = self._build_path(moduledir='blocks', prefixes=[self.prefix, 'blocks'], suffixes=[{'b': 'p'}], ext='h5')

        bpat = self._l2p(['blocks', '{f}.h5'])

        blockstem = bpat.replace('.h5', '')
        datastem = self._build_path(moduledir='.', prefixes=[self.prefix])

        self._paths.update({
            'estimate': {
                'inputs': {
                    'data': self.inputpaths['blockinfo']['data'],
                    'blockfiles': f'{bpat}',
                    },
                'outputs': {
                    **{'blockstem': blockstem},
                    # **{f'smalllabels': f'{blockstem}_smalllabels.pickle'},
                    **{f'borderlabels': f'{blockstem}_borderlabels.pickle'},
                    **{f'{vol}_csv': f'{blockstem}_features_{vol}.csv' for vol in self.compartments.keys()},
                    }
                },
            'postprocess': {
                'inputs': {
                    **{f'{vol}_csv': f'{blockstem}_features_{vol}.csv' for vol in self.compartments.keys()},
                    },
                'outputs': {
                    'feature_csv': f'{datastem}_features.csv',
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def estimate(self, **kwargs):
        """Calculate features of segments."""

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, block):
        """Calculate features of segments of a datablock."""

        block = self._blocks[block]
        origin = [block.slices[al].start for al in block.axlab if al in 'xyz']

        inputs = self._prep_paths_blockfiles(self.inputs, block)
        outputs = self._prep_paths_blockfiles(self.outputs, block)

        # Load segmentations and intensity data.
        label_ims, all_regions, elsize, axlab = self._load_label_images(block)
        data_ims, all_data = self._load_intens_images(block, [inputs['data']])

        # Filter out segments that touch the block borders.
        if self.filter_borderlabels:
            border_labelset = set([])
            for comp in self.compartments:
                border_labelset |= filter_borders(label_ims[comp])
            if outputs['borderlabels']:
                with open(outputs['borderlabels'], 'wb') as f:
                    pickle.dump(border_labelset, f)

        # Pick some rows to test correction
        # if self._anisotropy_correction:
        #     n_samples = 20
        #     rp = regionprops(all_regions[pf])
        #     labs = np.array([prop.label for prop in rp])
        #     labs = np.random.choice(labs, size=n_samples, replace=False)

        for pf, regions in all_regions.items():

            try:

                features = set(self.morphological_features) - set(self._morph_replacements)
                features |= set(['label'])
                rpt = regionprops_table(regions, properties=list(features))
                df = pd.DataFrame(rpt)

            except (IndexError, ValueError):

                df = self._get_empty_dataframe()

            else:

                features = set(self.intensity_features) - set(self._intens_replacements)
                df = self._add_intensity_features(df, regions, all_data, features)
                df = self._add_coordinate_corrections(df, origin)
                df = self._add_inertia_features(df)
                df = self._add_spatial_features(df, inputs)
                df = self._add_additional_features(df)

                # if self._anisotropy_correction:
                #
                #     def calculate_evals(mask, spacing):
                #         # adapted from pyradiomics
                #
                #         vcoords = np.where(mask != 0)
                #         Np = len(vcoords[0])
                #         coordinates = np.array(vcoords, dtype='int').transpose((1, 0))  # Transpose equals zip(*a)
                #         pcoords = coordinates * spacing[None, :]
                #         pcoords -= np.mean(pcoords, axis=0)  # Centered at 0
                #         pcoords /= np.sqrt(Np)
                #
                #         covariance = np.dot(pcoords.T.copy(), pcoords)  # inertia_tensor
                #
                #         evals = np.linalg.eigvals(covariance)
                #
                #         machine_errors = np.bitwise_and(evals < 0, evals > -1e-10)
                #         if np.sum(machine_errors) > 0:
                #             evals[machine_errors] = 0
                #         evals.sort()  # sort the eigenvalues from small to large
                #
                #         return evals
                #
                #     prefix = 'evals'
                #     df = df[df.index.isin(labs)]
                #     cols = [f'{prefix}-{col}' for col in [2, 1, 0]]
                #     for i, row in enumerate(df.iterrows()):
                #         l = int(row[0])
                #         # print(f'proc label {l}')
                #         mask = regions == l
                #         df.loc[l, cols] = calculate_evals(mask, np.array(elsize))
                #     df = self._add_inertia_features(df, prefix)

                df = df.set_index('label')

            if self.filter_borderlabels:
                df = df.drop(border_labelset & set(df.index))

            df.to_csv(outputs[f'{pf}_csv'])

    def set_feature_set(self):
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
        fsets_morph = {
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
                'major_axis_length',
                'minor_axis_length',
                'fractional_anisotropy',
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
                'major_axis_length',
                'minor_axis_length',
                'fractional_anisotropy',
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
                'max_intensity',
                'mean_intensity',
                'median_intensity',
                'variance_intensity',
                'weighted_centroid',
                # FIXME: OverflowError: Python int too large to convert to C long
                # 'weighted_moments',
                # 'weighted_moments_central',
                # 'weighted_moments_normalized',
                ),
            }

        def add_block_correction(features, featnames):
            for featname in featnames:
                if featname in features:
                    return list(set([f'{self._coord_prefix}{featname}']) | set(features))
                else:
                    return features


        if isinstance(self.morphological_features, list):
            pass
        elif self.morphological_features in fsets_morph.keys():
            self.morphological_features = fsets_morph[self.morphological_features]

        self.morphological_features = add_block_correction(self.morphological_features, ['centroid', 'bbox'])

        self._morph_replacements = [
            'major_axis_length',
            'minor_axis_length',
            'fractional_anisotropy',
            ]
        for featname in ['centroid', 'bbox']:
            if featname in self.morphological_features:
                self._morph_replacements += [f'{self._coord_prefix}{featname}']


        if isinstance(self.intensity_features, list):
            pass
        elif self.intensity_features in fsets_intens.keys():
            self.intensity_features = fsets_intens[self.intensity_features]
        self.intensity_features = add_block_correction(self.intensity_features, ['weighted_centroid'])

        self._intens_replacements = []
        for featname in ['weighted_centroid']:
            if featname in self.intensity_features:
                self._intens_replacements += [f'{self._coord_prefix}{featname}']

    def _load_label_images(self, block):

        # Load label images  # NOTE: assuming blockfile input for now
        label_ims = {}
        for pf, ids in self.compartments.items():
            filepath = f'{block.path}/{ids}'
            im = LabelImage(filepath, permission='r')
            im.load(load_data=False)
            label_ims[pf] = im

        # Slice label images
        all_regions = {}
        for pf, label_im in label_ims.items():
            label_im.slices = None  # label_im.slices = block.slices[:3]
            all_regions[pf] = label_im.slice_dataset().astype('int')

        return label_ims, all_regions, label_im.elsize, label_im.axlab

    def _load_intens_images(self, block, data_paths):

        intens_from_blocks = False  # TODO

        # Load intensity images  # NOTE: assuming 4D full input for now
        data_ims = {}
        for i, data_path in enumerate(data_paths):
            pf = 'im{:02d}'.format(i)

            if intens_from_blocks:
                data_path = self.filepaths[block.idx]

            data = Image(data_path, permission='r')
            data.load(load_data=False)
            ch_idx = data.axlab.index('c')
            chs = [int(ch) for ch in self.channels.keys()]
            names = [self.channels[ch] for ch in chs]
            data_ims[pf] = {'im': data, 'ch': chs, 'names': names}

        # Slice intensity images
        all_data = {}
        for dpf, datadict in data_ims.items():
            data = datadict['im']
            data.slices = [block.slices[al] for al in data.axlab]
            for ch, name in zip(datadict['ch'], datadict['names']):
                data.slices[data.axlab.index('c')] = slice(ch, ch + 1, 1)
                ch_data = data.slice_dataset()
                all_data[name] = ch_data

        return data_ims, all_data

    def _get_empty_dataframe(self):

        cols_mor = self._get_column_names(self.morphological_features)

        cols_int = [f'{name}_{col}' for idx, name in self.channels.items()
                    for col in self._get_column_names(self.intensity_features)]

        cols_spa = self._get_column_names(self.spatial_features)

        cols_add = self._get_column_names(self._additional_features)

        df = pd.DataFrame(columns=cols_mor + cols_int + cols_spa + cols_add)

        return df

    def _get_column_names(self, rp_props):

        cols = []
        for i, it in enumerate(rp_props):
            if 'centroid' in it or 'eigvals' in it:
                cols += [f'{it}-{dim}' for dim in range(3)]
            elif 'bbox' in it:
                cols += [f'{it}-{dim}' for dim in range(6)]
            elif 'moments' in it:
                # FIXME: need only half of the diagonal matrix
                cols += [f'{it}-{dim1}-{dim2}-{dim3}'
                         for dim1 in [0, 1, 2, 3]
                         for dim2 in [0, 1, 2, 3]
                         for dim3 in [0, 1, 2, 3]]
            else:
                cols += [it]

        return cols

    def _add_intensity_features(self, df, regions, datadict, features):

        extra_properties = []
        if 'median_intensity' in features:
            extra_properties += [median_intensity]
        if 'variance_intensity' in features:
            extra_properties += [variance_intensity]
        if 'quantile95_intensity' in features:
            extra_properties += [quantile95_intensity]

        cols_int = self._get_column_names(features)

        for ch_pf, ch_data in datadict.items():

            cols = [f'{ch_pf}_{col}' for col in cols_int]

            try:
                rpt = regionprops_table(regions, ch_data,
                                        properties=features,
                                        extra_properties=extra_properties)
            except ValueError:
                print(f'ValueError on INT {ch_pf}')
                df[cols] = None
            else:
                df[cols] = pd.DataFrame(rpt, index=df.index)

        return df

    def _add_inertia_features(self, df, prefix=''):

        names = {
            'FA': 'fractional_anisotropy',
            'MAAL': 'major_axis_length',
            'MIAL': 'minor_axis_length',
            }

        if not 'inertia_tensor_eigvals-0' in df.columns:
            return df
        if not set(names.values()) & set(self.morphological_features):
            return df

        if not prefix:
            prefix = 'inertia_tensor_eigvals'
        else:
            names = {k: f'{prefix}-{name}' for k, name in names.items()}

        cols_in = [f'{prefix}-{i}' for i in [0, 1, 2]]
        eigvals = np.clip(np.array(df[cols_in]), 0, np.inf)

        df[names['FA']] = fractional_anisotropy(eigvals)
        df[names['MAAL']] = get_ellips_axis_lengths(eigvals[:, 0])
        df[names['MIAL']] = get_ellips_axis_lengths(eigvals[:, -1])

        return df

    def _add_spatial_features(self, df, inputs):

        spatial_features_funs = {
            'dist_to_mask': self._add_dist_to_mask_feature,
        }
        for feat in self.spatial_features:
            df = spatial_features_funs[feat](df, inputs)

        return df

    def _add_dist_to_mask_feature(self, df, inputs=[]):

        try:
            d2e_path = inputs['dist_to_mask']
        except KeyError:
            return df
        else:
            if not d2e_path or not 'centroid-0' in df.columns:
                return df

            # Add distance to edge feature:
            # from (downsampled) distance transform on sample mask
            coms = [f'com_{d}' for d in 'zyx']
            coms = [f'centroid-{d}' for d in [0, 1, 2]]

            dt_im = Image(d2e_path, permission='r')
            dt_im.load(load_data=False)
            data = dt_im.slice_dataset()
            dt_im.close()

            ds_centroid = np.array(df[coms] / self.downsample_factors, dtype='int')
            ds_centroid = [data[p[0], p[1], p[2]] for p in ds_centroid]
            df['dist_to_mask'] = np.array(ds_centroid)

        return df

    def _add_additional_features(self, df):

        for feat in self._additional_features:
            df = self._additional_feature_funs[feat](df)

        return df

    def _add_coordinate_corrections(self, df, origin=[0, 0, 0], suffix=''):

        for featname in ['centroid', 'bbox', 'weighted_centroid']:

            ori = origin + origin if featname == 'bbox' else origin

            cols = self._get_column_names([featname])
            newcols = self._get_column_names([f'{self._coord_prefix}{featname}'])

            for i, (col, newcol) in enumerate(zip(cols, newcols)):
                if not col in df.columns:
                    continue
                df[newcol] = df[col] + ori[i]

        return df

    def _add_centroid_features(self, df, origin=[0, 0, 0]):

        if not 'centroid-0' in df.columns:
            return df

        # Add centroid position in full dataset (corrected with block origin)
        cens = [f'centroid-{i}' for i in [0, 1, 2]]
        coms = [f'com_{d}' for d in 'zyx']
        df[coms] = df[cens] + origin

        return df

    def postprocess(self, **kwargs):
        """Calculate compound features of segments and combine blocks."""

        self._prep_step('postprocess', kwargs)
        self.blocks = self.blocks or list(range(len(self._blocks)))
        self._postprocess()

    def _postprocess(self):
        """Combine features of datablocks."""

        li = []

        for block_idx in self.blocks:

            block = self._blocks[block_idx]
            inputs = self._prep_paths_blockfiles(self.inputs, block)
            outputs = self._prep_paths_blockfiles(self.outputs, block)
            #inputs = self._prep_paths(self.inputs, reps={'b': block.idx})
            #outputs = self._prep_paths(self.outputs, reps={'b': block.idx})

            # Read the csv's
            dfs = {}
            for pf, ids in self.compartments.items():
                dfs[pf] = pd.read_csv(inputs[f'{pf}_csv'], index_col='label', header=0)

            # Check for empty csv's.
            nrows = [len(df.index) for k, df in dfs.items()]
            if not any(nrows):
                continue

            df = self.select_features(dfs)

            li.append(df)

        # Concatenate all blocks.
        df = pd.concat(li, keys=self.blocks)
        df.index.names = ['block', 'label']
        df.reset_index(inplace=True)

        # drop duplicate labels  # NOTE: need to have filtered border in estimate!
        #df.drop_duplicates(subset='label', inplace=True)

        # Drop rows by any thresholds specified {colname: [Tlow, Thigh]}.
        for col, r in self.thresholds.items():
            r = [eval(str(a)) for a in r]
            b = df[col].isna() | (df[col] < r[0]) | (df[col] > r[1])
            df = df.drop(df[b].index)

        self.df = df

        df.to_csv(outputs['feature_csv'], index=False, encoding='utf-8-sig')

        return df

    def select_features(self, df_dict):

        # Gather features of the various types.
        dfs = []
        dfs += self.select_features_of_type(
            df_dict, self.compartments,
            self.morphological_features, self.morphological_feature_selection,
            )
        dfs += self.select_features_of_type(
            df_dict, self.compartments,
            self.intensity_features, self.intensity_feature_selection,
            self.channels, self.intensity_channel_selection,
            )
        dfs += self.select_features_of_type(
            df_dict, self.compartments,
            self.spatial_features, self.spatial_feature_selection,
            )
        dfs += self.select_features_of_type(
            df_dict, self.compartments,
            self._additional_features, self._additional_feature_selection,
            )

        # Create compound features.  # NB needs centroids
        if self.compound_feature_selection:
            sel = self.compound_feature_selection
        elif type(self.compound_features) == dict:
            sel = list(self.compound_features.keys())
        elif type(self.compound_features) == list:
            sel = self.compound_features

        if 'polarity' not in sel:
            return pd.concat(dfs, axis=1, join='inner')

        def polarity_um_switch():
            try:
                ids = list(self.compartments.values())[0]
                filepath = f'{self._blocks[0].path}/{ids}'
                im = LabelImage(filepath, permission='r')
                im.load(load_data=False)
                im.close()
            except:
                elsize = [1, 1, 1]
                name = 'polarity'
            else:
                elsize = im.elsize
                name = 'polarity-um'
            return name, elsize
        name, elsize = polarity_um_switch()

        dfs += [self._add_polarity_feature(df_dict, elsize, name)]

        return pd.concat(dfs, axis=1)

    def select_features_of_type(self, df_dict, compartments, features=[], feature_selection={}, channels={}, channel_selection={}):

        dfs = []

        # Select all if no selection specified.
        if not feature_selection and features:
            features = [feat for feat in features if feat != 'label']  # label loaded as index
            # if channels:
            #     feature_selection = {ckey: [{f: channels} for f in features] for ckey in compartments.keys()}
            # else:
            feature_selection = {ckey: features for ckey in compartments.keys()}

        for ckey, feats in feature_selection.items():

            if channels:  # feats is a list of dict of {featname: list of channel_idxs}
                cols = []
                for feat in feats:
                    for featname, chs in feat.items():
                        colnames = self._get_column_names([featname])
                        for col in list(colnames):
                            chs = channels.keys() if chs == [-1] else chs
                            for ch in chs:
                                chname = channels[ch]
                                cols += [f'{chname}_{col}']
            else:
                cols = self._get_column_names(feats)

            df = df_dict[ckey][cols]
            df.columns = [f'{col}_{ckey}'for col in cols]
            dfs.append(df)

        return dfs

    def _add_polarity_feature(self, df_dict, elsize=[1, 1, 1], name='polarity'):

        def get_points(cols, df, elsize):

            if type(cols) is str:
                cols = [cols.format(i) for i in [0, 1, 2]]
            dfp = df[cols]
            for col, es in zip(cols, elsize):
                dfp.loc[:, col] = dfp.loc[:, col] * es

            return dfp

        pdict = {
            'cell_center': {
                'compartment': 'full',
                'feature': 'centroid',
                'channel': None,
                'columns': [],
                },
            'nucl_center': {
                'compartment': 'nucl',
                'feature': 'centroid',
                'channel': None,
                'columns': [],
                },
            'cell_length': {
                'compartment': 'full',
                'feature': 'major_axis_length',
                'channel': None,
                'columns': [],
                },
        }
        for k, v in pdict.items():
            pdict[k] |= self.compound_features['polarity'][k]

        dfs = []
        for n, p in pdict.items():
            comp = p['compartment']

            colnames = self._get_column_names([p['feature']])
            ch = [self.channels[p['channel']]] if type(p['channel']) is int else []
            cols = ['_'.join(ch + [colname]) for colname in colnames]

            df_t = df_dict[comp][cols]
            df_t.columns = pdict[n]['columns'] = [f'{col}_{comp}' for col in cols]
            dfs.append(df_t)

        df_t = pd.concat(dfs, axis=1)

        dfa = get_points(pdict['cell_center']['columns'], df_t, elsize)
        dfb = get_points(pdict['nucl_center']['columns'], df_t, elsize)
        dist = np.sqrt((np.square(np.array(dfa)-np.array(dfb)).sum(axis=1)))
        df_pol = dist / df_t[pdict['cell_length']['columns'][0]]
        df_pol = df_pol.to_frame()
        df_pol.columns = [name]

        return df_pol


def filter_borders(labels, outpath=''):

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

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(border_labelset, f)

    return border_labelset


def filter_size(all_regions, min_labelsize, outpath=''):

    pf = 'nucl' if 'nucl' in all_regions.keys() else 'full'
    rp = regionprops(all_regions[pf])
    small_labels = [prop.label for prop in rp if prop.area < min_labelsize]

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(set(small_labels), f)

    maxlabel = np.amax(all_regions['full'])
    fw = np.zeros(maxlabel + 1, dtype='bool')
    fw[small_labels] = True
    sl_mask = np.array(fw)[all_regions['full']]
    for pf in all_regions.keys():
        all_regions[pf][sl_mask] = 0

    return all_regions


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


def median_intensity(image, intensity_image):
    return np.median(intensity_image[image])


def variance_intensity(image, intensity_image):
    return np.var(intensity_image[image])


def quantile95_intensity(image, intensity_image):
    return np.quantile(intensity_image[image], 0.95)


def fractional_anisotropy(eigvals):
    lm = np.sum(eigvals, axis=1) / 3
    fa = np.sqrt(3/2) * np.sqrt(np.sum( (eigvals.T - lm) ** 2, axis=0 ) ) / np.sqrt(np.sum(eigvals**2, axis=1))
    return fa


def get_ellips_axis_lengths(l):
    return 4 * np.sqrt(l)


if __name__ == "__main__":
    main(sys.argv[1:])
