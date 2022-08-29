#!/usr/bin/env python

"""Score segemntations.
"""

import os
import sys
import logging
import pickle
import shutil
import multiprocessing
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from glob import glob

from stapl3d import parse_args, Stapl3r, Image, LabelImage
from stapl3d.blocks import Block3r


import nibabel as nib
import tifffile as tiffpy
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops


logger = logging.getLogger(__name__)


def main(argv):
    """Score segmentations"""

    steps = ['estimate']
    args = parse_args('scoring', steps, *argv)

    scor3r = Scor3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        scor3r._fun_selector[step]()


class Scor3r(Block3r):
    """Score segmentations."""

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'scoring'

        super(Scor3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'estimate': self.estimate,
            #'timeseries': self.timeseries,
            })

        self._parallelization.update({
            'estimate': ['blocks'],
            })

        self._parameter_sets.update({
            'estimate': {
                'fpar': self._FPAR_NAMES,
                'ppar': (),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'ids_labels': 'segm/labels',
            'ids_groundthruth': '',
            'F_weight': 1.5,
            'OSS_weight_ADS': 0.5,
            'OSS_weight_F': 0.5,
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_scorer()

        self._init_log()

        self._prep_blocks()

        self._images = []
        self._labels = []

    def _init_paths_scorer(self):

        self._paths.update({
            'estimate': {
                'inputs': {
                    },
                'outputs': {
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def estimate_foo(self, **kwargs):
        """Score segmenations."""

        arglist = self._prep_step('estimate', kwargs)
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._estimate_block, arglist)

    def _estimate_block(self, block_idx):
        """Score segmentation."""

        pass

    def estimate(self, **kwargs):
        """Score segmenations."""

        arglist = self._prep_step('estimate', kwargs)
        arglist = arglist[:-1]  # for timeseries n-1
        with multiprocessing.Pool(processes=self._n_workers) as pool:
            pool.starmap(self._evaluate_pair, arglist)

    def _evaluate_pair(self, block_idx):
        """Score segmentation."""

        def read(image_in, slcs=[]):
            im = LabelImage(image_in)
            im.load()
            if slcs:
                im.slices = slcs
            data = im.slice_dataset()
            im.close()
            return data

        block = self._blocks[block_idx]
        filestem = block.path
        slcs = block.slices
        slcs[3] = slice(0, 1, None)
        tp0 = read(f'{filestem}.h5/{self.ids_labels}', slcs=slcs)

        block = self._blocks[block_idx + 1]
        filestem = block.path
        slcs = block.slices
        slcs[3] = slice(0, 1, None)
        tp1 = read(f'{filestem}.h5/{self.ids_labels}', slcs=slcs)

        print(len(np.unique(tp0)), len(np.unique(tp1)))

        assess_segmentation(
            tp0,
            tp1,
            F_weight=self.F_weight,
            OSS_weight_ADS=self.OSS_weight_ADS,
            OSS_weight_F=self.OSS_weight_F,
            outstem=f'{filestem}_scores',
            )




def get_counts(seg):
    labels, counts = np.unique(seg[seg.astype('bool')], return_counts=True)
    label_counts = dict(zip(labels, counts))
    label_counts.pop(0, None)
    return label_counts


def get_image(seg, seg_slices=[], seg_transpose=[]):

    if not isinstance(seg, np.ndarray):
        affine = nib.load(truth).affine
        seg = image_to_matrix(seg)
    else:
        affine = np.eye(4)

    if seg_slices:
        seg = seg[seg_slices]

    if seg_transpose:
        seg = np.transpose(seg, seg_transpose)

    return seg, affine


def get_coords(region, margin, dims):

    if len(region.bbox) > 4:
        z, y, x, Z, Y, X = tuple(region.bbox)
        z = max(0, z - margin)
        Z = min(dims[0], Z + margin)
    else:
        y, x, Y, X = tuple(region.bbox)
        z = 0
        Z = 1

    y = max(0, y - margin)
    x = max(0, x - margin)
    Y = min(dims[1], Y + margin)
    X = min(dims[2], X + margin)

    return x, X, y, Y, z, Z


def assess_segmentation(truth, segmentation, seg_slices=[], seg_transpose=[],
                        F_weight=1.5, OSS_weight_ADS=0.5, OSS_weight_F=0.5,
                        outstem=''):
    """Assign each segment to a ground truth label"""

    truth, affine = get_image(truth)
    segmentation, _ = get_image(segmentation, seg_slices, seg_transpose)

    """

    truth_labels_counts = get_counts(truth)
    seg_labels_counts = get_counts(segmentation)

    seg_labels_in_truth = set(np.unique(segmentation[truth.astype('bool')])) - set([0])
    seg_labels_in_truth_counts = {label: seg_labels_counts[label] for label in sorted(seg_labels_in_truth)}

    print(f'{len(truth_labels_counts.keys()):>8} labels in ground truth')
    print(f'{len(seg_labels_counts.keys()):>8} labels in segmentation')
    print(f'{len(seg_labels_in_truth_counts):>8} segmentation labels in ground truth mask')
    print('')

    # crop volumes
    bounding_box_coords = bounding_box_limits(segmentation)
    x0, x1, y0, y1, z0, z1 = bounding_box_coords
    truth = truth[x0:x1, y0:y1, z0:z1]
    segmentation = segmentation[x0:x1, y0:y1, z0:z1]


    """
    """
    import pandas as pd
    from skimage.segmentation import regionprops

    rp = regionprops(truth, segmentation)
    for prop in rp:
        pass

    features = ["coords", "area", "dice"]
    df = pd.DataFrame(columns=features)
    """


    dice_dict = {}
    dice_dict[0] = 0
    dice_dict_TP_seg = {}
    dice_dict_TP_seg[0] = 0
    opt_dice_scores = []
    dice_dict_from_seg = {}
    dice_dict_from_seg[0] = 0
    main_truth_per_seg = {}

    labels_per_truth = {}
    dice_scores = []
    dice_scores_only_tp = []
    jaccard_scores = []
    hausdorff_scores = []
    label_pairs = {}
    pair_dicts = {}

    dims = segmentation.shape
    margin = 50

    rp_seg = regionprops(segmentation)
    for region in rp_seg:

        x, X, y, Y, z, Z = get_coords(region, margin, dims)
        gt_region = truth[z:Z, y:Y, x:X]
        seg_region = segmentation[z:Z, y:Y, x:X]

        seg_label_mask = seg_region == region.label
        overlaps = get_counts(gt_region[seg_label_mask])

        try:
            max_overlap = sorted(overlaps, key=overlaps.get, reverse=True)[0]
        except IndexError:
            print('no dice')
            max_overlap = 0

        main_truth_per_seg[region.label] = max_overlap

        truth_label_mask = gt_region == max_overlap
        dice = calculate_dice(truth_label_mask, seg_label_mask)

        dice_dict_from_seg[region.label] = dice
        opt_dice_scores.append(dice)




    rp_gt = regionprops(truth)
    for region in rp_gt:

        # Get label region with margin.
        x, X, y, Y, z, Z = get_coords(region, margin, dims)
        gt_region = truth[z:Z, y:Y, x:X]
        seg_region = segmentation[z:Z, y:Y, x:X]

        # Find overlapping labels and largest overlapping.
        truth_label_mask = gt_region == region.label
        overlaps = get_counts(seg_region[truth_label_mask])
        try:
            max_overlap = sorted(overlaps, key=overlaps.get, reverse=True)[0]
        except IndexError:
            print('no dice')
            max_overlap = 0

#        main_truth_per_seg[region.label] = max_overlap

        seg_label_mask = seg_region == max_overlap


        possible_TP = []
        possible_TP_count = []
        for label, count in overlaps.items():
            # if the overlapping label was marked as the 'main truth segment' add it to the true positive candidates
            if main_truth_per_seg[label] == region.label:
                possible_TP.append(label)
                possible_TP_count.append(count)



        labels_per_truth[region.label] = []
        if len(possible_TP) == 0:
            dice_dict[region.label] = 0
            continue
        else:
            labels_per_truth[region.label] = possible_TP

            max_count = possible_TP[possible_TP_count.index(max(possible_TP_count))]
            single_seg_seg = segmentation == max_count



        full_dice = calculate_dice(single_seg_truth, full_single_seg_seg)
        dice_scores.append(full_dice)



        ## Calculate the Dice similarity score for each label
        dice = calculate_dice(single_seg_truth, single_seg_seg)
        dice_dict[truth_label] = dice
        dice_dict_TP_seg[max_count] = dice
        dice_scores_only_tp.append(dice)

        jaccard = calculate_jaccard(single_seg_truth, single_seg_seg)
        jaccard_scores.append(jaccard)

        pair_dicts[truth_label] = {
            'seg_label': max_count,
            'dice': dice,
            'jaccard': jaccard,
        }

        label_pairs[truth_label] = max_count





    ## Results
    nr_labels_seg = 0
    for seg_label, assigned_truth_label in main_truth_per_seg.items():
        if assigned_truth_label != 0 and seg_label != 0:
            nr_labels_seg += 1


    ### DICE FROM TRUTH PREFERS OVERSEGMENTATION
    if len(dice_scores_only_tp) > 0:
        average_dice_from_truth = np.mean(dice_scores_only_tp)
        average_jaccard = np.mean(jaccard_scores)
    else:
        average_dice_from_truth = 0
        average_jaccard = 0

    average_dice_from_truth_full = np.mean(dice_scores)

    overseg, underseg, precision, recall = calculate_over_under_segmentation(labels_per_truth, main_truth_per_seg)

    F1_score = calculate_fscore(1.0, precision, recall)
    F_score = calculate_fscore(F_weight, precision, recall)
    OSS = OSS_weight_ADS * average_dice_from_truth + OSS_weight_F * F_score

    scores = {
        'OSS': OSS,
        'ADS_TP': average_dice_from_truth,
        'JSS_TP': average_jaccard,
        'precision': precision,
        'recall': recall,
        'F1_score': F1_score,
        'F_weight': F_weight,
        'F_score': F_score,
        'ADS_OL': average_dice_from_truth_full,
        'N_GT': len(rp_gt),
        'N_SEG': nr_labels_seg,
        'OSEG': len(overseg),
        'USEG': len(underseg),
    }

    if outstem:

        with open('{}_pairwise.pickle'.format(outstem), 'wb') as f:
            pickle.dump(pair_dicts, f)

        with open('{}.pickle'.format(outstem), 'wb') as f:
            pickle.dump(scores, f)

    return scores, label_pairs


def image_to_matrix(image):

    if ".nii.gz" in image:
        # image_load= nib.load(image_path)
        image = nib.load(image)
        image = image.dataobj
        image_matrix = image[:, :, :]

    elif ".tiff" in image:
        image_matrix = tiffpy.imread(image)
        image_matrix = np.transpose(image_matrix, (2, 1, 0))
        image_matrix = np.flip(image_matrix, axis=1)
        image_matrix = np.flip(image_matrix, axis=2)

    return(image_matrix)


def bounding_box_limits(image):

    mask = image > 0
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1

    return x0, x1, y0, y1, z0, z1


def calculate_dice(truth_1_0, segmentation_1_0):
    TP=np.sum(np.logical_and(truth_1_0, segmentation_1_0))
    FP=np.sum(segmentation_1_0)-TP
    FN=np.sum(truth_1_0)-TP
    dice = (2.0*TP) / (2.0*TP + FP + FN)
    return(dice)


def calculate_jaccard(truth_1_0, segmentation_1_0):
    TP=np.sum(np.logical_and(truth_1_0, segmentation_1_0))
    FP=np.sum(segmentation_1_0)-TP
    FN=np.sum(truth_1_0)-TP
    jaccard = TP / (TP + FP + FN)
    return(jaccard)


def calculate_overall_sens_and_spec(truth, segmentation):
    """Calculate specificity and sensitivity over all segments vs background."""
    foreground_truth=(truth!=0).astype(int)
    foreground_seg=(segmentation[:]!=0).astype(int)

    foreground_TP = np.sum(np.logical_and(foreground_truth,foreground_seg))
    foreground_FP = np.sum(foreground_seg)-foreground_TP
    foreground_FN = np.sum(foreground_truth)-foreground_TP

    if len(truth.shape)==3:
        foreground_TN = (truth.shape[0]*truth.shape[1]*truth.shape[2])-foreground_TP-foreground_FP-foreground_FN
    else:
        foreground_TN = (truth.shape[0]*truth.shape[1])-foreground_TP-foreground_FP-foreground_FN

    sensitivity = foreground_TP / (foreground_TP+foreground_FN)
    specificity = foreground_TN / (foreground_TN+foreground_FP)
    print(foreground_TP, foreground_FP, foreground_FN, foreground_TN)
    return(sensitivity, specificity)


def calculate_over_under_segmentation(labels_per_truth, main_truth_per_seg):
    """Calculate over- and under-segmentation."""

    overseg, underseg, TP, FP, FN = [], [], 0, 0, 0

    for truth_label, sg_labels in labels_per_truth.items():

        if truth_label == 0:
            continue

        overseg_temp = 0
        underseg_temp = 0

        for seg_label in sg_labels:
            if main_truth_per_seg[seg_label] == truth_label:
                overseg_temp += 1

        if overseg_temp == 1:
            TP += 1
        elif overseg_temp > 1:
            overseg.append(truth_label) #If 2 or more segments mainly belongs to a truth_label, it is oversegmented
            TP += 1
            FP += overseg_temp - 1
        elif overseg_temp == 0:
            underseg.append(truth_label) #If no segment mainly belongs to a truth_label, it is undersegmented
            FN += 1

    print("TP: {}, FP: {}, FN: {}".format(TP, FP, FN))

    if TP + FP == 0:
        print("No TP or FP segments. Setting precision to 0")
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        print("No TP or FN segments. Setting recall to 0")
        recall = 0
    else:
        recall = TP / (TP + FN)

    return overseg, underseg, precision, recall


def calculate_fscore(F_weight, precision, recall):

    if precision == 0 and recall == 0:
        F_score = 0
    else:
        num = ( 1 + F_weight ** 2 ) * precision * recall
        den = ( F_weight ** 2 ) * precision + recall
        F_score = num / den

    return F_score


if __name__ == "__main__":
    main(sys.argv[1:])
