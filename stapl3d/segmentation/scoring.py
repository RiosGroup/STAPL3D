import numpy as np
import os
import nibabel as nib
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
import tifffile as tiffpy
#import SimpleITK as sitk
#import itk
import argparse
import datetime

import pickle

# parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser(description='Input parameters')
# parser.add_argument('-i', '--image', type=str, help='Segmentated image', required=False)
# parser.add_argument('-t', '--truth', type=str, help='Segmentation truth', required=False)
# args = parser.parse_args()


def score_segmentation(blockdir, dataset, blocks, models, idss):

    import multiprocessing

    def read(image_in, slcs=[]):
        from stapl3d import Image
        im = Image(image_in)
        im.load()
        if slcs:
            im.slices = slcs
        data = im.slice_dataset()
        im.close()
        return data

    arglist = []
    for block_id, d in blocks.items():

        blockstem = '{}_{}'.format(dataset, block_id)

        for model in models:

            filestem = os.path.join(blockdir, '{}_{}'.format(blockstem, model))
            filepath = '{}.h5'.format(filestem)

            for ids in idss:

                outstem = '{}{}_scores'.format(filestem, ids)

                if ids == '_unet3d': ids = ''

                arglist.append(
                    (
                        d['gt'],
                        read('{}/{}'.format(filepath, 'segm/labels{}'.format(ids)), slcs=[]),
                        [], [], 1.5, 0.5, 0.5,
                        outstem,
                    )
                )

    n_workers = min(14, len(arglist))
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(assess_segmentation, arglist)


def assess_segmentation(truth, segmentation, seg_slices=[], seg_transpose=[],
                        F_weight=1.5, OSS_weight_ADS=0.5, OSS_weight_F=0.5,
                        outstem=''):
    """Assign each segment to a ground truth label"""

    if not isinstance(truth, np.ndarray):
        affine = nib.load(truth).affine
        truth = image_to_matrix(truth)
    else:
        affine = np.eye(4)

    #print("truth shape:", truth.shape)

    truth_labels = np.ndarray.tolist(np.unique(truth))
    truth_labels, truth_counts = np.unique(truth, return_counts=True)
    truth_labels_counts = dict(zip(truth_labels, truth_counts))

    truth_mask = truth != 0

    if not isinstance(segmentation, np.ndarray):
        segmentation = image_to_matrix(segmentation)

    if seg_slices:
        segmentation = segmentation[seg_slices]
    if seg_transpose:
        segmentation = np.transpose(segmentation, seg_transpose)

    seg_labels_mask = np.ndarray.tolist(np.unique(segmentation * truth_mask))
    seg_labels_mask.remove(0)
    #print("seg shape:", segmentation.shape)

    seg_labels, seg_counts = np.unique(segmentation, return_counts=True)
    unfilt_seg_labels_counts = dict(zip(seg_labels, seg_counts))
    seg_labels_counts = {label: unfilt_seg_labels_counts[label]
                         for label in seg_labels_mask}

    print("Total # of labels:", len(seg_labels))
    print("Overlapping # of labels:", len(seg_labels_counts))

    overlapping_labels = [label for label in seg_labels_mask]
    masked_seg = segmentation * np.isin(segmentation, overlapping_labels)
    seg_labels, seg_counts = np.unique(masked_seg, return_counts=True)
    temp_out = nib.Nifti1Image(masked_seg, affine)
    nib.save(temp_out, "Only_overlapping_labels.nii.gz")
    bounding_box_coords = bounding_box_limits(masked_seg)
    x0, x1, y0, y1, z0, z1 = bounding_box_coords

    segmentation_cropped = segmentation[x0:x1, y0:y1, z0:z1]
    truth_cropped = truth[x0:x1, y0:y1, z0:z1]
#    print(segmentation_cropped.shape)
#    print(truth_cropped.shape)
#    print(sorted(list(seg_labels_counts.values())))
    dice_dict = {}
    dice_dict[0] = 0
    dice_dict_TP_seg = {}
    dice_dict_TP_seg[0] = 0
    opt_dice_scores = []
    dice_dict_from_seg = {}
    dice_dict_from_seg[0] = 0
    main_truth_per_seg = {}

    ## ASSIGN A MAIN TRUTH LABEL TO EACH SEGMENT
    print("####### FROM SEG")
    for seg_label, count in seg_labels_counts.items():
        if seg_label==0:
            continue
        single_seg_seg = (segmentation_cropped == seg_label)
        lab, count = np.unique(truth_cropped[single_seg_seg==1], return_counts=True)
        overlapping_labels = dict(zip(lab, count))
        seg_overlap = lab[np.unravel_index(np.argmax(count, axis=None), count.shape)]                       #### HEAVY
        single_seg_truth = (truth_cropped==seg_overlap)

        main_truth_per_seg[seg_label] = lab[np.unravel_index(np.argmax(count, axis=None), count.shape)]

        ### CALCULATE DICE SCORE FOR EACH SEGMENT COMAPRED TO THE MOST OVERLAPPING TRUTH SEGMENT
        if main_truth_per_seg[seg_label] == 0:
            continue

        dice = calculate_dice(single_seg_truth, single_seg_seg)
#        print(dice)
        opt_dice_scores.append(dice)
        dice_dict_from_seg[seg_label]=dice

    print(len(dice_dict_from_seg), len(seg_labels_counts))

    labels_per_truth = {}
    dice_scores = []
    dice_scores_only_tp = []
    jaccard_scores = []
    hausdorff_scores = []
    print("####### FROM TRUTH")
    for truth_label, truth_count in truth_labels_counts.items():
        single_seg_seg=None
        if truth_label==0:
            continue

        labels_per_truth[truth_label] = []
        single_seg_truth = (truth_cropped==truth_label).astype(int)
        lab, count = np.unique(segmentation_cropped[single_seg_truth==1], return_counts=True)
        overlapping_labels = dict(zip(lab, count))
        full_seg_overlap = lab[np.unravel_index(np.argmax(count, axis=None), count.shape)]
        full_single_seg_seg = (segmentation_cropped==full_seg_overlap).astype(int)

        possible_TP = []
        possible_TP_count = []
        for label, count in overlapping_labels.items():
            if label == 0:  # FIXME: check why label can be 0 here [found on nucleus scoring ID04ID05]
                continue
            if main_truth_per_seg[label]==truth_label:
                possible_TP.append(label)
                possible_TP_count.append(count)
        if len(possible_TP)==0:
            dice_dict[truth_label] = 0
            continue
        else:
            labels_per_truth[truth_label] = possible_TP
            max_count = possible_TP[possible_TP_count.index(max(possible_TP_count))]
            single_seg_seg = (segmentation_cropped==max_count).astype(int)

        if truth_label!=0:
            full_dice = calculate_dice(single_seg_truth, full_single_seg_seg)
            dice_scores.append(full_dice)
            ## Calculate the Dice similarity score for each label
            dice = calculate_dice(single_seg_truth, single_seg_seg)
            dice_dict[truth_label] = dice
            dice_dict_TP_seg[max_count] = dice
            dice_scores_only_tp.append(dice)
            jaccard = calculate_jaccard(single_seg_truth, single_seg_seg)
            jaccard_scores.append(jaccard)
    #print(dice_dict.keys())
    #print("LEN", len(dice_scores))
    #print("LEN", len(dice_scores_only_tp))
    ## Results
    if 0 in truth_labels:
        nr_labels_truth = len(truth_labels)-1
    else:
        nr_labels_truth = len(truth_labels)

    nr_labels_seg=0
    for seg_label, assigned_truth_label in main_truth_per_seg.items():
        if assigned_truth_label!=0 and seg_label!=0:
            nr_labels_seg += 1

    ### DICE FROM TRUTH PREFERS OVERSEGMENTATION
    if len(dice_scores_only_tp) > 0:
        average_dice_from_truth = np.mean(dice_scores_only_tp)
        average_jaccard = np.mean(jaccard_scores)
    else:
        average_dice_from_truth = 0
        average_jaccard = 0

    average_dice_from_truth_full = np.mean(dice_scores)
    # average_hausdorff = np.mean(hausdorff_scores)
    overseg, underseg, precision, recall=calculate_over_under_segmentation(labels_per_truth, main_truth_per_seg)

    def calculate_fscore(F_weight, precision, recall):
        if precision == 0 and recall == 0:
            F_score = 0
        else:
            F_score = ( ( 1 + F_weight ** 2 ) * precision * recall ) / ( ( ( F_weight ** 2 ) * precision ) + recall )
        return F_score

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
        'N_GT': nr_labels_truth,
        'N_SEG': nr_labels_seg,
        'OSEG': len(overseg),
        'USEG': len(underseg),
    }
    ## Generalized output
    print(" \n------------------------------------------------------------\n")
    print(" \n------------------------------------------------------------\n")
    print("OSS is: {}".format(OSS))
    print("Average dice score of all TP is {}".format(average_dice_from_truth))
    print("Average jaccard similarity score of all TP is {}".format(average_jaccard))
    print("Segmentation precision: {}".format(precision))
    print("Segmentation recall: {}".format(recall))
    print("F1-score is {}".format(F1_score))
    print("F-score with weight {} is {}".format(F_weight, F_score))
    print(" \n------------------------------------------------------------\n")
    print("Average dice score of all truth labels on overlap is {}".format(average_dice_from_truth_full))
    # print("(Still needs validation) Average hausdorff distance is {}".format(average_hausdorff))
    print("Segments in truth:", nr_labels_truth)
    print("Overlapping segments in automatic segmentation:", nr_labels_seg)
    print("Number of cells oversegmented: {}/{}".format(len(overseg), nr_labels_truth))
    print("Number of cells undersegmented: {}/{}".format(len(underseg), nr_labels_truth))
    #display_dice(dice_dict_TP_seg, dice_dict, truth, segmentation, affine, seg_labels)

    if outstem:
        with open('{}.pickle'.format(outstem), 'wb') as f:
            pickle.dump(scores, f)

    return(scores)

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
    x0,y0,z0=coords.min(axis=0)
    x1,y1,z1=coords.max(axis=0) + 1
    print(z0,y0,x0,z1,y1,x1)
    return(x0, x1, y0, y1, z0, z1)

def segment_border(segment):
    seg_without_border=binary_erosion(segment)
    border = segment * np.invert(seg_without_border)
    return (border)

def calculate_dice(truth_1_0, segmentation_1_0):
    TP=np.sum(np.logical_and(truth_1_0, segmentation_1_0))
    FP=np.sum(segmentation_1_0)-TP
    FN=np.sum(truth_1_0)-TP
    dice = (2.0*TP) / (2.0*TP + FP + FN)
    return(dice)

def calculate_hausdorff(truth_1_0, segmentation_1_0):
    data_spacing=[1,1,1]
    # truth_border=truth_1_0
    truth_border=segment_border(truth_1_0)
    truth_border = sitk.GetImageFromArray(truth_border)
    truth_border.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
    # seg_border=segmentation_1_0
    seg_border=segment_border(segmentation_1_0)
    seg_border = sitk.GetImageFromArray(seg_border)
    seg_border.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(truth_border, seg_border)
    hausdorff = hausdorff_distance_filter.GetAverageHausdorffDistance()
    # hausdorff=directed_hausdorff(truth_border, seg_border)
    # hausdorff=hausdorff_distance(truth_border, seg_border, distance="euclidean")
    return(hausdorff)

def calculate_jaccard(truth_1_0, segmentation_1_0):
    TP=np.sum(np.logical_and(truth_1_0, segmentation_1_0))
    FP=np.sum(segmentation_1_0)-TP
    FN=np.sum(truth_1_0)-TP
    jaccard = TP / (TP + FP + FN)
    return(jaccard)

## Calculate specificity and sensitivity over all segments vs background
def calculate_overall_sens_and_spec(truth, segmentation):
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
    ## Manually calculate over- and under-segmentation
    overseg=[]
    underseg=[]
    TP=0
    FP=0
    FN=0
    for truth_label, sg_labels in labels_per_truth.items():
        if truth_label==0:
            continue
        overseg_temp=0
        underseg_temp=0
        for seg_label in sg_labels:
            if main_truth_per_seg[seg_label]==truth_label:
                overseg_temp+=1
        if overseg_temp==1:
            TP+=1
        elif overseg_temp>1:
            overseg.append(truth_label) #If 2 or more segments mainly belongs to a truth_label, it is oversegmented
            TP+=1
            FP+=overseg_temp-1
        elif overseg_temp==0:
            underseg.append(truth_label) #If no segment mainly belongs to a truth_label, it is undersegmented
            FN+=1
    print("TP: {}, FP: {}, FN: {}".format(TP, FP, FN))
    if TP+FP==0:
        print("No TP or FP segments. Setting precision to 0")
        precision=0
    else:
        precision=TP/(TP+FP)

    if TP+FN==0:
        print("No TP or FN segments. Setting recall to 0")
        recall=0
    else:
        recall=TP/(TP+FN)
    return (overseg, underseg, precision, recall)

def display_dice(dice_dict_from_seg, dice_dict_from_truth , truth, seg, affine, seg_labels):
    # seg=seg[:,:,7]
    # truth=truth[:,:,7]
    seg=seg[:,:,:]
    truth=truth[:,:,:]

    plane=24
    plane_index=plane-1

    if truth.shape== (800, 800, 106):
        truth=truth[20:630, 275:575, 14:65]
    if seg.shape== (800, 800, 106):
        seg=seg[20:630, 275:575, 14:65]
    ###################################################################### FROM SEGMENTS
    k = np.array(list(dice_dict_from_seg.keys()))
    v = np.array(list(dice_dict_from_seg.values()))
    print(seg.shape)
    seg_cutout=seg[:]
    seg_cutout_out = nib.Nifti1Image(seg_cutout, affine)
    nib.save(seg_cutout_out, "seg_ws1.nii.gz")

    if seg.shape== (610, 300, 51):
        seg_cutout=seg[42:574, 15:269, plane_index]

    if truth.shape== (610, 300, 51):
        truth_singleplane=truth[42:574, 15:269, plane_index]
    seg_outline=find_boundaries(truth_singleplane, connectivity=1, mode='thick', background=0).astype(int)
    coordinates = np.column_stack(np.nonzero(seg_outline))[::-1]
    # outline=np.zeros(truth_singleplane.shape)
    outline_truth=seg_cutout.copy()
    outline_truth[tuple(coordinates.T)] = 0
    outline = nib.Nifti1Image(outline_truth, affine)
    nib.save(outline, "truth_outline.nii.gz")

    ws1_with_outline=seg_cutout.copy()
    ws1_with_outline[tuple(coordinates.T)] = 0
    ws1_with_outline = nib.Nifti1Image(ws1_with_outline, affine)
    nib.save(ws1_with_outline, "seg_ws1_outlined.nii.gz")


    print(truth.shape)
    temp_truth=truth[:]
    if truth.shape== (351, 251, 46):
        temp_truth=truth[63:263, 61:191, 15:29]
        temp_truth=temp_truth[:,:,plane_index]
    elif truth.shape== (610, 300, 51):
        temp_truth=truth[42:574, 15:269, plane_index]
    temp_truth = nib.Nifti1Image(temp_truth, affine)
    nib.save(temp_truth, "truthset_cutout.nii.gz")


    #### OVERLAY DICE OVER THE SEGMENTATION
    sidx = k.argsort()
    seg=seg*np.isin(seg, k)
    print(seg.shape)
    a = np.searchsorted(k,seg,sorter=sidx)
    truth_dice=v[a]
    dice_output=truth_dice

    dice_overlay = nib.Nifti1Image(truth_dice, affine)
    nib.save(dice_overlay, "full_seg_dice_overlay.nii.gz")

    if truth_dice.shape== (800, 800, 106):
        dice_output=dice_output[163:563, 201:601, :]
        dice_output=dice_output[199:399, 169:299, 30:50]
        dice_output=dice_output[:, :, 4:18]
    if truth_dice.shape== (610, 300, 51):
        dice_output=dice_output[42:574, 15:269, :]
    if truth_dice.shape== (351, 251, 46):
        dice_output=dice_output[63:263, 61:191, 15:29]
    print(dice_output.shape)
    # truth_dice=v[sidx[np.searchsorted(k,seg,sorter=sidx)]]
    dice_overlay = nib.Nifti1Image(dice_output, affine)

    nib.save(dice_overlay, "seg_dice_overlay.nii.gz")
    dice_output=dice_output[:,:,plane_index]
    dice_overlay = nib.Nifti1Image(dice_output, affine)
    nib.save(dice_overlay, "seg_dice_overlay_plane{}.nii.gz".format(plane))

    ############################################################################ FROM TRUTH

    #### OVERLAY DICE OVER THE TRUTH
    k = np.array(list(dice_dict_from_truth.keys()))
    v = np.array(list(dice_dict_from_truth.values()))

    sidx = k.argsort()
    a = np.searchsorted(k,truth,sorter=sidx)
    truth_dice=v[a]
    dice_output=truth_dice

    if truth_dice.shape== (800, 800, 106):
        dice_output=dice_output[163:563, 201:601, :]
        dice_output=dice_output[199:399, 169:299, 30:50]
        dice_output=dice_output[:, :, 4:18]
    if truth_dice.shape== (610, 300, 51):
        dice_output=dice_output[42:574, 15:269, :]
    if truth_dice.shape== (351, 251, 46):
        dice_output=dice_output[63:263, 61:191, 15:29]

    dice_3d=dice_output.flatten()
    nr_of_dice_colours=50
    colours=range(0,nr_of_dice_colours+1)
    step=1.0/nr_of_dice_colours
    values=list(np.arange(0,1+step,step))
    for id, val in enumerate(list(dice_3d)):
        for idx,comp_val in enumerate(values):
            if val<=comp_val:
                dice_3d[id]=colours[idx]
                break

    dice_3d=dice_3d.reshape(dice_output.shape)
    dice_3d = np.pad(dice_3d, ((20, 170), (275,225), (14, 41)))
    dice_3d = np.pad(dice_3d, ((42, 36), (15,31), (0, 0)))
    dice_3d = nib.Nifti1Image(dice_3d.astype("uint16"), affine)
    nib.save(dice_3d, "dice_3d.nii.gz")
    dice_overlay = nib.Nifti1Image(dice_output, affine)
    nib.save(dice_overlay, "truth_dice_overlay.nii.gz")
