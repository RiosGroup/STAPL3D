import os
import yaml
import glob

from stapl3d import imarisfiles, blocks
from stapl3d.preprocessing import shading, stitching, masking, biasfield
from stapl3d.segmentation import enhance, segment, zipping, features

projectdir = 'E:\\Ravi'
dataset = '200720_Exp7_AP_HFK_25x'
datadir = os.path.join(projectdir, dataset)
filestem = os.path.join(datadir, dataset)

img_file = '{}.czi'.format(filestem)
par_file = '{}.yml'.format(filestem)

with open(par_file, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

channel_dir = os.path.join(datadir, cfg['dirtree']['datadir']['channels'])
stitch_stem = '{}{}{}'.format(dataset, cfg['shading']['params']['postfix'], cfg['stitching']['params']['postfix'])
ims_stem = os.path.join(datadir, stitch_stem)
ims_file = '{}.ims'.format(ims_stem)
ims_ref = '{}{}.ims'.format(ims_stem, cfg['dataset']['ims_ref_postfix'])
im_preproc ='{}{}.ims'.format(ims_stem, cfg['biasfield']['params']['postfix'])

shading.estimate(img_file, par_file)
shading.postprocess(img_file, par_file)

### BREAK - for shading-apply and stitching
#shading.apply(img_file, par_file)
#stitching.estimate(img_file, par_file)

### BREAK - for imaris channel extraction
    """
    - open <dataset>_shading_stitching.ims in Imaris
    - Edit ==> Delete Channels ==> [delete all but first channel]
    - File ==> Save as ... ==> [save as <ims_ref>, i.e. <dataset>_shading_stitching_ref_uint16.ims]
    """

imarisfiles.ims_to_zeros(ims_ref)

### BREAK - for saving imaris reference file
    """
    - open <ims_ref> in Imaris
    - File ==> Save as ... ==> [save as <ims_ref>, i.e. <dataset>_shading_stitching_ref_uint16.ims; just save over the old one: it will reduce the filesize]
    """

imarisfiles.split_channels(ims_file, par_file)

masking.estimate(ims_file, par_file)

biasfield.estimate(ims_file, par_file)
# [TODO: bias stack]
biasfield.apply(ims_file, par_file)

imarisfiles.make_aggregate(im_preproc, ims_ref, os.path.join(channel_dir, stitch_stem), '_ch??', cfg['biasfield']['params']['postfix'])

blocks.split(im_preproc, par_file)
enhance.estimate(im_preproc, par_file)  # NOTE: not for nucl-only
segment.estimate(im_preproc, par_file)
# [TODO: seg postproc]
zipping.relabel(im_preproc, par_file)
zipping.copyblocks(im_preproc, par_file)
zipping.estimate(im_preproc, par_file)

segment.subsegment(im_preproc, par_file)  # NOTE: not for nucl-only

blocks.merge(im_preproc, par_file)

features.estimate(im_preproc, par_file)
features.postproc(im_preproc, par_file)
