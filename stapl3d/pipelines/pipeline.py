import os
import yaml
import glob

from stapl3d import imarisfiles, blocks
from stapl3d.preprocessing import shading, masking, biasfield
from stapl3d.segmentation import enhance, segment, zipping, features

projectdir = '.'
dataset = '200302_RL57_P30T_25x'
datadir = os.path.join(projectdir, dataset)
filestem = os.path.join(datadir, dataset)

czi_file = '{}.czi'.format(filestem)
par_file = '{}.yml'.format(filestem)

with open(par_file, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

channel_dir = os.path.join(datadir, cfg['dirtree']['datadir']['channels'])
stitch_stem = '{}{}{}'.format(dataset, cfg['shading']['params']['postfix'], cfg['stitching']['params']['postfix'])
ims_stem = os.path.join(datadir, stitch_stem)
ims_file = '{}.ims'.format(ims_stem)
ims_ref = '{}{}.ims'.format(ims_stem, cfg['dataset']['ims_ref_postfix'])
im_preproc ='{}{}.ims'.format(ims_stem, cfg['biasfield']['params']['postfix'])


shading.estimate(czi_file, par_file)
shading.postprocess(czi_file, par_file)
#shading.estimate_channel(czi_file, 0, 0, **cfg['shading']['params'])

### BREAK - for shading-apply and stitching

### BREAK - for imaris channel extraction
imarisfiles.ims_to_zeros(ims_ref)
### BREAK - for saving imaris reference file
imarisfiles.split_channels(ims_file, par_file)

masking.estimate(ims_file, par_file)

biasfield.estimate(ims_file, par_file)
# [TODO: bias stack]
biasfield.apply(ims_file, par_file)

imarisfiles.make_aggregate(im_preproc, ims_ref, os.path.join(channel_dir, stitch_stem), '_ch??', cfg['biasfield']['params']['postfix'])

blocks.split(im_preproc, par_file)

enhance.estimate(im_preproc, par_file)

segment.estimate(im_preproc, par_file)

# [TODO: seg postproc]
zipping.relabel(im_preproc, par_file)
zipping.copyblocks(im_preproc, par_file)
zipping.estimate(im_preproc, par_file)

segment.subsegment(im_preproc, par_file)

blocks.merge(im_preproc, par_file)

features.estimate(im_preproc, par_file)
features.postproc(im_preproc, par_file)
