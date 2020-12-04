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

projectdir = 'G:\\mkleinnijenhuis\\PMCdata\\Brain'
dataset = 'MA11_good'
datadir = os.path.join(projectdir, dataset)
filestem = os.path.join(datadir, dataset)

projectdir = '/Users/michielkleinnijenhuis/PMCdata/Brain'
dataset = '20200628_MBR01_s7'
# dataset = '20200628_MBR01_s7_hindbrain2'
datadir = os.path.join(projectdir, dataset)
filestem = os.path.join(datadir, dataset)


projectdir = 'E:\\Ravi'
dataset = '200824_RL67_10Ttransplant_redo_25x'
datadir = os.path.join(projectdir, dataset)
filestem = os.path.join(datadir, dataset)


par_file = '{}.yml'.format(filestem)

import shutil
import stapl3d
par_file_ref = os.path.join(os.path.dirname(stapl3d.__file__), 'pipelines', 'params.yml')
shutil.copy2(par_file_ref, par_file)

with open(par_file, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

img_file = '{}.{}'.format(filestem, cfg['dataset']['file_format'])

channel_dir = os.path.join(datadir, cfg['dirtree']['datadir']['channels'])
stitch_stem = '{}{}{}'.format(dataset, cfg['shading']['params']['postfix'], cfg['stitching']['params']['postfix'])
ims_stem = os.path.join(datadir, stitch_stem)
ims_file = '{}.ims'.format(ims_stem)
ims_ref = '{}{}.ims'.format(ims_stem, cfg['dataset']['ims_ref_postfix'])
im_preproc ='{}{}.ims'.format(ims_stem, cfg['biasfield']['params']['postfix'])

#shading.estimate(img_file, par_file)
#shading.postprocess(img_file, par_file)
### BREAK - for shading-apply and stitching
# shading.apply(img_file, par_file, correct=False, write_to_tif=False)

# stitching.write_stack_offsets(img_file)
# sudo chflags -R nouchg Fiji.app
# sudo xattr -rd com.apple.quarantine Fiji.app
# stitching.estimate(img_file, par_file, stitch_steps=[1, 2], FIJI='/Users/michielkleinnijenhuis/Downloads/Fiji.app/Contents/MacOS/ImageJ-macosx')
# stitching.estimate(img_file, par_file, stitch_steps=[3, 4, 5], channels=[cfg['stitching']['params']['channel']])
# for i in range(7):  # TODO: loop from within function
#     stitching.estimate(img_file, par_file, stitch_steps=[6], channels=[i])

### BREAK - for imaris channel extraction
"""
    - open <dataset>_shading_stitching.ims in Imaris
    - Edit ==> Delete Channels ==> [delete all but first channel]
    - File ==> Save as ... ==> [save as <ims_ref>, i.e. <dataset>_shading_stitching_ref_uint16.ims]
"""

# imarisfiles.ims_to_zeros(ims_ref)

### BREAK - for saving imaris reference file
"""
    - open <ims_ref> in Imaris
    - File ==> Save as ... ==> [save as <ims_ref>, i.e. <dataset>_shading_stitching_ref_uint16.ims; just save over the old one: it will reduce the filesize]
"""

# imarisfiles.split_channels(ims_file, par_file)
# masking.estimate(ims_file, par_file)
# biasfield.estimate(ims_file, par_file)
# # [TODO: bias stack]
# biasfield.apply(ims_file, par_file)
# imarisfiles.make_aggregate(im_preproc, ims_ref, os.path.join(channel_dir, stitch_stem), '_ch??', cfg['biasfield']['params']['postfix'])







im_preproc = os.path.join(datadir, '{}{}{}.ims'.format(dataset, cfg['shading']['params']['postfix'], cfg['stitching']['params']['postfix']))
im_preproc = os.path.join(datadir, '{}.h5/data'.format(dataset))
# blocks.split(im_preproc, par_file)
# enhance.estimate(im_preproc, par_file)  # NOTE: not for nucl-only
# segment.estimate(im_preproc, par_file, blocks=list(range(144, 396)))
#segment.estimate(im_preproc, par_file)
segment.estimate(im_preproc, par_file)

# [TODO: seg postproc]
zipping.relabel(im_preproc, par_file)
zipping.copyblocks(im_preproc, par_file)
zipping.estimate(im_preproc, par_file)
# zipping.relabel(im_preproc, par_file, ids='labels_memb')
# zipping.copyblocks(im_preproc, par_file, ids='labels_memb_relabeled')
# zipping.estimate(im_preproc, par_file, ids='labels_memb_relabeled_fix')

segment.subsegment(im_preproc, par_file)  # NOTE: not for nucl-only

blocks.merge(im_preproc, par_file)

features.estimate(im_preproc, par_file)
features.postproc(im_preproc, par_file)
