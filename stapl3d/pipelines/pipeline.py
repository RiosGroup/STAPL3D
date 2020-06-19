###==========================================================================###
### z-stack shading correction
###==========================================================================###

# 1. open 'Anaconda Prompt' and start python by:
# >>>   conda activate segmentation
# >>>   ipython

# 2. imports and parameter deinitions
import os
import multiprocessing
from membrane.czi_split_zstacks import estimate_zstack_profiles, fit_profiles

kdir = 'E:\\Ravi'
dataset = '200221_RL57_HFK7_25x'
n_channels = 8
noise_thr = 1000
quant_thr = 0.8

datadir = os.path.join(kdir, dataset)
os.makedirs(os.path.join(datadir, 'bias'), exist_ok=True)
filename = '{}.czi'.format(dataset)
filepath = os.path.join(datadir, filename)

### 3a. calculate medians
arglist = [(filepath, ch, noise_thr) for ch in range(0, n_channels)]
with multiprocessing.Pool(processes=n_channels) as pool:
    pool.starmap(estimate_zstack_profiles, arglist)

### 3b: process profiles  # NOTE: wait for step 3a to finish
fit_profiles(filepath, n_channels=n_channels, quant_thr=quant_thr)




###==========================================================================###
### apply the shading profile, stitch, and convert to .ims
# NOTE: Zen Black equivalent to Zen Blue, but doesn't generate an image pyramid
###==========================================================================###

"""
1. shading correction
   - for each channel to be corrected:
        -- Utilities - Create image subset and split [MK_stitch_01_ch00 preset]
        -- Load shading image
        -- Adjust - Shading Correction [MK_stitch_02 preset]
        -- Utilities - Fuse Image Subset (20 min/ch) [MK_stitch_03]
     OR -- Utilities - Add Channels (6 min/ch -- 10 min/ch)
        -- TODO?: clipping mask after shading correction? [-- Utilities - Add Channels]
    -- Generate Image Pyramid
    -- Save


2. stitching
## NOTE: Zen is fing useless for this: stitches a 2D plane which may not be representative
-- NOT all individually
    -- Save

3. imaris conversion
    - Drag and drop <dataset>_corr-stitching.czi in Imaris File Converter

"""
"""
# extract single channel and saved in Imaris
# now fill with zeros
import os
import h5py
import numpy as np
permission = 'r+'
#infile = 'C:\\Users\i.research_pc\\mkleinnijenhuis\\190910_rl57_fungi_16bit_25x_125um_corr-stitching_bfc_ch00.ims'
kdir = 'E:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '190909_RL57_Ce3D_16Bit_25x_100um_corr_stitching'
dataset = '200221_RL57_HFK7_25x_corr_stitching_CROP1'
dataset = '200221_RL57_HFK7_25x_corr_stitching_CROP2'
infile = os.path.join(kdir, dataset, '{}_ref_uint16.ims'.format(dataset))
im = h5py.File(infile, permission)
channels = [0]
n_reslev = len(im['/DataSet'])
for ch in channels:
    for rl in range(n_reslev):
        tploc = '/DataSet/ResolutionLevel {}/TimePoint 0'.format(rl)
        chloc = '{}/Channel {}'.format(tploc, ch)
        dsloc = '{}/Data'.format(chloc)
        ds = im[dsloc]
        ds[:] = np.zeros(ds.shape, dtype=ds.dtype)
im.close()
# open in Imaris and "Save as" <>_ref_uint16.ims
# convert to uint8 in Imaris  and "Save as" <>_ref_uint8.ims
# convert to float32 in Imaris  and "Save as" <>_ref_float32.ims

then scratch_split_multichannel_ims.py
"""

###==========================================================================###
### inhomogeneity correction
###==========================================================================###

# 1. open 'Anaconda Prompt' and start python by:
# >>>   conda activate simpleitk
# >>>   ipython

### mask generation and bfc estimation
# NOTE: bfc estimation fails: see error in error_bfc-itk.txt NB: FIXED by upgrade to simpleitk2.0.0
# NOTE: conda install -c conda-forge pypdf2
import os
from glob import glob
import multiprocessing
from image import reporting
from image.generate_dataset_mask import generate_dataset_mask
from image.bias_field_correction import (
    bias_field_correction,
    stack_bias,
    apply_bias_field_full,
    )


kdir = 'D:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'  # [bfc was generated on HPC]
resolution_level = 4
downsample_factors = [1, 4, 4, 1, 1]
dsfacs = [1, 64, 64, 1]
sigma = 48.0
use_median_thresholds = True
abs_threshold = 1000

kdir = 'E:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '200221_RL57_HFK7_25x_corr_stitching_CROP1'
resolution_level = 1
downsample_factors = [1, 8, 8, 1, 1]
dsfacs = [1, 16, 16, 1]
sigma = 48.0
use_median_thresholds = True
abs_threshold = 1000

kdir = 'E:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '200221_RL57_HFK7_25x_corr_stitching_CROP1b'
resolution_level = 1
downsample_factors = [1, 8, 8, 1, 1]
dsfacs = [1, 16, 16, 1]
sigma = 48.0
use_median_thresholds = False
abs_threshold = 1800

kdir = 'E:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '200221_RL57_HFK7_25x_corr_stitching_CROP2'
resolution_level = 1
downsample_factors = [1, 8, 8, 1, 1]
dsfacs = [1, 16, 16, 1]
sigma = 48.0
use_median_thresholds = False
abs_threshold = 1800

kdir = 'E:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '190909_RL57_Ce3D_16Bit_25x_100um_corr_stitching'
resolution_level = 5
downsample_factors = [1, 2, 2, 1, 1]
dsfacs = [1, 64, 64, 1]
sigma = 48.0
use_median_thresholds = False
abs_threshold = 1000


datadir = os.path.join(kdir, dataset)
filestem = os.path.join(datadir, dataset)
image_in = '{}.ims'.format(filestem)
mask_stem = '{}_mask'.format(filestem)
bias_in = '{}_bfc.h5/bias'.format(filestem)

image_in = '{}.ims'.format(filestem)
mask_in = '{}.h5/mask_thr00000'.format(mask_stem)
n_iterations = 50
n_fitlevels = 4
n_bspline_cps = [5, 5, 5]
outputstem = '{}'.format(filestem)
save_steps = True
n_channels = 8

generate_dataset_mask(
    image_in, resolution_level,
    sigma, use_median_thresholds, abs_threshold,
    outputstem=mask_stem, save_steps=True,
    )

for channel in range(0, n_channels):
    bias_field_correction(
        image_in, mask_in,
        channel,
        resolution_level,
        downsample_factors,
        n_iterations, n_fitlevels, n_bspline_cps,
        outputstem, save_steps,
    )

# per channel apply: requires imaris split first (only done for CE3D)
in_place = True
write_to_single_file = False
blocksize_xy = 1280
outputpath = ''
channel = None
arglist = []
for ch in range(0, n_channels):
    image_in_ch = '{}_bfc_ch{:02d}.ims'.format(filestem, ch)
    bias_in_ch = '{}_bfc_ch{:02d}.h5/bias'.format(filestem, ch)
    args = (
        image_in_ch, bias_in_ch, dsfacs,
        in_place, write_to_single_file,
        blocksize_xy, outputpath, channel,
    )
    arglist.append(args)

with multiprocessing.Pool(processes=n_channels) as pool:
    pool.starmap(apply_bias_field_full, arglist)


stack_bias(filestem)
apply_bias_field_full(image_in, bias_in, dsfacs=dsfacs, in_place=True)

""" CROP2 failed for blocksize_xy=1280 and blocksize=1000
[1;31mTypeError[0m: Can't broadcast (20, 32, 4) -> (20, 32, 3)
# 4124 4123 327
# apply_bias_field_full(image_in, bias_in, dsfacs=dsfacs, in_place=True, blocksize_xy=0)
"""

# TODO: zips with full path; change to relative
import zipfile
zips = glob('{}_bfc_*-params.pickle'.format(filestem))
zips.sort()
outputpath = '{}_bfc-params.zip'.format(filestem)
zf = zipfile.ZipFile(outputpath, mode='w')
for pfile in zips:
    zf.write(pfile)
zf.close()

pdfs = glob('{}_bfc_*-report.pdf'.format(filestem))
pdfs.sort()
outputpath = '{}_bfc-report.pdf'.format(filestem)
reporting.merge_reports(pdfs, outputpath)

"""
arglist = []
for channel in range(0, n_channels):
    args = (
        image_in, mask_in,
        channel,
        resolution_level,
        downsample_factors,
        n_iterations, n_fitlevels, n_bspline_cps,
        outputstem, save_steps,
    )
    arglist.append(args)
# FIXME: multiprocessing may be removed in most cases of deployment on workstations; N4 already threaded
with multiprocessing.Pool(processes=1) as pool:
    pool.starmap(bias_field_correction, arglist)
"""

""" NOTE on blocksize
#blocksize = [im.dims[0], 5120, 5120, 1, 1]  # takes max 200GB of memory
#blocksize = [im.dims[0], 10240, 10240, 1, 1]  # TODO: check if I can make it this
#blocksize = [im.dims[0], 8192, 8192, 1, 1]  # TODO: or make it this
#blocksize = [im.dims[0], 8704, 8704, 1, 1]  # TODO: or make it this for creating 4 blocks of the largest dataset so far
"""


""" FOR checking mask carefully
from wmem.stack2stack import stack2stack
mask_in_nii = '{}-mask_thr00000.nii.gz'.format(mask_stem)
stack2stack(mask_in, inlayout='zyx', outlayout='xyz', outputpath=mask_in_nii)

mask_mean = '{}.h5/mean'.format(mask_stem)
mask_mean_nii = '{}-mean.nii.gz'.format(mask_stem)
stack2stack(mask_mean, inlayout='zyx', outlayout='xyz', outputpath=mask_mean_nii)

mask_smooth = '{}.h5/smooth'.format(mask_stem)
mask_smooth_nii = '{}-smooth.nii.gz'.format(mask_stem)
stack2stack(mask_smooth, inlayout='zyx', outlayout='xyz', outputpath=mask_smooth_nii)
"""


###==========================================================================###
### split into blocks and average channels
###==========================================================================###
from image.combine_vols import combine_vols
from wmem import wmeMPI, Image
import numpy as np
import multiprocessing

image_in = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching\\190910_rl57_fungi_16bit_25x_125um_corr-stitching_bfc.ims'
dataslices = None
bs = 1280
blockmargin = [0, 64, 64, 0, 0]
vol_idxs = [3, 5, 6, 7]
weights = [0.5, 0.5, 1, 1]
bias_image = ''
outputpath = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching\\blocks_1280\\190910_rl57_fungi_16bit_25x_125um_corr-stitching_{}.h5/ods'
save_steps = False,
protective = False,
usempi = False

# get the number of blocks
mpi = wmeMPI(usempi=False)
im = Image(image_in, permission='r')
im.load(load_data=False)
blocksize = im.dims
blocksize[1] = bs
blocksize[2] = bs
mpi.set_blocks(im, blocksize, blockmargin)
mpi.scatter_series()
im.close()

n_blocks = len(mpi.series)
n_proc = 12
jobsize = int(np.ceil(n_blocks / n_proc))
blockstarts = range(0, n_blocks, jobsize)


args = [
    image_in, dataslices,
    blocksize, blockmargin, 'insert_blockrange',
    vol_idxs,
    weights,
    bias_image,
    outputpath,
    save_steps, protective, usempi,
    ]

def get_arglist_blocks(args, blockstarts, jobsize=12):
    arglist = []
    br_idx = args.index('insert_blockrange')
    for startblock in blockstarts:
        stopblock = min(n_blocks, startblock + jobsize)
        blockrange = [startblock, stopblock]
        args[br_idx] = blockrange
        arglist.append(tuple(args))
    return arglist

arglist = get_arglist_blocks(args, blockstarts, jobsize)
with multiprocessing.Pool(processes=n_proc) as pool:
    pool.starmap(combine_vols, arglist)


###==========================================================================###
### membrane enhancement
###==========================================================================###
#cs_zyx=`get_chunksize "${datadir}/${dataset}.ims/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"`
#cs_xyz=`get_reversed_sentence "${cs_zyx}"`
#get_cmd_array_nii-to-h5 "${filestem}" 'memb/' 'preprocess' 'xyz' 'zyx' "-s ${cs_xyz}" >> "${qsubfile}"
#get_cmd_array_nii-to-h5 "${filestem}" 'memb/' 'planarity' 'xyz' 'zyx' "-s ${cs_xyz}" >> "${qsubfile}"
#get_cmd_array_memb-seg "${filestem}" 'memb/planarity' 'memb/sum' 'nucl/dapi' 'mean' -n -p "$dapi_shift" -d "$dapi_thr" "-s 7 21 21 -t 1.0 -S -o ${filestem}" >> "${qsubfile}"

### membrane enhancement (not running on Windows-64)
import os
import multiprocessing
from glob import glob
from wmem.stack2stack import stack2stack

n_proc = 12
kdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
datadir = os.path.join(kdir, dataset)
blockdir = os.path.join(datadir, 'blocks_1280')

ipf = ''
group = 'memb'
ids = 'sum'

filelist = glob(os.path.join(blockdir, '{}_*{}.h5'.format(dataset, ipf)))
filelist.sort()

image_in = ''
dataslices = None
dset_name = ''
blockoffset = []
datatype = 'uint8'
uint8conv = True
inlayout = 'zyx'
outlayout = 'xyz'
elsize = []
chunksize = []
outputpath = ''
save_steps = False
protective = False
usempi = False

args = [
    image_in,
    dataslices,
    dset_name,
    blockoffset,
    datatype,
    uint8conv,
    inlayout,
    outlayout,
    elsize,
    chunksize,
    outputpath,
    save_steps,
    protective,
    usempi,
]

arglist = []
for datafile in filelist:
    args[0] = '{}/{}/{}'.format(datafile, group, ids)
    args[10] = datafile.replace('.h5', '_{}-{}.nii.gz'.format(group, ids))
    arglist.append(tuple(args))

with multiprocessing.Pool(processes=n_proc) as pool:
    pool.starmap(stack2stack, arglist)




### ACME
"""
# FIXME: may try to compile
echo "\${MRbin}/cellPreprocess" \
    "\${filestem}_memb-sum.nii.gz" \
    "\${filestem}_memb-preprocess.nii.gz" \
    0.5
echo "\${MRbin}/multiscalePlateMeasureImageFilter" \
    "\${filestem}_memb-preprocess.nii.gz" \
    "\${filestem}_memb-planarity.nii.gz" \
    "\${filestem}_memb-eigen.mha" \
    1.1
# echo "\${MRbin}/membraneVotingField3D" \
#     "\${filestem}_memb-planarity.nii.gz" \
#     "\${filestem}_memb-eigen.mha" \
#     "\${filestem}_memb-TV.mha" \
#     1.0
# echo "\${MRbin}/membraneSegmentation" \
#     "\${filestem}_memb-preprocess.nii.gz" \
#     "\${filestem}_memb-TV.mha" \
#     "\${filestem}_memb-segment.mha" \
#     1.0
"""

from wmem import Image
image_in = '{}/memb/sum'.format(filelist[0])
im = Image(image_in, permission='r')
im.load(load_data=False)
chunksize = im.chunks[::-1]
im.close()

image_in = ''
dataslices = None
dset_name = ''
blockoffset = []
datatype = ''
uint8conv = False
inlayout = 'xyz'
outlayout = 'zyx'
elsize = []
chunksize = chunksize
outputpath = ''
save_steps = False
protective = False
usempi = False

args = [
    image_in,
    dataslices,
    dset_name,
    blockoffset,
    datatype,
    uint8conv,
    inlayout,
    outlayout,
    elsize,
    chunksize,
    outputpath,
    save_steps,
    protective,
    usempi,
]

group = 'memb'
idss = ['preprocess', 'planarity']

for ids in idss:

    arglist = []
    for datafile in filelist:
        args[0] = datafile.replace('.h5', '_{}-{}.nii.gz'.format(group, ids))
        args[10] = '{}/{}/{}'.format(datafile, group, ids)
        arglist.append(tuple(args))

    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.starmap(stack2stack, arglist)

# FIXME: error because wmem.Image return value
#[1;31mMaybeEncodingError[0m: Error sending result: '[<wmem.Image object at 0x00000181AE2CE2C8>, <wmem.Image object at 0x00000181AAB7AD88>, <wmem.Image object at 0x00000181AC60E5C8>, <wmem.Image object at 0x00000181AE212A48>]'. Reason: 'TypeError("can't pickle _thread._local objects")'




###==========================================================================###
### segmentation  ~9h on BRAIN (with n_proc=12; 182 blocks => 36min/block)
###==========================================================================###

import os
import multiprocessing
from glob import glob
from membrane.extract_segments import extract_segments_new as extract_segments

n_proc = 12
kdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
datadir = os.path.join(kdir, dataset)
datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
blockdir = os.path.join(datadir, 'blocks_1280')

ipf = ''
filelist = glob(os.path.join(blockdir, '{}_*{}.h5'.format(dataset, ipf)))
filelist.sort()

plan_path = ''
memb_path = ''
dapi_path = ''
mean_path = ''
dapi_shift_planes = 3  # NOTE: dataset specific
dapi_thr = 5000
peaks_size = [7, 21, 21]  # NOTE: dataset resolution specific
peaks_thr = 1.0
outputstem = ''
save_steps = True

args = [
    plan_path,
    memb_path,
    dapi_path,
    mean_path,
    dapi_shift_planes,
    dapi_thr,
    peaks_size,
    peaks_thr,
    outputstem,
    save_steps,
]

arglist = []
for datafile in filelist:
    args[0] = '{}/{}/{}'.format(datafile, 'memb', 'planarity')
    args[1] = '{}/{}/{}'.format(datafile, 'memb', 'sum')
    args[2] = '{}/{}/{}'.format(datafile, 'nucl', 'dapi')
    args[3] = '{}/{}'.format(datafile, 'mean')
    args[8] = datafile.replace('.h5', '')
    arglist.append(tuple(args))

with multiprocessing.Pool(processes=n_proc) as pool:
    pool.starmap(extract_segments, arglist)

# FIXME: [1;31mMaybeEncodingError[0m: Error sending result: '[<wmem.LabelImage object at 0x0000023F7C352B48>, <wmem.LabelImage object at 0x0000023F07264208>, <wmem.LabelImage object at 0x0000023F00E9B448>, <wmem.LabelImage object at 0x0000023F2084C288>]'. Reason: 'TypeError("can't pickle _thread._local objects")'


from image import reporting
pdfs = glob(os.path.join(blockdir, '{}_*{}_seg-report.pdf'.format(dataset, ipf)))
pdfs.sort()
outputpath = os.path.join(datadir, '{}_seg-report.pdf'.format(dataset))
reporting.merge_reports(pdfs, outputpath)


###==========================================================================###
### resgementation (~5min par relabel; ~15 min par copy; ~12h non-parl reseg; ~2.5h par reseg)
### TODO: finish and time parallel reseg
###==========================================================================###

prep = True
if prep:
    import os
    import multiprocessing
    from glob import glob
    from membrane.resegment_block_boundaries import (
        resegment_block_boundaries,
        relabel_parallel,
        delete_blocks_parallel,
        copy_blocks_parallel,
        get_maxlabels_from_attribute,
        )

    n_proc = 12
    kdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney'
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    datadir = os.path.join(kdir, dataset)
    datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
    blockdir = os.path.join(datadir, 'blocks_1280')

    ipf = ''
    filelist = glob(os.path.join(blockdir, '{}_*{}.h5'.format(dataset, ipf)))
    filelist.sort()

relabel = False
if relabel:
    maxlabelfile = os.path.join(blockdir, '{}_maxlabels.txt'.format(dataset))
    maxlabels = get_maxlabels_from_attribute(filelist, 'segm/labels_memb_del', maxlabelfile)

    image_in = ''
    block_idx = 0
    maxlabelfile = maxlabelfile
    pf='relabeled'

    args = [
        image_in,
        block_idx,
        maxlabelfile,
        pf,
    ]

    arglist = []
    for block_idx, datafile in enumerate(filelist):
        args[0] = '{}/{}/{}'.format(datafile, 'segm', 'labels_memb_del')
        args[1] = block_idx
        arglist.append(tuple(args))

    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.starmap(relabel_parallel, arglist)

copy_blocks = True
if copy_blocks:
    pf='relabeled'
    maxlabelfile = os.path.join(blockdir, '{}_maxlabels_{}.txt'.format(dataset, pf))
    maxlabels = get_maxlabels_from_attribute(filelist, 'segm/labels_memb_del_{}'.format(pf), maxlabelfile)

    image_in = ''
    block_idx = 0
    postfix='fix2'

    args = [
        image_in,
        block_idx,
        postfix,
    ]

    arglist = []
    for block_idx, datafile in enumerate(filelist):
        args[0] = '{}/{}/{}'.format(datafile, 'segm', 'labels_memb_del_relabeled')
        args[1] = block_idx
        args[2] = 'fix2'
        arglist.append(tuple(args))

    #with multiprocessing.Pool(processes=n_proc) as pool:
    #    pool.starmap(delete_blocks_parallel, arglist)

    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.starmap(copy_blocks_parallel, arglist)

resegment = True
if resegment:

    pf='relabeled_fix2'

    maxlabelfile = os.path.join(blockdir, '{}_maxlabels_{}.txt'.format(dataset, pf))
    maxlabels = get_maxlabels_from_attribute(filelist, 'segm/labels_memb_del_{}'.format(pf), maxlabelfile)
    print('maxlabs after copy', maxlabels)

    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    ipf = ''
    filelist = glob(os.path.join(blockdir, '{}_*{}.h5'.format(dataset, ipf)))
    filelist.sort()

    from wmem import Image
    image_in = '{}/{}'.format(filelist[0], 'mean')
    im = Image(image_in, permission='r')
    im.load(load_data=False)
    Z = im.dims[0]
    im.close()

    bs = 1280
    bm = 64

    images_in=['{}/{}/{}_{}'.format(datafile, 'segm', 'labels_memb_del', pf) for datafile in filelist]
    blocksize=[Z, bs, bs]
    blockmargin=[0, bm, bm]
    axis=0
    seamnumbers=[-1, -1, -1]
    mask_dataset=''
    relabel=False
    maxlabel=''
    in_place=True
    outputstem=os.path.join(blockdir, dataset)
    save_steps=False
    args = [
        images_in,
        blocksize,
        blockmargin,
        axis,
        seamnumbers,
        mask_dataset,
        relabel,
        maxlabel,
        in_place,
        outputstem,
        save_steps,
    ]


    def get_arglist(args, axis, starts, stops, steps):
        arglist = []
        for seam_y in range(starts[0], stops[0], steps[0]):
            for seam_x in range(starts[1], stops[1], steps[1]):  # single it for axes=[1,2]
                seamnumbers = [-1, seam_y, seam_x]
                args[3] = axis
                if axis == 0:
                    args[4] = [seamnumbers[d] if d != axis else -1 for d in [0, 1, 2]]
                else:
                    args[4] = [seam_y if d == axis else -1 for d in [0, 1, 2]]
                arglist.append(tuple(args))
        return arglist

    def reseg_par(args, axis, seamgrid, starts, stops, steps, n_proc):
        arglist = get_arglist(args, axis, starts, stops, steps)
        print('submitting {:3d} jobs over {:3d} processes'.format(len(arglist), n_proc))
        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(resegment_block_boundaries, arglist)

    import numpy as np
    n_proc_max = 12
    n_seams_yx=[13, 12]
    seams = list(range(np.prod(n_seams_yx)))
    seamgrid = np.reshape(seams, n_seams_yx)

    # resegment seamlines in 4 steps: axis x / y and even / odd lines
    for axis, n_seams in zip([1, 2], n_seams_yx):
        n_proc = min(n_proc_max, int(np.ceil(n_seams / 2)))
        for offset in [0, 1]:
            reseg_par(args, axis, seamgrid, starts=[offset, 0], stops=[n_seams, 1], steps=[2, 2], n_proc=n_proc)

    # resegment seamquads in 4 groups even/even, even/odd, odd/even, odd/odd seamline intersections
    nproc = 12
    for start_y in [0, 1]:
        for start_x in [0, 1]:
            reseg_par(args, axis=0, seamgrid=seamgrid, starts=[start_y, start_x], stops=n_seams_yx, steps=[2, 2], n_proc=n_proc)


###==========================================================================###
# non-parallelized
"""
from wmem import Image
image_in = '{}/{}'.format(filelist[0], 'mean')
im = Image(image_in, permission='r')
im.load(load_data=False)
Z = im.dims[0]
im.close()
bs = 1280
bm = 64
pf='relabeled_fix'
resegment_block_boundaries(
    images_in=['{}/{}/{}_{}'.format(datafile, 'segm', 'labels_memb_del', pf)
               for datafile in filelist],
    blocksize=[Z, bs, bs],
    blockmargin=[0, bm, bm],
    mask_dataset='',
    relabel=False,
    maxlabel=maxlabels[-1],
    in_place=True,
    outputstem=os.path.join(blockdir, dataset),
    save_steps=False,
)
"""
###==========================================================================###


from image import reporting
pdfs = glob(os.path.join(blockdir, '{}_reseg_*-report.pdf'.format(dataset)))
pdfs.sort()
outputpath = os.path.join(datadir, '{}_reseg-report.pdf'.format(dataset))
reporting.merge_reports(pdfs, outputpath)




###==========================================================================###
### mergeblocks SCRATCH  ~30 min on BRAIN
###==========================================================================###

import os
import multiprocessing
from glob import glob
from membrane.extract_segments import extract_segments_new as extract_segments
from wmem.mergeblocks import mergeblocks
from wmem import Image

n_proc = 12
#kdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney'
dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
#datadir = os.path.join(kdir, dataset)
datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
blockdir = os.path.join(datadir, 'blocks_1280')

ipf = ''
filelist = glob(os.path.join(blockdir, '{}_*{}.h5'.format(dataset, ipf)))
filelist.sort()

image_in = os.path.join(datadir, '{}_bfc.ims'.format(dataset))
#image_in = '{}/{}'.format(filelist[0], 'mean')
im = Image(image_in, permission='r')
im.load(load_data=False)
Z = im.dims[0]
fullsize = im.dims[:3]
im.close()
bs = 1280
bm = 64

ids = 'segm/labels_memb'
ids = 'segm/labels_memb_del'
ids = 'segm/labels_memb_del_relabeled'
ids = 'segm/labels_memb_del_relabeled_fix'
ids = 'segm/labels_memb_del_relabeled_fix2'
images_in=['{}/{}'.format(datafile, ids) for datafile in filelist]
mergeblocks(
    images_in=images_in,
    outputpath=os.path.join(datadir, '{}.h5/{}'.format(dataset, ids)),
    blocksize=[Z, bs, bs],
    blockmargin=[0, bm, bm],
    fullsize=fullsize,
)


###==========================================================================###
### feature extraction  (9h for n_proc=10 with 182 blocks => 30 min / block)
# TODO: randomized mpi.series [instead of blockrange] for efieciency
### taken from combine_vols TODO: remove duplicate code
###==========================================================================###
import os
import multiprocessing
from glob import glob
from membrane.extract_segments import extract_segments_new as extract_segments
from membrane.export_regionprops import export_regionprops, postprocess_features
from wmem.mergeblocks import mergeblocks
from wmem import Image, wmeMPI
import numpy as np

# test BRAIN relabeled: with [GOOD?] fix
datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
datadir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
ids = 'segm/labels_memb_del_relabeled_fix'

seg_path = os.path.join(datadir, '{}.h5/{}'.format(dataset, ids))
featdir = os.path.join(datadir, 'profiling', 'features')

if False:
    # test BRAIN relabeled: with [GOOD?] fix
    datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    ids = 'segm/labels_memb_del_relabeled_fix'
    seg_path = os.path.join(datadir, '{}.h5/{}'.format(dataset, ids))
    featdir = os.path.join(datadir, 'BRAIN_relabeled_fix_new')

    # test BRAIN relabeled: with [BAD?] fix => resegmentation reports indicates erroneous segments on the block boundaries
    datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    ids = 'segm/labels_memb_del_relabeled_fix'
    seg_path = os.path.join(datadir, '{}.h5/{}'.format(dataset, ids))
    featdir = os.path.join(datadir, 'BRAIN_relabeled_fix')

    # orig HPC relabeled: no fix
    datadir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    ids = 'segm/labels_memb_del_relabeled'
    seg_path = os.path.join(datadir, '{}_testmerge_segm3.h5/{}'.format(dataset, ids))
    featdir = os.path.join(datadir, 'HPC_relabeled')

    # orig HPC relabeled: with [BAD?] fix => dist_to_edge backproject shows erroneous values on the block boundaries
    datadir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    ids = 'segm/labels_memb_del_relabeled_fix'
    seg_path = os.path.join(datadir, '{}.h5/{}'.format(dataset, ids))
    featdir = os.path.join(datadir, 'HPC_relabeled_fix')


os.makedirs(featdir, exist_ok=True)
outputstem = os.path.join(featdir, dataset)
data_path = os.path.join(datadir, '{}_bfc.ims'.format(dataset))
aux_data_path = os.path.join(datadir, '{}_mask.h5/mask_thr00000_edt'.format(dataset))
bs = 1280
bm = 64

seg_paths=[seg_path]
data_path=data_path
aux_data_path=aux_data_path
downsample_factors=[1, 16, 16]
bias_path=''
outputstem=outputstem
blocksize=[106, bs, bs, 8, 1]
blockmargin=[0, bm, bm, 0, 0]
blockrange=[]
channels=[]
filter_borderlabels=True
min_labelsize=50
split_features=True
fset_morph='maximal'
fset_int='maximal'

# get the number of blocks
mpi = wmeMPI(usempi=False)
im = Image(data_path, permission='r')
im.load(load_data=False)
blocksize = im.dims
blocksize[1] = bs
blocksize[2] = bs
mpi.set_blocks(im, blocksize, blockmargin)
mpi.scatter_series()
im.close()

n_blocks = len(mpi.series)
n_proc = 12
jobsize = int(np.ceil(n_blocks / n_proc))
blockstarts = range(0, n_blocks, jobsize)


args = [
    seg_paths,
    data_path,
    aux_data_path,
    downsample_factors,
    bias_path,
    outputstem,
    blocksize,
    blockmargin,
    'insert_blockrange',
    channels,
    filter_borderlabels,
    min_labelsize,
    split_features,
    fset_morph,
    fset_int,
    ]

def get_arglist_blocks(args, blockstarts, jobsize=12):
    arglist = []
    br_idx = args.index('insert_blockrange')
    for startblock in blockstarts:
        stopblock = min(n_blocks, startblock + jobsize)
        blockrange = [startblock, stopblock]
        args[br_idx] = blockrange
        arglist.append(tuple(args))

    return arglist

arglist = get_arglist_blocks(args, blockstarts, jobsize)

with multiprocessing.Pool(processes=n_proc) as pool:
    pool.starmap(export_regionprops, arglist)


postprocess_features(
    seg_paths,
    blocksize=blocksize,
    blockmargin=blockmargin,
    blockrange=[],
    csv_dir=featdir,
    csv_stem=dataset,
    feat_pf='_features',
    segm_pfs=['full', 'memb', 'nucl'],
    ext='csv',
    save_border_labels=True,
    split_features=True,
    fset_morph=fset_morph,
    fset_intens=fset_int,
    )





###==========================================================================###
### backproject
###==========================================================================###
### merged-blocks to imarisfile
import os
import h5py
import numpy as np
import pandas as pd
from wmem import Image, LabelImage
from image import backproject

datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
postfix = 'features'
featdir = os.path.join(datadir, postfix)
dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
group = 'segm/'
ids = 'labels_memb_del_relabeled_fix'
outdir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'

if False:
    datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
    postfix = 'BRAIN_relabeled_fix_new'
    featdir = os.path.join(datadir, postfix)
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    group = 'segm/'
    ids = 'labels_memb_del_relabeled_fix'
    outdir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'

    datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis'
    postfix = 'BRAIN_relabeled_fix'
    featdir = os.path.join(datadir, postfix)
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    group = 'segm/'
    ids = 'labels_memb_del_relabeled_fix'
    outdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'

    datadir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    postfix = 'HPC_relabeled'
    featdir = os.path.join(datadir, postfix)
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    group = 'segm/'
    ids = 'labels_memb_del_relabeled'

    datadir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    postfix = 'HPC_relabeled_fix'
    featdir = os.path.join(datadir, postfix)
    dataset = '190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    group = 'segm/'
    ids = 'labels_memb_del_relabeled_fix'
    outdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'

lowres = False
if lowres:
    edir = 'E:\\mkleinnijenhuis\\PMCdata\\Kidney'
    seg_path = os.path.join(edir, '{}_RL4_seg_{}.nii.gz'.format(dataset, postfix))
else:
    seg_path = os.path.join(datadir, '{}.h5/{}{}'.format(dataset, group, ids))
labels = LabelImage(seg_path)
labels.load(load_data=False)

ml_method = 'csv'
if ml_method == 'ulabels':
    filestem = os.path.join(datadir, dataset)
    ulabelfile = '{}_{}_ulabels.npy'.format(filestem, ids)
    #labels.set_maxlabel(); np.save(ulabelfile, labels.ulabels);
    ulabels = np.load(ulabelfile)
    ulabels = np.delete(ulabels, 0)
    maxlabel = np.amax(ulabels)
elif ml_method == 'attrs':
    # TODO: should be able to get this maxlabel easily real quick with
    def get_maxlabel(blockdir, dataset, ipf='', ids='labels'):
        from glob import glob
        from membrane.resegment_block_boundaries import get_maxlabels_from_attribute
        filelist = glob(os.path.join(blockdir, '{}_*{}.h5'.format(dataset, ipf)))
        filelist.sort()
        maxlabels = get_maxlabels_from_attribute(filelist, ids, '')
        return max(maxlabels)
    blockdir = os.path.join(datadir, 'blocks_1280')
    maxlabel = get_maxlabel(blockdir, dataset, ipf='', ids='segm/labels_memb_del_relabeled_fix')
elif ml_method == 'csv':
    datadir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching'
    profdir = os.path.join(datadir, 'profiling')
    profstem = os.path.join(profdir, dataset)
    csv_path = '{}{}.csv'.format(profstem, '_features')
    df = pd.read_csv(csv_path)
    maxlabel = df['label'].max()
    # FIXME: csv-maxlabel is different from adata-maxlabel!!


csv_path = os.path.join(featdir, '{}_features.csv'.format(dataset))
labelkey = 'label'
key = 'dist_to_edge'; normalize = False; scale_uint16 = False; replace_nan = True;
key = 'block'; normalize = False; scale_uint16 = False; replace_nan = True;
key = 'label'; normalize = True; scale_uint16 = True; replace_nan = True;
key = 'volume'; normalize = True; scale_uint16 = True; replace_nan = True;
key = 'extent'
#key = 'six2'; normalize = False; scale_uint16 = False; replace_nan = True;
#key = 'dapi'; normalize = False; scale_uint16 = False; replace_nan = True;
key = 'fractional_anisotropy'; normalize = False; scale_uint16 = True; replace_nan = True;

name = key

import shutil
ref_ims = 'D:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching\\190910_rl57_fungi_16bit_25x_125um_corr-stitching_ref_uint16.ims'
outpath = os.path.join(datadir, '{}_bp_{}.ims'.format(dataset, name))
shutil.copy2(ref_ims, outpath)


keys = ['dapi', 'ki67', 'pax8', 'ncam1', 'cadh1', 'cadh6', 'factin']; normalize = False; scale_uint16 = False; replace_nan = True;
keys = [key]
for key in keys:
    name = '{}'.format(key)
    outpath = os.path.join(datadir, '{}_bp_{}.ims'.format(dataset, name))
    shutil.copy2(ref_ims, outpath)
    backproject.csv_to_im(
        labels,
        csv_path,
        labelkey,
        key,
        name,
        maxlabel,
        normalize,
        scale_uint16,
        replace_nan,
        channel=-1,
        outpath=outpath,
    )




###### backproject a umap series
#if seg_path.endswith('.nii.gz'):
#    outpath = os.path.join(featdir, '{}_{}.nii.gz'.format(dataset, key))
#else:
#    outpath = os.path.join(outdir, '{}_bp.ims'.format(dataset))

testdir = 'G:\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching\\profiling\\test'
testdir = 'C:\\Users\\i.research_pc\\mkleinnijenhuis\\PMCdata\\Kidney\\190910_rl57_fungi_16bit_25x_125um_corr-stitching\\profiling\\test'
pfs = ['nmnone_ms05_nn30_nc3', 'nmscaled_ms05_nn30_nc3', 'nmlogscaled_ms05_nn30_nc3'] # see pipeline_scanpy.py
keys = ['leiden-0.40', 'leiden-0.80']; normalize = False; scale_uint16 = False; replace_nan = True;
keys = ['dpt_pseudotime']; normalize = False; scale_uint16 = True; replace_nan = True;
for pf in pfs:
    for key in keys:
        name = '{}_{}'.format(pf, key)
        csv_path = os.path.join(testdir, '{}_{}.csv'.format(dataset, name))
        outpath = os.path.join(datadir, '{}_bp_{}.ims'.format(dataset, name))
        shutil.copy2(ref_ims, outpath)
        labelkey = 'Id' # TODO: in the same format and as one csv

        backproject.csv_to_im(
            labels,
            csv_path,
            labelkey,
            key,
            name,
            maxlabel,
            normalize,
            scale_uint16,
            replace_nan,
            channel=-1,
            outpath=outpath,
        )

labels.close()
