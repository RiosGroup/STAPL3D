###==========================================================================###
### RUN PIPELINE
# FIXME config load from clean
# TODO: write to specific blockdirs for different runs
###==========================================================================###
### analysis preparation

source "${HOME}/.stapl3d.ini" && load_stapl3d_config

projectdir='/hpc/pmc_rios/mkleinnijenhuis/Kidney'
projectdir='/hpc/pmc_rios/Kidney'
dataset='190910_rl57_fungi_16bit_25x_125um_corr-stitching'
# dataset='200706_AP_P30T_LSR3D_25x_150um'
# dataset='190921_P25T_RL57_FUnGI_16Bit_25x'
# dataset='200803_RL64_BCorganoids_25x'
# dataset='200924_RL57_HFK_multipleresolutions_63x_zstack2'
# dataset='200924_RL57_HFK_multipleresolutions_63x_zstack3'
# dataset='200929_RL68A_HFK_63x_zstack1'
dataset='200929_RL68A_HFK_63x'
dataset='mitochondria_airyscan_25x'
# projectdir='/hpc/pmc_rios/mkleinnijenhuis/Brain'
# dataset='20200717_MBR04_slice6_tilescan_zstack_25x'
load_dataset "${projectdir}" "${dataset}"

parfile="${dataset}.yml"
[[ -f "${parfile}" ]] || init_dataset
load_parameters "${dataset}" -v "${parfile}"


# ln -s /hpc/pmc_rios/Kidney/200706_AP_P30T_LSR3D_25x_150um/200706_AP_P30T_LSR3D_25x_150um.czi 200706_AP_P30T_LSR3D_25x_150um.czi
# ln -s /hpc/pmc_rios/Kidney/200706_AP_P30T_LSR3D_25x_150um/200706_AP_P30T_LSR3D_25x_150um_shading_stitching.ims 200706_AP_P30T_LSR3D_25x_150um_shading_stitching.ims
# ln -s /hpc/pmc_rios/Kidney/200706_AP_P30T_LSR3D_25x_150um/200706_AP_P30T_LSR3D_25x_150um_shading_stitching_biasfield.ims 200706_AP_P30T_LSR3D_25x_150um_shading_stitching_biasfield.ims
# ln -s /hpc/pmc_rios/Kidney/200706_AP_P30T_LSR3D_25x_150um/200706_AP_P30T_LSR3D_25x_150um_shading_stitching_mask.h5 200706_AP_P30T_LSR3D_25x_150um_shading_stitching_mask.h5
datadir_rvi='/hpc/pmc_rios/Kidney/200706_AP_P30T_LSR3D_25x_150um'
ln -s $datadir_rvi/${dataset}_shading_stitching_biasfield_segm-labels_memb_del_relabeled_fix_full.h5 $datadir/${dataset}_shading_stitching_biasfield_segm-labels_memb_del_relabeled_fix_full.h5
ln -s $datadir_rvi/${dataset}_shading_stitching_biasfield_segm-labels_memb_del_relabeled_fix_memb.h5 $datadir/${dataset}_shading_stitching_biasfield_segm-labels_memb_del_relabeled_fix_memb.h5
ln -s $datadir_rvi/${dataset}_shading_stitching_biasfield_segm-labels_memb_del_relabeled_fix_nucl.h5 $datadir/${dataset}_shading_stitching_biasfield_segm-labels_memb_del_relabeled_fix_nucl.h5
for blockstem in "${blockstems[@]}"; do
    ln -s /hpc/pmc_rios/Kidney/200706_AP_P30T_LSR3D_25x_150um/blocks/$blockstem.h5 /hpc/pmc_rios/mkleinnijenhuis/Kidney/200706_AP_P30T_LSR3D_25x_150um/blocks/$blockstem.h5
done


###==========================================================================###
### preprocessing
jid=''
submit $( generate_script shading ) $jid
submit $( generate_script shading_postproc ) $jid
submit $( generate_script shading_apply ) $jid

submit $( generate_script stitching_prep ) $jid
submit $( generate_script stitching_load ) $jid
submit $( generate_script stitching_calc ) $jid
submit $( generate_script stitching_fuse ) $jid
# load_parameters "${dataset}" -v

jid=''
submit $( generate_script mask ) $jid
# submit $( generate_script splitchannels ) $jid
# submit $( generate_script ims_aggregate1 ) $jid
submit $( generate_script biasfield ) $jid
submit $( generate_script biasfield_apply ) $jid
# submit $( generate_script biasfield_stack ) $jid
submit $( generate_script ims_aggregate2 ) $jid


###==========================================================================###
### segmentation
# submit $( generate_script block_segmentation ) $jid
submit $( generate_script splitblocks ) $jid
submit $( generate_script membrane_enhancement ) $jid
submit $( generate_script segmentation ) $jid
submit $( generate_script segmentation_postproc ) $jid
submit $( generate_script segmentation_gather ) $jid


###==========================================================================###
### zipping
submit $( generate_script relabel ) $jid
submit $( generate_script relabel_gather ) $jid

submit $( generate_script copyblocks ) $jid
submit $( generate_script copyblocks_gather ) $jid

stops_zyx=( $(((nx-1)*(ny-1))) $((ny-1)) $((nx-1)) )
for axis in 1 2; do
    stop="${stops_zyx[axis]}"
    for start in 0 1; do
        zipjob="ziplines${axis}${start}"
        submit $( generate_script "${zipjob}" ) $jid
        submit $( generate_script zipping_gather ) $jid
    done
done

for start_x in 0 1; do
    for start_y in 0 1; do
        zipjob="zipquads${start_x}${start_y}"
        submit $( generate_script "${zipjob}" ) $jid
        submit $( generate_script zipping_gather ) $jid
    done
done

submit $( generate_script zipping_postproc ) $jid


###==========================================================================###
### merging
submit $( generate_script subsegment ) $jid
submit $( generate_script mergeblocks ) $jid
# TODO: merge the idss?


###==========================================================================###
### features
submit $( generate_script features ) $jid
submit $( generate_script features_postproc ) $jid




###==========================================================================###
### stardist
submit $( generate_script stardist_train ) $jid
# stardist_normalization_range  # NB on bigbig-mem
# TODO get histogram from blocks
submit $( generate_script stardist_nblocks ) $jid
# FIXME: wait for it
# TODO: make blocklayout equal between stardist and stapl3d
nblocks=$(<$datadir/blocks_stardist/nblocks.txt)
submit $( generate_script stardist_predict ) $jid
submit $( generate_script stardist_gather ) $jid
submit $( generate_script stardist_mergeblocks ) $jid


###==========================================================================###
### plantseg
submit $( generate_script unet3d_memb_train ) $jid
submit $( generate_script unet3d_memb_predict ) $jid  # FIXME: model version error for models trained with multiple GPUs
# copy unet-model to ~/.plantseg_models/
submit $( generate_script plantseg_predict ) $jid
submit $( generate_script mergeblocks ) $jid


###==========================================================================###
### unet for nuclei mask
submit $( generate_script unet3d_nucl_train ) $jid
submit $( generate_script unet3d_nucl_predict ) $jid
submit $( generate_script mergeblocks ) $jid


###==========================================================================###
### ki67cleanup
jid=
# cp /hpc/pmc_rios/Kidney/190910_rl57_fungi_16bit_25x_125um_corr-stitching/190910_rl57_fungi_16bit_25x_125um_corr-stitching_02496-03904_12736-14144_00000-00106_ki67_fullBG_11feats.ilp ${datadir}/
submit $( generate_script splitblocks_ki67 ) $jid
submit $( generate_script apply_ilastik ) $jid
submit $( generate_script mergeblocks_ilastik ) $jid
