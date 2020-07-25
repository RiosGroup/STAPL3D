###==========================================================================###
### RUN PIPELINE
###==========================================================================###
### analysis preparation

source "${HOME}/.stapl3d.ini" && load_stapl3d_config

projectdir='/hpc/pmc_rios/Kidney'
dataset='200706_AP_P30T_LSR3D_25x_150um'

load_dataset "${projectdir}" "${dataset}"
[[ -f "${datadir}/${dataset}.yml" ]] || init_dataset
load_parameters "${dataset}" -v


###==========================================================================###
### preprocessing
jid=''
submit $( generate_script shading ) $jid
submit $( generate_script shading_postproc ) $jid
submit $( generate_script shading_apply ) $jid

submit $( generate_script stitching0 ) $jid
submit $( generate_script stitching1 ) $jid  # MACRO1 # TODO: voxel size automation
submit $( generate_script stitching2 ) $jid
submit $( generate_script stitching3 ) $jid
submit $( generate_script stitching4 ) $jid
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
