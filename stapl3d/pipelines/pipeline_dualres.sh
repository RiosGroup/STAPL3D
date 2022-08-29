source "${HOME}/.stapl3d.ini" && load_stapl3d_config

projectdir='/hpc/pmc_rios/mkleinnijenhuis/1.projects/TEST/NatProt_STAPL3D'
dataset='200929_RL68A_HFK'
parfile="${dataset}.yml"
jid=''

load_dataset "${projectdir}" "${dataset}"

[[ -f "${parfile}" ]] || init_dataset
dos2unix $parfile
load_parameters "${dataset}" -v "${parfile}"



## dual-res analysis
image_in="{f}.czi"
set_filepaths "*.czi"
submit $( generate_script splitter split ) $jid

submit $( generate_script coregistration estimate ) $jid

blockdir=${datadir}/blocks
submit $( generate_script unet3d_memb predict ) $jid

image_in="{f}_63x_zstack*.czi"
set_filepaths "*_63x_zstack*.czi"
submit $( generate_script segmentation estimate ) $jid

submit $( generate_script coregistration apply ) $jid

submit $( generate_script stardist train ) $jid
submit $( generate_script unet3d_memb train ) $jid
submit $( generate_script unet3d_nucl train ) $jid






jid=''
submit $( generate_script shading estimate ) $jid
submit $( generate_script shading postprocess ) $jid
submit $( generate_script shading apply ) $jid

submit $( generate_script stitching prep ) $jid
submit $( generate_script stitching load ) $jid
submit $( generate_script stitching calc ) $jid
submit $( generate_script stitching fuse ) $jid
submit $( generate_script stitching postprocess ) $jid

submit $( generate_script biasfield estimate ) $jid
submit $( generate_script biasfield apply) $jid
submit $( generate_script stitching postprocess ) $jid

submit $( generate_script splitter split ) $jid

submit $( generate_script stardist nblocks ) $jid  # TODO: stardistblocks to STAPL3D blocks
nblocks=$(<$datadir/blocks_stardist_RL2/nblocks.txt) && echo 'nblocks: ' $nblocks
submit $( generate_script stardist predict ) $jid
submit $( generate_script stardist merge ) $jid

blockdir=$datadir/blocks
submit $( generate_script unet3d_nucl predict ) $jid
submit $( generate_script unet3d_memb predict ) $jid

submit $( generate_script segmentation estimate ) $jid
submit $( generate_script segmentation postprocess ) $jid
submit $( generate_script segmentation gather ) $jid


# TODO
submit $( generate_script zipping relabel  ) $jid
#submit $( generate_script zipping gather ) $jid
submit $( generate_script zipping copyblocks ) $jid
submit $( generate_script zipping gather ) $jid

stops_zyx=( $(((nx-1)*(ny-1))) $((ny-1)) $((nx-1)) )
for axis in 1 2; do
    stop="${stops_zyx[axis]}"
    for start in 0 1; do
        zipjob="ziplines${axis}${start}"
        submit $( generate_script "${zipjob}" ) $jid
        submit $( generate_script zipping gather ) $jid
    done
done

for start_x in 0 1; do
    for start_y in 0 1; do
        zipjob="zipquads${start_x}${start_y}"
        submit $( generate_script "${zipjob}" ) $jid
        submit $( generate_script zipping gather ) $jid
    done
done

submit $( generate_script zipping postproc ) $jid


submit $( generate_script subsegmentation estimate ) $jid
submit $( generate_script merger merge ) $jid


###==========================================================================###
### features
submit $( generate_script features estimate ) $jid
submit $( generate_script features postprocess ) $jid
