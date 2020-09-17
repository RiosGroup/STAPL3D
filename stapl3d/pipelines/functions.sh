#!/bin/bash


###==========================================================================###
### functions to prepare processing
###==========================================================================###

function load_dataset {

    local projectdir="${1}"
    local dataset="${2}"

    set_datadir "${projectdir}" "${dataset}"

    echo ""
    echo " ### project directory is '${projectdir}'"
    echo " ### processing dataset '${dataset}'"
    echo " ### data directory is '${datadir}'"
    echo ""

}


function set_datadir {

    local projectdir="${1}"
    local dataset="${2}"

    mkdir -p "${projectdir}"

    datadir="${projectdir}/${dataset}"
    mkdir -p "${datadir}" && cd "${datadir}"

}


function init_dataset {

    cp "${STAPL3D}/pipelines/params.yml" "${datadir}/${dataset}.yml"

    echo ""
    echo " ### copied default parameter file to ${datadir}/${dataset}.yml"
    echo ""

    # set_ZYXCT_ims -v "${datadir}/${dataset}.ims"
    # write_ZYXCT_to_yml ${dataset} "${datadir}/${dataset}_dims.yml"

}


function load_parameters {

    local dataset="${1}"
    local verbose="${2}"
    local parfile="${3}"

    [[ -z $parfile ]] && parfile="${datadir}/${dataset}.yml"
    eval $( parse_yaml "${parfile}" "" )

    [[ "${verbose}" == '-v' ]] && {
        echo ""
        echo " ### parameters imported from parameter file:"
        echo ""
        parse_yaml "${parfile}"
        echo "" ; }

    set_dirtree "${datadir}"

    dataset_shading="${dataset}${shading__params__postfix}"
    dataset_stitching="${dataset_shading}${stitching__params__postfix}"
    dataset_biasfield="${dataset_stitching}${biasfield__params__postfix}"
    dataset_preproc="${dataset_biasfield}"

    check_dims M "$M" || set_dims_tiled "${datadir}/${dataset}_dims_tiled.yml"
    if ! [ -z "$M" ]
    then
        echo ""
        echo " ### tiled dimensions are M * Z-tY-tX-C-T='${M} * ${Z} ${tY} ${tX} ${C} ${T}'"
        echo " ### parallelization: ${M} Z-stacks"
        echo ""
    else
        echo ""
        echo " ### WARNING: could not determine tilescan dimensions"
        echo " ### please check your configuration"
        echo ""
    fi

    check_dims X "$X" || set_ZYXCT "${datadir}/${dataset}_dims_stitched.yml"
    if ! [ -z "$X" ]
    then
        bs="${dataset__blocksize_xy}" && check_dims bs "$bs" || set_blocksize
        bm="${dataset__blockmargin_xy}" && check_dims bm "$bm" || bm=64
        set_blocks "${bs}" "${bm}"
        echo ""
        echo " ### stitched dimensions are ZYXCT='${Z} ${Y} ${X} ${C} ${T}'"
        echo " ### parallelization: ${#blockstems[@]} blocks (${nx} x ${ny}) of blocksize ${bs} with margin ${bm}"
        echo ""
    else
        echo ""
        echo " ### WARNING: could not determine stitched dimensions"
        echo " ### please check your configuration"
        echo ""
    fi

    if ! [ -z "$C" ]
    then
        set_channelstems "${dataset_preproc}"
        echo ""
        echo " ### parallelization: ${#channelstems[@]} channels"
        echo ""
    else
        echo ""
        echo " ### WARNING: could not determine number of channels"
        echo " ### please check your configuration"
        echo ""
    fi

}


function parse_yaml {
    # https://gist.github.com/pkuczynski/8665367
    # consider: https://github.com/0k/shyaml

    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
         }
     }'

}


function set_dims_tiled {

    if ! [ -z "${dataset__M}" ]
    then
        # TODO: check all individually (at the end)
        # [[ -z "${dataset__M}" ]] || M="${dataset__M}"
        Z="${dataset__Z}"
        Z="${dataset__Z}"
        tY="${dataset__tY}"
        tX="${dataset__tX}"
        C="${dataset__C}"
        T="${dataset__T}"
        M="${dataset__M}"
        elsize_z="${dataset__elsize_z}"
        elsize_y="${dataset__elsize_y}"
        elsize_x="${dataset__elsize_x}"
    elif [ -f ${1} ]
    then
        eval $( parse_yaml "${1}" "" )
    else
        fpath="${datadir}/${dataset}.${dataset__file_format}"
        [[ -f "${fpath}" ]] && get_tiled_dims "${fpath}" && get_elsizes "${fpath}"
        write_tiled_to_yml "${1}"
    fi

}
function set_ZYXCT {

    if ! [ -z "${dataset__X}" ]
    then
        # TODO: check all individually (at the end)
        # [[ -z "${dataset__X}" ]] || M="${dataset__X}"
        Z="${dataset__Z}"
        Y="${dataset__Y}"
        X="${dataset__X}"
        C="${dataset__C}"
        T="${dataset__T}"
    elif [ -f ${1} ]
    then
        eval $( parse_yaml "${1}" "" )
    elif [ -f "${datadir}/${dataset}.ims" ]
    then
        set_ZYXCT_ims '-v' "${datadir}/${dataset}.ims"
        write_ZYXCT_to_yml "${1}"
    elif [ -f "${datadir}/${dataset_stitching}.ims" ]
    then
        set_ZYXCT_ims '-v' "${datadir}/${dataset_stitching}.ims"
        write_ZYXCT_to_yml "${1}"
    fi

}


function write_tiled_to_yml {

    local parfile="${1}"

    echo "Z: ${Z}" > "${parfile}"
    echo "tY: ${tY}" >> "${parfile}"
    echo "tX: ${tX}" >> "${parfile}"
    echo "C: ${C}" >> "${parfile}"
    echo "T: ${T}" >> "${parfile}"
    echo "M: ${M}" >> "${parfile}"
    echo "elsize_z: ${elsize_z}" >> "${parfile}"
    echo "elsize_y: ${elsize_y}" >> "${parfile}"
    echo "elsize_x: ${elsize_x}" >> "${parfile}"

    echo " ### written M-Z-tY-tX-C-T='${M} ${Z} ${tY} ${tX} ${C} ${T}' to ${parfile}"
    echo " ### written elsize_zyx='${elsize_z} ${elsize_y} ${telsize_x}' to ${parfile}"

}
function write_ZYXCT_to_yml {

    local parfile="${1}"

    echo "Z: ${Z}" > "${parfile}"
    echo "Y: ${Y}" >> "${parfile}"
    echo "X: ${X}" >> "${parfile}"
    echo "C: ${C}" >> "${parfile}"
    echo "T: ${T}" >> "${parfile}"

    echo " ### written Z-Y-X-C-T='${Z} ${Y} ${X} ${C} ${T}' to ${parfile}"

}


function get_tiled_dims {

    local filepath="${1}"

    source "${CONDA_SH}"
    conda activate ${submit_defaults__submit__conda_env}
    tiled_dims=`python -c "import os; from stapl3d.preprocessing import shading; iminfo = shading.get_image_info('${filepath}'); print('M={};C={};Z={};tY={};tX={};T={};'.format(int(iminfo['nstacks']), int(iminfo['nchannels']), int(iminfo['nplanes']), int(iminfo['ncols']), int(iminfo['nrows']), int(iminfo['ntimepoints'])));"`
    eval $tiled_dims
    conda deactivate

}


function get_elsizes {

    local filepath="${1}"

    source "${CONDA_SH}"
    conda activate ${submit_defaults__submit__conda_env}
    elsize_xyz=`python -c "import os; from stapl3d.preprocessing import shading; iminfo = shading.get_image_info('${filepath}'); print('elsize_x={};elsize_y={};elsize_z={}'.format(float(iminfo['elsize_zyxc'][2]), float(iminfo['elsize_zyxc'][1]), float(iminfo['elsize_zyxc'][0])));"`
    eval $elsize_xyz
    conda deactivate

}


function set_dirtree {

    local datadir="${1}"

    stackdir="${datadir}/${dirtree__datadir__stacks}"
    mkdir -p "${stackdir}"

    blockdir="${datadir}/${dirtree__datadir__blocks}"
    mkdir -p "${blockdir}"

    channeldir="${datadir}/${dirtree__datadir__channels}"
    mkdir -p "${channeldir}"

    shadingdir="${datadir}/${dirtree__datadir__shading}"
    mkdir -p "${shadingdir}"

    stitchingdir="${datadir}/${dirtree__datadir__stitching}"
    mkdir -p "${stitchingdir}"

    biasfielddir="${datadir}/${dirtree__datadir__biasfield}"
    mkdir -p "${biasfielddir}"

    profdir="${datadir}/${dirtree__datadir__profiling}"
    mkdir -p "${profdir}"

    featdir="${datadir}/${dirtree__datadir__blocks}"
    mkdir -p "${featdir}"

    jobdir="${datadir}/${dirtree__datadir__jobfiles}"
    mkdir -p "${jobdir}"

}


function set_blocksize {

    # TODO: determine from $Z
    bs=640

}


function set_blocks {

    local bs="${1}"
    local bm="${2}"

    blocksize="$Z $bs $bs $C $T"
    blockmargin="0 $bm $bm 0 0"
    set_blockdims
    set_blockstems "${dataset_preproc}"

}


function set_ZYXCT_ims {

    local verbose=$1
    shift 1
    local imarisfile=${*}

    Z=`get_dim_ims Z "${imarisfile}"`
    Y=`get_dim_ims Y "${imarisfile}"`
    X=`get_dim_ims X "${imarisfile}"`
    C=`get_dim_ims C "${imarisfile}"`
    T=`get_dim_ims T "${imarisfile}"`
    if [ "$verbose" == "-v" ]; then
        echo $Z $Y $X $C $T
    fi


}


function get_dim_ims {

    local dim=${1}
    shift 1
    local imarisfile="${*}"

    if [[ "$dim" == 'T' ]]; then
        echo `h5ls "${imarisfile}/DataSet/ResolutionLevel 0" | wc -l`
    elif [[ "$dim" == 'C' ]]; then
        echo `h5ls "${imarisfile}/DataSet/ResolutionLevel 0/TimePoint 0" | wc -l`
    else
        echo `find_dim_from_h5 "${dim}" "${imarisfile}"`
    fi

}


function find_dim_from_h5 {

    local dim=${1}
    shift 1
    local imarisfile="${*}"

    local line=`h5ls -v ${imarisfile}/DataSetInfo | grep -A 2 "Attribute: ${dim}"  | tail -n 1 | grep "Data" | sed 's/ //g'`
    if [[ "$line" == "Data:" ]]; then
        line=`h5ls -v ${imarisfile}/DataSetInfo | grep -A 3 "Attribute: ${dim}" | tail -n 1  | sed 's/(0)//'`
    else
        line=`h5ls -v ${imarisfile}/DataSetInfo | grep -A 2 "Attribute: ${dim}" | tail -n 1 | tr -d 'Data:'`
    fi
    echo "$line" | tr -d ' ' | tr -d , | tr -d '"'

}


function set_blockdims {

    nx=`python -c "from math import ceil; print(int(ceil($X/$bs)))"`
    ny=`python -c "from math import ceil; print(int(ceil($Y/$bs)))"`
    nb=$((nx*ny))

}


function set_blockstems {
    # Generate an array <blockstems> of block identifiers.
    # taking the form "dataset_x-X_y-Y_z-Z"
    # with voxel coordinates zero-padded to 5 digits

    local dataset=$1
    local verbose=$2
    local bx bX by bY bz bZ
    local dstem

    unset block_ids
    block_ids=()

    unset blockstems
    blockstems=()

    for x in `seq 0 $bs $(( X-1 ))`; do
        bX=$( get_coords_upper $x $bm $bs $X)
        bx=$( get_coords_lower $x $bm )
        for y in `seq 0 $bs $(( Y-1 ))`; do
            bY=$( get_coords_upper $y $bm $bs $Y)
            by=$( get_coords_lower $y $bm )
            for z in `seq 0 $Z $(( Z-1 ))`; do
                bZ=$( get_coords_upper $z 0 $Z $Z)
                bz=$( get_coords_lower $z 0 )

                block_id="$( get_block_id $bx $bX $by $bY $bz $bZ )"
                block_ids+=( "$block_id" )
                if [ "$verbose" == "-v" ]; then
                    echo "$block_id"
                fi

                dstem="${dataset}_${block_id}"
                blockstems+=( "$dstem" )
                if [ "$verbose" == "-v" ]; then
                    echo "$dstem"
                fi

            done
        done
    done

}


function set_blockstems_stardist {
    unset blockstems
    blockstems=()
    for block_id in `seq -f "%05g" 0 ${nblocks}`; do
        blockstems+=( ${dataset}_block$block_id )
    done
    datastems=( "${blockstems[@]}" )
}


function get_coords_upper {
    # Get upper coordinate of the block.
    # Adds the blocksize and margin to lower coordinate,
    # and picks between that value and max extent of the dimension,
    # whichever is lower

    local co=$1
    local margin=$2
    local size=$3
    local comax=$4
    local CO

    CO=$(( co + size + margin )) &&
        CO=$(( CO < comax ? CO : comax ))

    echo "$CO"

}


function get_coords_lower {
    # Get lower coordinate of the block.
    # Subtracts the margin,
    # and picks between that value and 0,
    # whichever is higher

    local co=$1
    local margin=$2

    co=$(( co - margin )) &&
        co=$(( co > 0 ? co : 0 ))

    echo "$co"

}


function get_block_id {
    # Get a block identifier from coordinates.

    local x=$1
    local X=$2
    local y=$3
    local Y=$4
    local z=$5
    local Z=$6

    local xrange=`printf %05d $x`-`printf %05d $X`
    local yrange=`printf %05d $y`-`printf %05d $Y`
    local zrange=`printf %05d $z`-`printf %05d $Z`

    local block_id=${xrange}_${yrange}_${zrange}

    echo "$block_id"

}


function set_channelstems {

    local dataset="$1"

    unset channelstems
    channelstems=()

    for c in `seq 0 $(( C - 1 ))`; do
        channelstems+=( "${dataset}_ch`printf %02d $c`" )
    done

}


###==========================================================================###
### some utils
###==========================================================================###

function check_dims {

    re='^[0-9]+$'
    if [[ -z $2 ]] ; then
        [[ -z $3 ]] || echo "dim $1 not set" >&2
        return 1
    elif ! [[ $2 =~ $re ]] ; then
        [[ -z $3 ]] || echo "invalid dim for $1" >&2
        return 1
    else
        return 0
    fi

}


# function get_chunksize {
#
#     echo `h5ls -v "${1}" |
#         grep "Chunks" |
#         grep -o -P '(?<={).*(?=})' |
#         tr -d ,`
#
# }


# function get_reversed_sentence {
#
#     echo "${1}" | awk '{ for (i=NF; i>1; i--) printf("%s ",$i); print $1; }'
#
# }


function set_images_in {

    local filestem="${1}"
    local ids="${2}"
    local verbose="${3}"

    images_in=()
    for fname in `ls ${filestem}_?????-?????_?????-?????_?????-?????.h5`; do
        images_in+=( ${fname}/$ids )
    done
    if [ "${verbose}" == '-v' ]
    then
        for image_in in "${images_in[@]}"; do
            echo ${image_in}
        done
    fi

}

# TODO: make the regex an argument
function set_images_in_stardist {
    local filestem="${1}"
    local ids="${2}"
    local verbose="${3}"
    images_in=()
    for fname in `ls ${filestem}_block?????.h5`; do
        images_in+=( ${fname}/$ids )
    done
    if [ "${verbose}" == '-v' ]
    then
        for image_in in "${images_in[@]}"; do
            echo ${image_in}
        done
    fi
}
function find_missing_datastems {
    local datadir=$1
    local postfix=$2
    local ext=$3
    unset missing
    declare -a missing
    for datastem in "${datastems[@]}"; do
        [ -f "$datadir$datastem.$ext" ] ||
            { missing+=( "$datastem" ); echo $datastem ; }
    done
    datastems=( "${missing[@]}" )
}


function set_channels_in {

    local filestem="${1}"

    channels_in=( )
    for fname in `ls ${filestem}_ch??.ims`; do
        channels_in+=( ${fname} )
    done

}


function gather_maxlabels {

    local maxlabelfile="${1}"

    > "${maxlabelfile}"
    for im in ${images_in[@]}; do
        maxlabel=`h5ls -v ${im} | grep -A 2 "maxlabel" | tail -n 1 | tr -d 'Data:' | tr -d ' '`
        echo "${maxlabel}" >> "${maxlabelfile}"
    done

}


function set_zipquads {

    local start_x="${1}"
    local start_y="${2}"
    local step_x="${3}"
    local step_y="${4}"
    local stop_x="${5}"
    local stop_y="${6}"

    unset zipquads
    zipquads=()
    for x in `seq ${start_x} ${step_x} $((stop_x - 1))`; do
        for y in `seq ${start_y} ${step_y} $((stop_y - 1))`; do
            zipquads+=($((x*stop_y + y)))
        done
    done

}


function set_seamnumbers {

    local axis="${1}"
    local TASK_ID="${2}"
    local start_x="${3}"
    local start_y="${4}"
    local nseams_x="${5}"
    local nseams_y="${6}"

    if [ "${axis}" == "0" ]
    then
        local idx=$((TASK_ID - 1))
        set_zipquads "${start_x}" "${start_y}" 2 2 "${nseams_x}" "${nseams_y}"
        local seamnumber="${zipquads[idx]}"
	    local seam_x=$((seamnumber / nseams_y))
    	local seam_y=$((seamnumber % nseams_y))
        seamnumbers="-1 ${seam_y} ${seam_x}"
    elif [ "${axis}" == "1" ]
    then
        seamnumbers="-1 $((TASK_ID - 1)) -1"
    elif [ "${axis}" == "2" ]
    then
        seamnumbers="-1 -1 $((TASK_ID - 1))"
    else
        seamnumbers="-1 -1 -1"
    fi


}


function set_idss {

    local pattern=$1
    local sep=$2

    unset idss
    idss=( `( set -o posix ; set ) | grep "^${pattern}" | cut -f1 -d"${sep}"` )

}


function mergeblocks_outputpath {

    local format=$1
    local ids=$2

    if [ "${format}" == "ims" ]
    then
        # ng_CROP1nucl/dapi_mask_nuclei_.ims
        out_path="${datadir}/${dataset}_${ids////-}.ims"
        cp ${datadir}/${dataset}${dataset__ims_ref_postfix}.ims ${out_path}
    elif [ "${format}" == "h5" ]
    then
        out_path="${datadir}/${dataset}_${ids////-}.h5/${ids}"
    else
        out_path="${datadir}/${dataset}_${ids////-}.h5/${ids}"
    fi

}


function get_blockstem_index {
    # Locate a block identifier in an array and return it's index.

    local blockstem=$1

    for i in "${!blockstems[@]}"; do
       if [[ "${blockstems[$i]}" = "${blockstem}" ]]; then
           echo "${i}";
       fi
    done

}


###==========================================================================###
### functions for job submission
###==========================================================================###

function submit {

    local scriptfile="${1}"
    local dep_jid="${2}"

    case "${compute_env}" in
        'SGE')
            [[ -z $dep_jid ]] && dep='' || dep="-hold_jid $dep_jid"
            [[ -z $array ]] && arrayspec='' || arrayspec="-t $range"
            if [ "$dep_jid" == 'h' ]
            then
                echo "not submitting $scriptfile"
                echo "qsub -cwd $arrayspec $dep $scriptfile"
            else
                jid=$( qsub -cwd $arrayspec $dep $scriptfile )
                jid=`echo $jid | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`
                echo "submitted $scriptfile as ${jid} with ${dep}"
            fi
            ;;
        'SLURM')
            [[ -z $dep_jid ]] && dep='' || dep="--dependency=afterok:$dep_jid"
            if [ "$dep_jid" == 'h' ]
            then
                echo "not submitting $scriptfile"
                echo "sbatch --parsable $dep $scriptfile"
            else
                jid=$( sbatch --parsable $dep $scriptfile )
                echo "submitted $scriptfile as ${jid} with ${dep}"
            fi
            ;;
    esac

}


function bash_directives {
    # Generate bash directives for submission script.

    echo '#!/usr/bin/env bash'
    # echo 'set -ux'
    echo ''

}


function sbatch_directives {
    # Generate sbatch directives for submission script.

    expand_submit_pars "$@"

    echo "#SBATCH --job-name=$jobname"
    echo "#SBATCH --partition=$partition"
    [[ "$partition" == 'gpu' ]] &&
        echo "#SBATCH --gpus-per-node=$gpus_per_node"
    echo "#SBATCH --mem=$mem"
    echo "#SBATCH --time=$wtime"
    echo "#SBATCH --nodes=$nodes"
    echo "#SBATCH --ntasks-per-node=$tasks"
    [[ -z $array ]] &&
        { output="${subfile}.o%A"; error="${subfile}.e%A"; } ||
        { echo "#SBATCH --array=$range" &&
        { output="${subfile}.o%A.%a"; error="${subfile}.e%A.%a"; } ; }
    echo "#SBATCH --output=$output"
    echo "#SBATCH --error=$error"
    echo ''

}


function pbs_directives {
    # Generate pbs directives for submission script.

    expand_submit_pars "$@"

    echo "#PBS -N EM_$jobname"
    echo "#PBS -l mem=$mem"
    echo "#PBS -l walltime=$wtime"
    echo "#PBS -l nodes=$nodes:ppn=$tasks"
    [[ -z $array ]] || echo "#PBS -t $range"
    echo "#PBS -V"
    echo ''

}


function conda_cmds {
    # Generate conda directives for submission script.

    eval conda_env="\$${stage}__submit__conda_env"
    [[ -z "${conda_env}" ]] && conda_env=${submit_defaults__submit__conda_env}

    if [ ! -z $conda_env ]
    then
        echo source "${CONDA_SH}"
        echo conda activate "${conda_env}"
        echo ''
    fi

}


function base_cmds {
    # Generate basic dataset directives.

    case "${compute_env}" in
        'SGE')
            echo TASK_ID="\${SGE_TASK_ID}"
            ;;
        'SLURM')
            echo TASK_ID="\${SLURM_ARRAY_TASK_ID}"
            ;;
    esac
    echo ''
    echo idx="\$((TASK_ID - 1))"
    echo ''

    echo ''
    echo dataset="${dataset}"
    echo ''
    echo projectdir="${projectdir}"
    echo datadir="${datadir}"
    echo stackdir="${stackdir}"
    echo channeldir="${channeldir}"
    echo blockdir="${blockdir}"
    echo profdir="${profdir}"
    echo featdir="${featdir}"
    echo jobdir="${jobdir}"
    echo ''
    echo source "${STAPL3D}/pipelines/functions.sh"
    echo load_dataset "${projectdir}" "${dataset}"
    echo load_parameters "${dataset}"
    echo ''

    if [ "$array" == 'channel_plane' ]
    then
        echo ch_idx="\$((idx/Z))"
        echo pl_idx="\$((idx%Z))"
        echo channelstem="\${channeldir}/\${channelstems[ch_idx]}"
    else
        echo channelstem="\${channeldir}/\${channelstems[idx]}"
    fi

    echo filestem="\${datadir}/\${dataset}"
    echo shading_stem="\${filestem}${shading__params__postfix}"
    echo stitching_stem="\${shading_stem}${stitching__params__postfix}"
    echo biasfield_stem="\${stitching_stem}${biasfield__params__postfix}"
    echo block_id="\${block_ids[idx]}"
    echo blockstem="\${blockdir}/\${dataset_preproc}_\${block_id}"

}


function no_parallelization {
    # Generate directives for processing without parallelization.

    echo ''

}


function stack_parallelization {
    # Generate directives for parallel stacks.

    echo ''

}


function channel_plane_parallelization {
    # Generate directives for parallel channels.

    echo ''

}


function channel_parallelization {
    # Generate directives for parallel channels.

    echo ''

}


function block_parallelization {
    # Generate directives for parallel blocks.

    echo ''

}


function zipline_parallelization {

    echo ''

}


function zipquad_parallelization {

    echo ''

}


function idss_parallelization {
    # Generate directives for parallel channels.

    echo "set_idss ${stage}__ids..__ids= ="
    echo ''

}


function stardistblock_parallelization {
    # Generate directives for parallel blocks.
    echo ''
    echo blockd_id=\`printf %05d \$idx\`
    echo blockdir_stardist="${blockdir}_stardist"
    echo dataset_preproc="\${dataset}"
    echo blockstem_stardist="\${blockdir_stardist}/\${dataset_preproc}_\${block_id}"
    echo ''
}


function _parallelization {

    echo ''

}


function finishing_directives {
    # Generate directives for parallel channels.

    eval conda_env="\$${stage}__submit__conda_env"
    [[ -z "${conda_env}" ]] && conda_env=${submit_defaults__submit__conda_env}
    if [ ! -z $conda_env ]
    then
        echo ''
        echo conda deactivate
        echo ''
    fi

    # echo "sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode -j \${SLURM_JOB_ID} > ${subfile}.a\${SLURM_JOB_ID}"  # TODO $PBS_JOBID
    # echo ''

}



###==========================================================================###
### functions for job generation [generalized]
###==========================================================================###

function set_submit_pars {

    local stage="$1"

    unset submit_pars
    submit_pars=()
    add_submit_par $stage 'array' 'no'
    add_submit_par $stage 'nodes' '1'
    add_submit_par $stage 'tasks' '1'
    add_submit_par $stage 'mem' '10G'
    add_submit_par $stage 'wtime' '01:00:00'

    case "${submit_pars[0]}" in
        'no')
            array_stop="1"
            ;;
        'stack')
            array_stop="$M"
            ;;
        'plane')
            array_stop="$Z"
            ;;
        'channel')
            array_stop="$C"
            ;;
        'channel_plane')
            array_stop="$((C*Z))"
            ;;
        'block')
            array_stop="${#blockstems[@]}"
            ;;
        'stardistblock')
            array_stop="${nblocks}"
            ;;
        'zipline')
            array_start="$((start + 1))"
            array_stop="${stop}"
            array_step=2
            ;;
        'zipquad')
            set_zipquads ${start_x} ${start_y} 2 2 $((nx-1)) $((ny-1))
            array_stop="${#zipquads[@]}"
            ;;
        'idss')
            set_idss "${stage}__params__ids..__ids=" '='
            array_stop="${#idss[@]}"
            ;;
    esac

    unset array_range
    build_array_range $stage 'start' '1' ''
    build_array_range $stage 'stop' '1' '-'
    build_array_range $stage 'step' '1' ':'
    build_array_range $stage 'simul' '0' '%'

    submit_pars+=( $array_range )

    add_submit_par $stage 'partition' 'cpu'
    add_submit_par $stage 'gpus_per_node' '1'

}


function add_submit_par {

    local stage="${1}"
    local varname="${2}"
    local default="${3}"

    eval var=\$${stage}__submit__${varname}
    [[ -z "${var}" ]] && eval var=\$submit_defaults__submit__${varname}
    [[ -z "${var}" ]] && var="${default}"

    submit_pars+=( ${var} )

}


function build_array_range {

    local stage="${1}"
    local varname="${2}"
    local default="${3}"
    local divider="${4}"

    eval var=\$array_${varname}
    [[ -z "${var}" ]] && eval var=\$${stage}__submit__${varname}
    [[ -z "${var}" ]] && eval var=\$submit_defaults__submit__${varname}
    [[ -z "${var}" ]] && var="${default}"

    [[ "${var}" != "0" ]] && array_range+="${divider}${var}"

}


function expand_submit_pars {

    [[ -z "$1" ]] && export array='' || export array=$1
    export nodes=$2
    export tasks=$3
    export mem=$4
    export wtime=$5
    export range=$6
    export partition=$7
    export gpus_per_node=$8

}


function generate_script {

    local stage=$1
    shift 1

    jobname="go_${dataset__alias}_${stage}"
    subfile="${datadir}/${jobname}.sh"

    local submit_pars=( "$@" )
    [[ "${stage}" == ziplines* ]] && stage='ziplines'
    [[ "${stage}" == zipquads* ]] && stage='zipquads'
    [[ ${#submit_pars[@]} -eq 0 ]] && set_submit_pars ${stage}

    bash_directives > "${subfile}"

    case "${compute_env}" in
        'SGE')
            pbs_directives "${submit_pars[@]}" >> "${subfile}"
            ;;
        'SLURM')
            sbatch_directives "${submit_pars[@]}" >> "${subfile}"
            ;;
    esac

    base_cmds >> "${subfile}"

    eval "${submit_pars[0]}"_parallelization >> "${subfile}"

    conda_cmds >> "${subfile}"

    eval get_cmd_${stage} >> "${subfile}"

    finishing_directives >> "${subfile}"

    echo "${subfile}"

}


###==========================================================================###
### functions to generate bash commands
###==========================================================================###

function get_py_shading {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'ch_idx = int(sys.argv[3])'
    echo 'pl_idx = int(sys.argv[4])'
    echo ''
    echo "from stapl3d.preprocessing import shading"
    echo "shading.estimate(
        image_in,
        parameter_file,
        channels=[ch_idx],
        planes=[pl_idx],
        )"

}
function get_cmd_shading {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${filestem}.${shading__file_format}" \
        "${parfile}" \
        "\${ch_idx}" "\${pl_idx}"

}


function get_py_shading_postproc {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.preprocessing import shading"
    echo "shading.postprocess(
        image_in,
        parameter_file,
        channels=[idx],
        )"

}
function get_cmd_shading_postproc {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${filestem}.${shading__file_format}" \
        "${parfile}" \
        "\${idx}"

}


function get_py_shading_apply {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.preprocessing import shading"
    echo "shading.czi_split_zstacks(
        image_in,
        offset=idx,
        nstacks=1,
        correct=True,
        )"

}
function get_cmd_shading_apply {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${filestem}.${shading__file_format}" \
        "${parfile}" \
        "\${idx}"

}


function get_py_stitching_prep {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo ''
    echo "from stapl3d.preprocessing import stitching"
    echo "stitching.write_stack_offsets(image_in)"

}
function get_cmd_stitching_prep {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${filestem}.${shading__file_format}"

}
function get_cmd_stitching_load {
    echo $FIJI --headless --console -macro \
        "$STAPL3D/preprocessing/stitching.ijm" \
        \"1 ${stackdir}/${dataset}${shading__params__postfix} ${stitchingdir} \
        ${dataset} \${idx} ${stitching__params__postfix} \
        ${elsize_z} ${elsize_y} ${elsize_x}\"
    echo $FIJI --headless --console -macro \
        "$STAPL3D/preprocessing/stitching.ijm" \
        \"2 ${stackdir}/${dataset}${shading__params__postfix} ${stitchingdir} \
        ${dataset} \${idx} ${stitching__params__postfix} \
        ${elsize_z} ${elsize_y} ${elsize_x}\"
}
function get_cmd_stitching_calc {
    echo $FIJI --headless --console -macro \
        "$STAPL3D/preprocessing/stitching.ijm" \
        \"3 ${stackdir}/${dataset}${shading__params__postfix} ${stitchingdir} \
        ${dataset} ${stitching__params__channel} ${stitching__params__postfix} \
        ${elsize_z} ${elsize_y} ${elsize_x}\"
    echo $FIJI --headless --console -macro \
        "$STAPL3D/preprocessing/stitching.ijm" \
        \"4 ${stackdir}/${dataset}${shading__params__postfix} ${stitchingdir} \
        ${dataset} ${stitching__params__channel} ${stitching__params__postfix} \
        ${elsize_z} ${elsize_y} ${elsize_x}\"
    echo $FIJI --headless --console -macro \
        "$STAPL3D/preprocessing/stitching.ijm" \
        \"5 ${stackdir}/${dataset}${shading__params__postfix} ${stitchingdir} \
        ${dataset} ${stitching__params__channel} ${stitching__params__postfix} \
        ${elsize_z} ${elsize_y} ${elsize_x}\"
}
function get_py_stitching_fuse {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'filestem = sys.argv[1]'
    echo 'channel = int(sys.argv[2])'
    echo 'channel_ref = int(sys.argv[3])'
    echo 'dapi_shift = float(sys.argv[4])'
    echo ''
    echo "from stapl3d.preprocessing import stitching"
    echo "stitching.adapt_xml(filestem, channel, channel_ref, dapi_shift)"
}
function get_cmd_stitching_fuse {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "${stitchingdir}/${dataset}" \  # "${stitchingdir}/${dataset}${shading__params__postfix}" \
        "\${idx}" \
        "${stitching__params__channel}" \
        "${dataset__dapi_shift}"

    echo $FIJI --headless --console -macro \
        "$STAPL3D/preprocessing/stitching.ijm" \
        \"6 ${stackdir}/${dataset}${shading__params__postfix} ${stitchingdir} \
        ${dataset} \${idx} ${stitching__params__postfix} \
        ${elsize_z} ${elsize_y} ${elsize_x}\"
}


function get_py_mask {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo ''
    echo "from stapl3d.preprocessing import masking"
    echo "masking.estimate(
        image_in,
        parameter_file,
        )"

}
function get_cmd_mask {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "${parfile}"

}


function get_py_splitchannels {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d import imarisfiles"
    echo "imarisfiles.split_channels(
        image_in,
        parameter_file,
        channels=[idx],
        )"

}
function get_cmd_splitchannels {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_ims_aggregate1 {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'out_path = sys.argv[1]'
    echo 'ref_path = sys.argv[2]'
    echo 'inputstem = sys.argv[3]'
    echo 'channel_pat = sys.argv[4]'
    echo 'postfix = sys.argv[5]'
    echo ''
    echo "from stapl3d.imarisfiles import make_aggregate"
    echo "make_aggregate(out_path, ref_path, inputstem, channel_pat, postfix)"

}
function get_cmd_ims_aggregate1 {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "\${stitching_stem}${dataset__ims_ref_postfix}.ims" \
        "${channeldir}/${dataset_stitching}" '_ch??' ''

}


function get_py_biasfield {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.preprocessing import biasfield"
    echo "biasfield.estimate(
        image_in,
        parameter_file,
        channels=[idx],
        )"

}
function get_cmd_biasfield {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_biasfield_stack {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'inputstem = sys.argv[1]'
    echo 'outputstem = sys.argv[2]'
    echo 'biasfield_postfix = sys.argv[3]'
    echo ''
    echo "from glob import glob"
    echo "from stapl3d.preprocessing import biasfield"
    echo "from stapl3d import reporting"
    echo ''
    echo "inputpat = '{}_ch??{}'.format(inputstem, biasfield_postfix)"
    echo ''
    echo "inputfiles = glob('{}.h5'.format(inputpat))"
    echo "inputfiles.sort()"
    echo "outputfile = '{}.h5'.format(outputstem)"
    echo "biasfield.stack_bias(inputfiles, outputfile)"
    echo ''
    echo "pdfs = glob('{}.pdf'.format(inputpat))"
    echo "pdfs.sort()"
    echo "pdf_out = '{}.pdf'.format(outputstem)"
    echo "reporting.merge_reports(pdfs, pdf_out)"
    echo ''
    echo "pickles = glob('{}.pickle'.format(inputpat))"
    echo "pickles.sort()"
    echo "zip_out = '{}.zip'.format(outputstem)"
    echo "reporting.zip_parameters(pickles, zip_out)"

}
function get_cmd_biasfield_stack {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "${biasfielddir}/${dataset_stitching}" \
        "${datadir}/${dataset_biasfield}" \
        "${biasfield__params__postfix}"

}


function get_py_biasfield_apply {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.preprocessing import biasfield"
    echo "biasfield.apply(
        image_in,
        parameter_file,
        channels=[idx],
        )"

}
function get_cmd_biasfield_apply {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_ims_aggregate2 {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'out_path = sys.argv[1]'
    echo 'ref_path = sys.argv[2]'
    echo 'inputstem = sys.argv[3]'
    echo 'channel_pat = sys.argv[4]'
    echo 'postfix = sys.argv[5]'
    echo ''
    echo "from stapl3d.imarisfiles import make_aggregate"
    echo "make_aggregate(out_path, ref_path, inputstem, channel_pat, postfix)"

}
function get_cmd_ims_aggregate2 {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "\${stitching_stem}${dataset__ims_ref_postfix}.ims" \
        "${channeldir}/${dataset_stitching}" '_ch??' "${biasfield__params__postfix}"

}


function get_cmd_block_segmentation {

    jobname="go_${dataset__alias}_splitblocks"
    get_cmd_splitblocks
    jobname="go_${dataset__alias}_membrane_enhancement"
    get_cmd_membrane_enhancement
    jobname="go_${dataset__alias}_segmentation"
    get_cmd_segmentation

}


function get_py_splitblocks {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d import blocks"
    echo "blocks.split(
        image_in,
        parameter_file,
        blocks=[idx],
        )"
}
function get_cmd_splitblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_membrane_enhancement {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.segmentation import enhance"
    echo "enhance.estimate(
        image_in,
        parameter_file,
        blocks=[idx],
        )"
}
function get_cmd_membrane_enhancement {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}"

    echo ''
    echo "rm \${blockdir}/${dataset_preproc}_\${block_id}_memb-eigen.mha"
    echo "rm \${blockdir}/${dataset_preproc}_\${block_id}_memb-*.nii.gz"

}


function get_py_segmentation {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.segmentation import segment"
    echo "segment.estimate(
        image_in,
        parameter_file,
        blocks=[idx],
        )"
}
function get_cmd_segmentation {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


# TODO: move glob inside function
function get_py_segmentation_postproc {

    echo '#!/usr/bin/env python'
    echo ''
    echo "from glob import glob"
    echo "from stapl3d.reporting import merge_reports"
    echo "pdfs = glob('${blockdir}/${dataset_preproc}_?????-?????_?????-?????_?????-?????_seg.pdf')"
    echo "pdfs.sort()"
    echo "merge_reports(
        pdfs,
        '${datadir}/${dataset_preproc}_seg.pdf',
        )"

}
function get_cmd_segmentation_postproc {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}"

    echo "rm ${blockdir}/${dataset_preproc}_?????-?????_?????-?????_?????-?????_seg.pdf"

}


function get_py_relabel {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'idx = int(sys.argv[2])'
    echo 'maxlabelfile = sys.argv[3]'
    echo 'postfix = sys.argv[4]'
    echo ''
    echo 'from stapl3d.segmentation.zipping import relabel_parallel'
    echo "relabel_parallel(
        image_in,
        idx,
        maxlabelfile,
        pf=postfix,
        )"

}
function get_cmd_relabel {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${blockstem}.h5/segm/${segmentation__params__segments_ods}" \
        "\${idx}" \
        "${blockdir}/${dataset_preproc}_maxlabels_${segmentation__params__segments_ods}.txt" \
        "${relabel__params__postfix}"

}


function get_py_copyblocks {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'idx = int(sys.argv[2])'
    echo ''
    echo 'from stapl3d.segmentation.zipping import copy_blocks_parallel'
    echo "copy_blocks_parallel(
        image_in,
        idx,
        )"

}
function get_cmd_copyblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
    "\${blockstem}.h5/segm/${segmentation__params__segments_ods}${relabel__params__postfix}" \
    "\${idx}"

}


function get_cmd_ziplines { get_cmd_zipping "${axis}" ; }
function get_cmd_zipquads { get_cmd_zipping "${zipquads__params__axis}" ; }
function get_py_zipping {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'blocksize = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]'
    echo 'blockmargin = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])]'
    echo 'axis = int(sys.argv[7])'
    echo 'seamnumbers = [int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])]'
    echo 'maxlabelfile = sys.argv[11]'
    echo 'ids_nucl = sys.argv[12]'
    echo 'ids_memb_chan = sys.argv[13]'
    echo 'outputstem = sys.argv[14]'
    echo 'images_in = sys.argv[15:]'
    echo ''
    echo 'from stapl3d.segmentation.zipping import resegment_block_boundaries'
    echo "resegment_block_boundaries(
        images_in,
        blocksize,
        blockmargin,
        axis,
        seamnumbers,
        mask_dataset='',
        relabel=False,
        maxlabel=maxlabelfile,
        in_place=True,
        ids_nucl=ids_nucl,
        ids_memb_chan=ids_memb_chan,
        outputstem=outputstem,
        save_steps=False,
        )"

}
function get_cmd_zipping {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_zipping > "$pyfile"

    local axis="${1}"
    local ids="segm/${segmentation__params__segments_ods}${zipping__params__postfix}"

    echo ''
    echo set_images_in "${blockdir}/${dataset_preproc}" "${ids}"
    echo set_seamnumbers "${axis}" "\${TASK_ID}" "${start_x}" "${start_y}" $((nx - 1)) $((ny - 1))

    echo python "${pyfile}" \
        "${Z}" "${bs}" "${bs}" \
        "0" "${bm}" "${bm}" \
        "${axis}" \
        "\${seamnumbers}" \
        "${blockdir}/${dataset_preproc}_maxlabels_${segmentation__params__segments_ods}${zipping__params__postfix}.txt" \
        "${zipping__params__ids_nucl}" \
        "${zipping__params__ids_memb_chan}" \
        "${blockdir}/${dataset_preproc}" \
        "\${images_in[@]}"

}


# TODO: move glob inside function
function get_py_zipping_postproc {

    echo '#!/usr/bin/env python'
    echo ''
    echo "from glob import glob"
    echo "from stapl3d.reporting import merge_reports"
    echo "pdfs = glob('${blockdir}/${dataset_preproc}_reseg_axis?-seam??-j???.pdf')"
    echo "pdfs.sort()"
    echo "merge_reports(
        pdfs,
        '${datadir}/${dataset_preproc}_reseg.pdf',
        )"

}
function get_cmd_zipping_postproc {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}"

    echo "rm ${blockdir}/${dataset_preproc}_reseg_axis?-seam??-j???.pdf"

}


function get_cmd_segmentation_gather { get_cmd_gather "${stage}" ; }
function get_cmd_relabel_gather { get_cmd_gather "${stage}" ; }
function get_cmd_copyblocks_gather { get_cmd_gather "${stage}" ; }
function get_cmd_zipping_gather { get_cmd_gather "${stage}" ; }
function get_cmd_gather {

    local stage="${1}"
    eval postfix="\${${stage}__params__postfix}"

    echo set_images_in \
        "${blockdir}/${dataset_preproc}" \
        "segm/${segmentation__params__segments_ods}${postfix}"
    echo maxlabelfile="${blockdir}/${dataset_preproc}_maxlabels_${segmentation__params__segments_ods}${postfix}.txt"
    echo gather_maxlabels "\${maxlabelfile}"

}


function get_py_subsegment {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo 'from stapl3d.segmentation import segment'
    echo "segment.subsegment(
        image_in,
        parameter_file,
        blocks=[idx],
        )"

}
function get_cmd_subsegment {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_mergeblocks {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo 'from stapl3d import blocks'
    echo "blocks.merge(
        image_in,
        parameter_file,
        step_id='${stage}',
        idss_select=[idx],
        )"

}
function get_cmd_mergeblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_features {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo 'from stapl3d.segmentation import features'
    echo "features.estimate(
        image_in,
        parameter_file,
        blocks=[idx],
        )"

}
function get_cmd_features {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}"

}


function get_py_features_postproc {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo ''
    echo 'from stapl3d.segmentation import features'
    echo "features.postproc(
        image_in,
        parameter_file,
        )"

}
function get_cmd_features_postproc {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}"

}


function get_py_stardist_train {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'basedir = sys.argv[1]'
    echo 'run_name = sys.argv[2]'
    echo ''
    echo 'from stapl3d.segmentation import stardist_nuclei'
    echo "stardist_nuclei.stardist_train(basedir, run_name)"

}
function get_cmd_stardist_train {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "${stardist_train__params__stardir}" \
        "${stardist_train__params__model_name}"

}

function get_py_stardist_predict {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'modeldir = sys.argv[1]'
    echo 'modelname = sys.argv[2]'
    echo 'image_in = sys.argv[3]'
    echo 'idx = int(sys.argv[4])'
    echo 'nomi = float(sys.argv[5])'
    echo 'noma = float(sys.argv[6])'
    echo 'print_nblocks = bool(sys.argv[7])'
    echo ''
    echo 'from stapl3d.segmentation import stardist_nuclei'
    echo "stardist_nuclei.stardist_predict(
        modeldir,
        modelname,
        image_in,
        idx,
        normalization=[nomi, noma],
        print_nblocks=False,
        )"

}
function get_cmd_stardist_predict {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "${stardist_predict__params__stardir}" \
        "${stardist_predict__params__model_name}" \
        "\${filestem}${stardist_predict__params__dapi_postfix}.ims" \
        "\${idx}" \
        "${stardist_predict__params__nomi}" "${stardist_predict__params__noma}" \
        "False"

    # echo python "${pyfile}" \
    #     "${stardist_predict__params__stardir}" \
    #     "${stardist_predict__params__model_name}" \
    #     "\${biasfield_stem}${stardist_predict__params__dapi_postfix}.ims" \
    #     "\${idx}" \
    #     "${stardist_predict__params__nomi}" "${stardist_predict__params__noma}"

}
function get_py_stardist_nblocks {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'modeldir = sys.argv[1]'
    echo 'modelname = sys.argv[2]'
    echo 'image_in = sys.argv[3]'
    echo 'idx = int(sys.argv[4])'
    echo 'nomi = float(sys.argv[5])'
    echo 'noma = float(sys.argv[6])'
    echo 'print_nblocks = bool(sys.argv[7])'
    echo ''
    echo 'from stapl3d.segmentation import stardist_nuclei'
    echo "stardist_nuclei.stardist_predict(
        modeldir,
        modelname,
        image_in,
        idx,
        normalization=[nomi, noma],
        print_nblocks=True,
        )"

}
function get_cmd_stardist_nblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "${stardist_predict__params__stardir}" \
        "${stardist_predict__params__model_name}" \
        "\${filestem}${stardist_predict__params__dapi_postfix}.ims" \
        "\${idx}" \
        "${stardist_predict__params__nomi}" "${stardist_predict__params__noma}" \
        "True"

}


function get_cmd_stardist_gather {

    local stage="${1}"

    stardistblock_parallelization

    eval postfix="\${${stage}__params__postfix}"
    echo set_images_in_stardist \
        "\${blockdir_stardist}/\${dataset_preproc}${stardist_predict__params__dapi_postfix}" \
        "${stardist__params__segments_ods}${postfix}"
    echo maxlabelfile="\${blockdir_stardist}/\${dataset_preproc}${stardist_predict__params__dapi_postfix}_maxlabels${postfix}.txt"
    echo gather_maxlabels "\${maxlabelfile}"

}


function get_py_stardist_mergeblocks {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'blockdir = sys.argv[1]'
    echo 'image_in_ref = sys.argv[2]'
    echo ''
    echo 'from stapl3d.segmentation import stardist_nuclei'
    echo "stardist_nuclei.stardist_mergeblocks(
        blockdir,
        image_in_ref,
        )"

}
function get_cmd_stardist_mergeblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    stardistblock_parallelization

    echo python "${pyfile}" \
        "\${blockdir_stardist}" \
        "\${filestem}${stardist_predict__params__dapi_postfix}.ims"

}


function get_cmd_unet3d_memb_train {

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_memb_train__params__unetdir?;\
         s?MODELNAME?$unet3d_memb_train__params__modelname?g" \
         "${STAPL3D}/pipelines/unet3d_memb_train.yml" \
         > "${datadir}/${dataset}_unet3d_memb_train.yml"

    echo train3dunet --config "${datadir}/${dataset}_unet3d_memb_train.yml"

}


function get_cmd_unet3d_memb_predict {
    # FIXME: kidney model gives error on load via predict3dunet => needs DataParallel wrapping

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_memb_predict__params__unetdir?;\
         s?MODELNAME?$unet3d_memb_predict__params__modelname?;\
         s?IDS_MEMBRANE?$unet3d_memb_predict__params__ids_membrane?;\
         s?PATH_TO_THE_TEST_SET?${blockdir}?g" \
         "${STAPL3D}/pipelines/unet3d_memb_predict.yml" \
         > "${datadir}/${dataset}_unet3d_memb_predict.yml"

    echo predict3dunet --config "${datadir}/${dataset}_unet3d_memb_predict.yml"

}


function get_cmd_unet3d_nucl_train {

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_nucl_train__params__unetdir?;\
         s?MODELNAME?$unet3d_nucl_train__params__modelname?g" \
         "${STAPL3D}/pipelines/unet3d_nucl_train.yml" \
         > "${datadir}/${dataset}_unet3d_nucl_train.yml"

    echo train3dunet --config "${datadir}/${dataset}_unet3d_nucl_train.yml"

}


function get_cmd_unet3d_nucl_predict {
    # FIXME: kidney model gives error on load via predict3dunet => needs DataParallel wrapping

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_nucl_predict__params__unetdir?;\
         s?MODELNAME?$unet3d_nucl_predict__params__modelname?;\
         s?IDS_NUCLEUS?$unet3d_nucl_predict__params__ids_nucleus?;\
         s?PATH_TO_THE_TEST_SET?${blockdir}?g" \
         "${STAPL3D}/pipelines/unet3d_nucl_predict.yml" \
         > "${datadir}/${dataset}_unet3d_nucl_predict.yml"

    echo predict3dunet --config "${datadir}/${dataset}_unet3d_nucl_predict.yml"

}


function get_py_plantseg_predict {
    ### stapl3d-blocks to plantseg-input

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'filepath_in = sys.argv[1]'
    echo 'filepath_out = sys.argv[2]'
    echo 'dset_in = sys.argv[3]'
    echo 'dset_out = sys.argv[4]'
    echo 'delete = sys.argv[5]'
    echo 'links = sys.argv[6]'
    echo 'is_unet = sys.argv[7]'
    echo ''
    echo 'from stapl3d import blocks'
    echo "blocks.link_blocks(filepath_in, filepath_out, dset_in, dset_out, delete, links, is_unet)"

}
function prep_plantseg_predict {
    # link plantseg dirtree to blockstems

    local blocks_ps="$1"

    mkdir -p "${datadir}/${blocks_ps}"
    for filepath in `ls ${blockdir}/*.h5`; do
        filename=`basename "${filepath}"`
        mkdir -p "${datadir}/${blocks_ps}/${filename%.h5}"
        ln -s "${blockdir}/${filename}" "${datadir}/${blocks_ps}/${filename%.h5}/${filename}"
    done

}
function get_cmd_plantseg_predict {

    echo blockstem=\${blockdir}/\${dataset}_\${block_id}  # for HFK16w
    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"
    echo python "${pyfile}" "\${blockstem}.h5" "\${blockstem}.h5" 'raw' 'memb/mean' 'True' 'True' 'False'
    echo prep_plantseg_predict "${plantseg_predict__params__ps_blockdir}"
    echo ""
    echo ps_path="\${datadir}/${plantseg_predict__params__ps_blockdir}/\${dataset}_\${block_id}"
    echo sed "\"s?BLOCKS_PLANTSEG?\${ps_path}?;s?MODELNAME?$plantseg_predict__params__modelname?;s?DIM_Z?106?g\"" "\${STAPL3D}/pipelines/plantseg_config.yml" \> "\${ps_path}/\${dataset}_\${block_id}_plantseg.yml"
    echo plantseg --config "\${ps_path}/\${dataset}_\${block_id}_plantseg.yml"
    echo ""
    echo python "${pyfile}" "\${ps_path}/${plantseg_predict__params__modelname}/\${dataset}_\${block_id}_predictions.h5" "\${blockstem}.h5" 'predictions' 'memb/3dunet' 'True' 'False' 'True'

}




}


