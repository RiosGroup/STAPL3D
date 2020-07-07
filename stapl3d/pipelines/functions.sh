#!/bin/bash


###==========================================================================###
### functions to prepare processing
###==========================================================================###

function load_dataset {

    local projectdir="${1}"
    local dataset="${2}"

    echo "###==========================================================================###"
    echo "### processing dataset ${dataset}"

    set_datadir "${projectdir}" "${dataset}"
    echo "### data directory is ${datadir}"
    # TODO: exit on undefined PROJECT directory?

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
    set_ZYXCT_ims -v "${datadir}/${dataset}.ims"
    write_ZYXCT_to_yml ${dataset} "${datadir}/${dataset}_dims.yml"

}


function load_parameters {

    local dataset="${1}"
    local parfile

    parfile="${datadir}/${dataset}.yml"
    eval $( parse_yaml "${parfile}" "" )
    [[ "$2" == '-v' ]] &&
        echo "### dataset details provided:" && parse_yaml "${parfile}"

    #parfile="${datadir}/${dataset}_params.yml"
    #eval $( parse_yaml "${parfile}" "" )
    #[[ "$2" == '-v' ]] && {
    #    echo "### using parameters:" && parse_yaml "${parfile}"
    #    echo "###==========================================================================###"
    #    echo "" ; }
    # TODO: exit on missing parameterfile [or revert to a default one in the package]
    # cp <package/pipelines/params.yml ${datadir}/${dataset}_params.yml

    set_dirtree "${datadir}"

    check_dims Z "$Z" || set_ZYXCT "${datadir}/${dataset}_dims.yml"
    check_dims Z "$Z" -v
    echo "### data dimensions are ZYXCT='${Z} ${Y} ${X} ${C} ${T}'"

    [[ "$2" == '-v' ]] && {
        echo "###==========================================================================###"
        echo "" ; }
    # TODO: exit on undefined ZYXCT

    bs="${dataset__bs}" && check_dims bs "$bs" || set_blocksize
    bm="${dataset__bm}" && check_dims bm "$bm" || bm=64

    dataset_preproc="${dataset}${shading__postfix}${stitching__postfix}${biasfield__postfix}"

    set_channelstems "${dataset_preproc}"

    set_blocks "${bs}" "${bm}"

    echo "### parallelization: ${#channelstems[@]} channels"
    echo "### parallelization: ${#blockstems[@]} blocks (${nx} x ${ny}) of blocksize ${bs} with margin ${bm}"
    echo "###==========================================================================###"
    echo ""

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


function set_ZYXCT {

    if ! [ -z "${dataset__Z}" ]
    then
        Z="${dataset__Z}"
        Y="${dataset__Y}"
        X="${dataset__X}"
        C="${dataset__C}"
        T="${dataset__T}"
    elif [ -f ${1} ]
    then
        eval $( parse_yaml "${1}" "" )
    elif ! [ -z "${datadir}/${dataset}.ims" ]
    then
        set_ZYXCT_ims '-v' "${datadir}/${dataset}.ims"
        write_ZYXCT_to_yml ${dataset} "${1}"
    elif ! [ -z "${datadir}/${dataset}_bfc.ims" ]
    then
        set_ZYXCT_ims '-v' "${datadir}/${dataset}.ims"
        write_ZYXCT_to_yml ${dataset} "${1}"
    fi

}


function write_ZYXCT_to_yml {

    local dataset="${1}"
    local parfile="${2}"

    echo "Z: ${Z}" > "${parfile}"
    echo "Y: ${Y}" >> "${parfile}"
    echo "X: ${X}" >> "${parfile}"
    echo "C: ${C}" >> "${parfile}"
    echo "T: ${T}" >> "${parfile}"

    echo " --- written ZYXCT='${Z} ${Y} ${X} ${C} ${T}' to ${parfile}"

}


function set_dirtree {

    local datadir="${1}"

    blockdir="${datadir}/${dirtree__datadir__blocks}"
    mkdir -p "${blockdir}"

    channeldir="${datadir}/${dirtree__datadir__channels}"
    mkdir -p "${channeldir}"

    shadingdir="${datadir}/${dirtree__datadir__shading}"
    mkdir -p "${shadingdir}"

    biasfielddir="${datadir}/${dirtree__datadir__biasfield}"
    mkdir -p "${biasfielddir}"

    profdir="${datadir}/${dirtree__datadir__profiling__base}"
    mkdir -p "${profdir}"

    featdir="${profdir}/${dirtree__datadir__profiling__featdir}"
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

                dstem="$( get_blockstem $dataset $bx $bX $by $bY $bz $bZ )"
                blockstems+=( "$dstem" )
                if [ "$verbose" == "-v" ]; then
                    echo "$dstem"
                fi

            done
        done
    done

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


function get_blockstem {
    # Get a block identifier from coordinates.

    local dataset=$1
    local x=$2
    local X=$3
    local y=$4
    local Y=$5
    local z=$6
    local Z=$7

    local xrange=`printf %05d $x`-`printf %05d $X`
    local yrange=`printf %05d $y`-`printf %05d $Y`
    local zrange=`printf %05d $z`-`printf %05d $Z`

    local dstem=${dataset}_${xrange}_${yrange}_${zrange}

    echo "$dstem"

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


###==========================================================================###
### functions for job submission
###==========================================================================###

function submit {

    local scriptfile="${1}"
    local dep_jid="${2}"

    [[ -z $dep_jid ]] && dep='' || dep="--dependency=afterok:$dep_jid"
    if [ "$dep_jid" == 'h' ]
    then
        echo "not submitting $scriptfile"
        echo "sbatch --parsable $dep $scriptfile"
    else
        jid=$( sbatch --parsable $dep $scriptfile )
        echo "submitted $scriptfile as ${jid} with ${dep}"
    fi

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
    echo "#SBATCH --mem=$mem"
    echo "#SBATCH --time=$wtime"
    echo "#SBATCH --nodes=$nodes"
    echo "#SBATCH --ntasks-per-node=$tasks"
    [[ -z $array ]] &&
        { output="${subfile}.o%A"; error="${subfile}.e%A"; } ||
        { echo "#SBATCH --array=$range" &&
        { output="${subfile}.o%A_%a"; error="${subfile}.e%A_%a"; } ; }
    echo "#SBATCH --output=$output"
    echo "#SBATCH --error=$error"
    echo ''

}


function conda_cmds {
    # Generate conda directives for submission script.

    eval conda_env="\$${stage}__conda__env"

    echo source "${CONDA_SH}"
    echo conda activate "${conda_env}"
    echo ''

}


function base_cmds {
    # Generate basic dataset directives.

    echo ''
    echo dataset="${dataset}"
    echo ''
    echo projectdir="${projectdir}"
    echo datadir="${datadir}"
    echo channeldir="${channeldir}"
    echo blockdir="${blockdir}"
    echo profdir="${profdir}"
    echo featdir="${featdir}"
    echo jobdir="${jobdir}"
    echo ''
    echo source "${STAPL3D}/pipelines/functions.sh"
    echo load_dataset "\${projectdir}" "\${dataset}"
    echo load_parameters "${dataset}"
    echo ''

    echo idx="\$((SLURM_ARRAY_TASK_ID-1))"
    echo filestem="${datadir}/${dataset}"
    echo shading_stem="\${filestem}${shading__postfix}"
    echo stitching_stem="\${shading_stem}${stitching__postfix}"
    echo biasfield_stem="\${stitching_stem}${biasfield__postfix}"
    echo channelstem="${channeldir}/\${channelstems[idx]}"
    echo blockstem="${blockdir}/\${blockstems[idx]}"

}


function no_parallelization {
    # Generate directives for processing without parallelization.

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


function _parallelization {

    echo ''

}


function finishing_directives {
    # Generate directives for parallel channels.

    echo ''
    echo 'conda deactivate'
    echo ''
    # echo "sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode -j \${SLURM_JOB_ID} > ${subfile}.a\${SLURM_JOB_ID}"  # TODO $PBS_JOBID
    # echo ''

}



###==========================================================================###
### functions for job generation [generalized]
###==========================================================================###

function set_submit_pars {

    local stage="$1"
    local range

    unset submit_pars
    submit_pars=()

    eval submit_pars+=( \$${stage}__submit__array )
    eval submit_pars+=( \$${stage}__submit__nodes )
    eval submit_pars+=( \$${stage}__submit__tasks )
    eval submit_pars+=( \$${stage}__submit__mem )
    eval submit_pars+=( \$${stage}__submit__wtime )

    case "${submit_pars[0]}" in
        'no')
            range="1-1:1"
            ;;
        'channel')
            range="1-$C:1"
            ;;
        'block')
            range="1-${#blockstems[@]}:1"
            ;;
        'zipline')
            range="$((start + 1))-${stop}:2"
            ;;
        'zipquad')
            set_zipquads ${start_x} ${start_y} 2 2 $((nx-1)) $((ny-1))
            range="1-${#zipquads[@]}:1"
            ;;
        'idss')
            set_idss "${stage}__ids..__ids=" '='
            range="1-${#idss[@]}:1"
            ;;
    esac

    submit_pars+=( $range )

}


function expand_submit_pars {

    [[ -z "$1" ]] && export array='' || export array=$1
    export nodes=$2
    export tasks=$3
    export mem=$4
    export wtime=$5
    export range=$6

}


function generate_script {

    local stage=$1
    shift 1

    jobname="go_${dataset__alias}_${stage}"
    subfile="${datadir}/${jobname}.sh"

    local submit_pars=( "$@" )
    # local parset="${stage}"
    # [[ "${stage}" == ziplines* ]] && parset='ziplines'
    # [[ "${stage}" == zipquads* ]] && parset='zipquads'
    [[ "${stage}" == ziplines* ]] && stage='ziplines'
    [[ "${stage}" == zipquads* ]] && stage='zipquads'
    [[ ${#submit_pars[@]} -eq 0 ]] && set_submit_pars ${stage}

    bash_directives > "${subfile}"
    sbatch_directives "${submit_pars[@]}" >> "${subfile}"
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

function get_py_shading_estimation {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo ''
    echo "from stapl3d.preprocessing import shading"
    echo "shading.estimate(
        image_in,
        parameter_file,
        channels=[idx],
        )"

}
function get_cmd_shading_estimation {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${filestem}.${shading__file_format}" \
        "\${filestem}.yml" \
        "\${idx}"

}


function get_py_generate_mask {

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
function get_cmd_generate_mask {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "\${filestem}.yml"

}


function get_py_bias_estimation {

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
function get_cmd_bias_estimation {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "\${filestem}.yml" \
        "\${idx}"

}


function get_py_bias_stack {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'outputstem = sys.argv[1]'
    echo 'inputfiles = sys.argv[2:]'
    echo ''
    echo "from stapl3d.preprocessing.biasfield import stack_bias"
    echo "stack_bias(
        inputfiles,
        outputstem,
        )"
    echo ''
    echo "from stapl3d.reporting import zip_parameters"
    echo "zip_parameters(
        '${biasfielddir}/${dataset}',
        'biasfield',
        )"
    echo ''
    echo "from stapl3d.reporting import merge_reports"
    echo "from glob import glob"
    echo "merge_reports(
        glob('${biasfielddir}/${dataset}_ch??${biasfield__params__postfix}').sort(),
        '${datadir}/${dataset}${biasfield__params__postfix}.pdf',
        )"

}
function get_cmd_bias_stack {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local channels_in=()
    local stitch_stem="${biasfielddir}/${dataset}${shading__params__postfix}${stitching__params__postfix}"
    for c in `seq 0 $(( C - 1 ))`; do
        channels_in+=("${stitch_stem}_ch`printf %02d $c`${biasfield__params__postfix}")
    done

    echo python "${pyfile}" \
        "\${biasfield_stem}" \
        "${channels_in[@]}"

    # echo "rm ${biasfielddir}/${dataset}_ch??${biasfield__params__postfix}.pdf"
    # echo "rm ${biasfielddir}/${dataset}_ch??${biasfield__params__postfix}.pickle"

}


function get_py_bias_apply {

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
function get_cmd_bias_apply {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" \
        "\${stitching_stem}.ims" \
        "\${filestem}.yml" \
        "\${idx}"

}


function get_py_ims_aggregate {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'out_path = sys.argv[1]'
    echo 'ref_path = sys.argv[2]'
    echo 'inputfiles = sys.argv[3:]'
    echo ''
    echo "from stapl3d.imarisfiles import make_aggregate"
    echo "make_aggregate(
        inputfiles,
        out_path,
        ref_path,
        )"

}
function get_cmd_ims_aggregate {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local chpat="${channeldir}/${dataset}_ch??${shading__params__postfix}${stitching__params__postfix}${biasfield__params__postfix}.ims"
    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "\${stitching_stem}${dataset__ims_ref_postfix}.ims" \
        `ls ${chpat}`

}


function get_cmd_block_segmentation {

    get_cmd_splitblocks
    get_cmd_membrane_enhancement
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
        "\${filestem}.yml" \
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
        "\${filestem}.yml" \
        "\${idx}"

    echo ''
    echo "rm \${blockstem}${biasfield__params__postfix}_memb-eigen.mha"
    echo "rm \${blockstem}${biasfield__params__postfix}_memb-*.nii.gz"

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
        "\${filestem}.yml" \
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
        "${blockdir}/${dataset_preproc}_maxlabels.txt" \
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
function get_cmd_zipquads { get_cmd_zipping "${zipquads__axis}" ; }
function get_py_zipping {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'blocksize = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]'
    echo 'blockmargin = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])]'
    echo 'axis = int(sys.argv[7])'
    echo 'seamnumbers = [int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])]'
    echo 'maxlabelfile = sys.argv[11]'
    echo 'outputstem = sys.argv[12]'
    echo 'images_in = sys.argv[13:]'
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
    echo set_seamnumbers "${axis}" "\${SLURM_ARRAY_TASK_ID}" "${start_x}" "${start_y}" $((nx - 1)) $((ny - 1))

    echo python "${pyfile}" \
        "${Z}" "${bs}" "${bs}" \
        "0" "${bm}" "${bm}" \
        "${axis}" \
        "\${seamnumbers}" \
        "${blockdir}/${dataset_preproc}_maxlabels${zipping__params__postfix}.txt" \
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
    echo maxlabelfile="${blockdir}/${dataset_preproc}_maxlabels${postfix}.txt"
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
        "\${filestem}.yml" \
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
        blocks=[idx],
        )"

}
function get_cmd_mergeblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "\${filestem}.yml" \
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
        "\${filestem}.yml" \
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
        "\${filestem}.yml"

}
