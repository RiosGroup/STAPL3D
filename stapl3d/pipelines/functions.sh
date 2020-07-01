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
    set_ZYXCT_ims '-v' "${datadir}/${dataset}.ims"
    write_ZYXCT_to_yml ${dataset} "${datadir}/${dataset}_dims.yml"

}


function load_parameters {

    local dataset="${1}"
    local parfile

    parfile="${datadir}/${dataset}.yml"
    eval $( parse_yaml "${parfile}" "" )
    [[ "$2" == '-v' ]] &&
        echo "### dataset details provided:" && parse_yaml "${parfile}"

    parfile="${datadir}/${dataset}_params.yml"
    eval $( parse_yaml "${parfile}" "" )
    [[ "$2" == '-v' ]] && {
        echo "### using parameters:" && parse_yaml "${parfile}"
        echo "###==========================================================================###"
        echo "" ; }
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

    set_channelstems
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
    set_blockstems

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

    nx=$((X/bs + 1))
    ny=$((Y/bs + 1))
    nb=$((nx*ny))

}


function set_blockstems {
    # Generate an array <blockstems> of block identifiers.
    # taking the form "dataset_x-X_y-Y_z-Z"
    # with voxel coordinates zero-padded to 5 digits

    local verbose=$1
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
    echo load_parameters "${dataset}" -v
    echo ''

}


function no_parallelization {
    # Generate directives for processing without parallelization.

    echo "idx=\$((SLURM_ARRAY_TASK_ID-1))"
    echo "filestem=${datadir}/${dataset}"
    echo ''

}


function channel_parallelization {
    # Generate directives for parallel channels.

    echo "idx=\$((SLURM_ARRAY_TASK_ID-1))"
    echo "filestem=${datadir}/${dataset}"
    echo "channelstem=${channeldir}/\${channelstems[idx]}"
    echo ''

}


function block_parallelization {
    # Generate directives for parallel blocks.

    echo "idx=\$((SLURM_ARRAY_TASK_ID-1))"
    echo "filestem=${datadir}/${dataset}"
    echo "blockstem=${blockdir}/\${blockstems[idx]}"
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
    echo "idx=\$((SLURM_ARRAY_TASK_ID-1))"
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
    echo 'inputfile = sys.argv[1]'
    echo 'channel = int(sys.argv[2])'
    echo ''
    echo "from stapl3d.preprocessing import shading"
    echo "shading.estimate_channel(
        inputfile,
        channel,
        noise_threshold=${shading__noise_threshold},
        metric='${shading__metric}',
        quantile_threshold=${shading__quantile_threshold},
        polynomial_order=${shading__polynomial_order},
        outputdir='${datadir}/${dirtree__datadir__shading}',
        )"

}
function get_cmd_shading_estimation {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local file_format="${shading__file_format}"
    echo python "${pyfile}" "\${filestem}.${file_format}" "\${idx}"

}


function get_py_generate_mask {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'inputfile = sys.argv[1]'
    echo 'parameterfile = sys.argv[2]'
    echo ''
    echo "from stapl3d.preprocessing import masking"
    echo "masking.estimate(inputfile, parameterfile)"

}
function get_cmd_generate_mask {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    echo python "${pyfile}" "\${filestem}.ims" "\${filestem}.yml"

}


function get_py_bias_estimation {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'inputfile = sys.argv[1]'
    echo 'channel = int(sys.argv[2])'
    echo 'maskfile = sys.argv[3]'
    echo ''
    echo "from stapl3d.preprocessing import biasfield"
    echo "biasfield.estimate_channel(
        inputfile,
        channel,
        mask_in=maskfile,
        resolution_level=${biasfield__resolution_level},
        downsample_factors=[${biasfield__downsample_factors__z}, ${biasfield__downsample_factors__y}, ${biasfield__downsample_factors__x}, ${biasfield__downsample_factors__c}, ${biasfield__downsample_factors__t}],
        n_iterations=${biasfield__n_iterations},
        n_fitlevels=${biasfield__n_fitlevels},
        n_bspline_cps=[${biasfield__n_bspline_cps__x}, ${biasfield__n_bspline_cps__y}, ${biasfield__n_bspline_cps__z}],
        outputdir='${datadir}/${dirtree__datadir__biasfield}',
        )"

}
function get_cmd_bias_estimation {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local maskfile="\${filestem}${mask__postfix}.h5/mask"
    echo python "${pyfile}" "\${filestem}.ims" "\${idx}" "${maskfile}"

}


function get_py_bias_stack {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'outputstem = sys.argv[1]'
    echo 'inputfiles = sys.argv[2:]'
    echo ''
    echo "from stapl3d.preprocessing.biasfield import stack_bias"
    echo "stack_bias(inputfiles, outputstem)"
    echo ''
    echo "from stapl3d.reporting import zip_parameters"
    echo "zip_parameters('${biasfielddir}/${dataset}', '${datadir}/${dataset}', 'biasfield')"

}
function get_cmd_bias_stack {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local channels_in=()
    local bpf="${bias_estimation__bias_postfix}"
    for chstem in "${channelstems[@]}"; do
        channels_in+=("${biasfielddir}/${chstem}${bpf}")
    done

    echo python "${pyfile}" "\${filestem}${bpf}" "${channels_in[@]}"

    # TODO: also replace by python equiv
    echo ''
    echo pdfunite \
        "${biasfielddir}/${dataset}_ch??${bpf}.pdf" \
        "${datadir}/${dataset}${bpf}.pdf"
    # echo rm "${biasfielddir}/${dataset}_ch??${bpf}.pdf"
    # echo "    rm ${biasfielddir}/${dataset}_ch??${bpf}.pickle"

}


function get_py_bias_apply {

    # read from 4D, write to channel
    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'bias_in = sys.argv[2]'
    echo 'ref_path = sys.argv[3]'
    echo 'outputpath = sys.argv[4]'
    echo 'channel = int(sys.argv[5])'
    echo ''
    echo 'import shutil'
    echo 'shutil.copy2(ref_path, outputpath)'
    echo ''
    echo "from stapl3d.preprocessing import biasfield"
    echo "biasfield.apply_channel(
        image_in,
        bias_in,
        outputpath,
        channel=channel,
        downsample_factors=[${biasfield_apply__downsample_factors__z}, ${biasfield_apply__downsample_factors__y}, ${biasfield_apply__downsample_factors__x}, ${biasfield_apply__downsample_factors__c}, ${biasfield_apply__downsample_factors__t}],
        blocksize_xy=${biasfield_apply__blocksize_xy},
        )"

}
function get_cmd_bias_apply {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local rpf="${dataset__ims_ref_postfix}"
    local bpf="${biasfield__postfix}"

    echo python "${pyfile}" \
        "\${filestem}.ims" \
        "\${filestem}${bpf}.h5/bias" \
        "\${filestem}${rpf}.ims" \
        "\${channelstem}${bpf}.ims" \
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
    echo "make_aggregate(inputfiles, out_path, ref_path)"

}
function get_cmd_ims_aggregate {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    local rpf="${dataset__ims_ref_postfix}"
    local bpf="${biasfield__postfix}"

    echo python "${pyfile}" \
        "\${filestem}${bpf}.ims" \
        "\${filestem}${rpf}.ims" \
        "${channeldir}/${dataset}_ch??${bpf}.ims"

}


function get_cmd_block_segmentation {

    get_cmd_splitblocks
    get_cmd_membrane_enhancement
    get_cmd_segmentation

}


function get_cmd_splitblocks {

    local rpf="${dataset__ims_ref_postfix}"
    local bpf="${bias_estimation__bias_postfix}"

    blockrange_start='$((SLURM_ARRAY_TASK_ID-1))'
    blockrange_end='$((SLURM_ARRAY_TASK_ID))'

    if [ "${splitblocks__bias_apply}" == 'true' ]
    then
        bias_args="--bias_image \${filestem}${bpf}.h5/bias --bias_dsfacs 1 ${dataset__dst} ${dataset__dst} 1"
    else
        bias_args=''
    fi
    echo python -W ignore "${STAPL3D}/channels.py" \
        --inputfile "\${filestem}${bpf}.ims" \
        --blocksize "${blocksize}" \
        --blockmargin "${blockmargin}" \
        --blockrange "${blockrange_start} ${blockrange_end}" \
        --memb_idxs "${dataset__memb_idxs}" \
        --memb_weights "${dataset__memb_weights}" \
        --nucl_idxs "${dataset__nucl_idxs}" \
        --nucl_weights "${dataset__nucl_weights}" \
        --mean_idxs "${dataset__mean_idxs}" \
        --mean_weights "${dataset__mean_weights}" \
        --output_channels "${dataset__dapi_chan}" \
        --outputprefix "${blockdir}/${dataset}" \
        "${bias_args}"

}


function get_py_convert1 {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'blockstem = sys.argv[1]'
    echo 'grp = sys.argv[2]'
    echo 'ids = sys.argv[3]'
    echo "from stapl3d.channels import h5_nii_convert"
    echo "image_in = '{}.h5/{}/{}'.format(blockstem, grp, ids)"
    echo "image_out = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)"
    echo "h5_nii_convert(image_in, image_out, datatype='uint8')"
}
function get_py_convert2 {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'blockstem = sys.argv[1]'
    echo 'grp = sys.argv[2]'
    echo 'ids = sys.argv[3]'
    echo "from stapl3d.channels import h5_nii_convert"
    echo "image_in = '{}_{}-{}.nii.gz'.format(blockstem, grp, ids)"
    echo "image_out = '{}.h5/{}/{}'.format(blockstem, grp, ids)"
    echo "h5_nii_convert(image_in, image_out)"
}
function get_cmd_membrane_enhancement {

    pyfile1="${datadir}/${jobname}_convert1.py"
    eval get_py_convert1 > "$pyfile"
    pyfile2="${datadir}/${jobname}_convert2.py"
    eval get_py_convert2 > "$pyfile"

    echo ''
    echo python "${pyfile1}" "\${blockstem}" 'memb' 'mean'

    echo ''
    echo "${MRbin}/cellPreprocess" \
        "\${blockstem}_memb-mean.nii.gz" \
        "\${blockstem}_memb-preprocess.nii.gz" \
        "${membrane_enhancement__median_filter_par}"
    echo "${MRbin}/multiscalePlateMeasureImageFilter" \
        "\${blockstem}_memb-preprocess.nii.gz" \
        "\${blockstem}_memb-planarity.nii.gz" \
        "\${blockstem}_memb-eigen.mha" \
        "${membrane_enhancement__membrane_filter_par}"

    echo ''
    echo python "${pyfile2}" "\${blockstem}" 'memb' 'preprocess'
    echo python "${pyfile2}" "\${blockstem}" 'memb' 'planarity'

    echo ''
    echo "rm \${blockstem}_memb-eigen.mha"
    echo "rm \${blockstem}_memb-*.nii.gz"

}


function get_cmd_segmentation {

    echo python -W ignore "${STAPL3D}/segmentation/features.py" \
        "\${blockstem}.h5/${segmentation__ids_memb_mask}" \
        "\${blockstem}.h5/${segmentation__ids_memb_chan}" \
        "\${blockstem}.h5/${segmentation__ids_nucl_chan}" \
        "\${blockstem}.h5/${segmentation__ids_dset_mean}" \
        "\${filestem}${segmentation__param_postfix}" \
        -S -o "\${blockstem}"

}


function get_cmd_segmentation_postproc {

    echo pdfunite \
        "${blockdir}/${dataset}_?????-?????_?????-?????_?????-?????_seg-report.pdf" \
        "\${filestem}_seg-report.pdf"
    echo "rm ${blockdir}/${dataset}_?????-?????_?????-?????_?????-?????_seg-report.pdf"

}


function get_py_relabel {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'block_idx = int(sys.argv[2])'
    echo 'maxlabelfile = sys.argv[3]'
    echo 'postfix = sys.argv[4]'
    echo ''
    echo 'from stapl3d.segmentation.zipping import relabel_parallel'
    echo "relabel_parallel(image_in, block_idx, maxlabelfile, pf=postfix)"

}
function get_cmd_relabel {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "${pyfile}"

    eval postfix="\${${stage}__postfix}"
    local ids="segm/${segmentation__segments_ods}"

    local maxlabelfile="${blockdir}/${dataset}_maxlabels.txt"

    echo python "${pyfile}" "\${blockstem}.h5/${ids}" "\${idx}" "${maxlabelfile}" "${postfix}"

}


function get_py_copyblocks {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'block_idx = int(sys.argv[2])'
    echo ''
    echo 'from stapl3d.segmentation.zipping import copy_blocks_parallel'
    echo "copy_blocks_parallel(image_in, block_idx)"

}
function get_cmd_copyblocks {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    eval postfix="\${${stage}__postfix}"
    local ids="segm/${segmentation__segments_ods}${postfix}"

    echo python "${pyfile}" "\${blockstem}.h5/${ids}" "\${idx}"

}


function get_cmd_ziplines { get_cmd_zipping "${axis}" ; }
function get_cmd_zipquads { get_cmd_zipping "${zipquads__axis}" ; }
function get_cmd_zipping {

    local axis="${1}"
    local ids="segm/${segmentation__segments_ods}${zipping__postfix}"

    echo ''
    echo set_images_in "${blockdir}/${dataset}" "${ids}"
    echo set_seamnumbers "${axis}" "\${SLURM_ARRAY_TASK_ID}" "${start_x}" "${start_y}" $((nx - 1)) $((ny - 1))

    echo python -W ignore "${STAPL3D}/segmentation/zipping.py" \
        -i "\${images_in[@]}" \
        -s "${Z}" "${bs}" "${bs}" \
        -m 0 "${bm}" "${bm}" \
        -A ${axis} -L "\${seamnumbers}" \
        -p -l "${blockdir}/${dataset}_maxlabels${zipping__postfix}.txt" \
        -o "${blockdir}/${dataset}"

}
function get_cmd_zipping_postproc {

    echo pdfunite "${blockdir}/${dataset}_reseg_axis?-seam??-j???-report.pdf" "${datadir}/${dataset}_reseg-report.pdf"
    echo rm "${blockdir}/${dataset}_reseg_axis?-seam??-j???-report.pdf"

}


function get_cmd_segmentation_gather { get_cmd_gather "${stage}" ; }
function get_cmd_relabel_gather { get_cmd_gather "${stage}" ; }
function get_cmd_copyblocks_gather { get_cmd_gather "${stage}" ; }
function get_cmd_zipping_gather { get_cmd_gather "${stage}" ; }
function get_cmd_gather {

    local stage="${1}"
    eval postfix="\${${stage}__postfix}"
    local prefix="${segmentation__segments_ods}"
    local ids="segm/${prefix}${postfix}"

    echo set_images_in "${blockdir}/${dataset}" "${ids}"
    echo maxlabelfile="${blockdir}/${dataset}_maxlabels${postfix}.txt"
    echo gather_maxlabels "\${maxlabelfile}"

}


function get_py_copydataset {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'filestem = sys.argv[1]'
    echo 'ids = sys.argv[2]'
    echo 'ods = sys.argv[3]'
    echo ''
    echo "from stapl3d import LabelImage"
    echo "im = LabelImage('{}.h5{}'.format(filestem, ids))"
    echo "im.load(load_data=False)"
    echo "im.file[ods] = im.file[ids]"
    echo "im.close()"

}
function get_cmd_copydataset {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo "python ${pyfile} \${blockstem} ${copydataset__ids} ${copydataset__ods}"

}


function get_py_splitsegments {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'filestem = sys.argv[1]'
    echo 'ids = sys.argv[2]'
    echo ''
    echo 'from stapl3d.segmentation.segment import split_segments'
    echo "split_segments('{}.h5{}'.format(filestem, ids), outputstem=filestem)"

}
function get_cmd_splitsegments {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo "python ${pyfile} \${blockstem} ${splitsegments__ids}"

}


function get_cmd_mergeblocks {

    echo idxf=\`printf %02g \${idx}\`
    echo eval format="\\\${mergeblocks__ids\${idxf}__format}"
    echo eval ids="\\\${mergeblocks__ids\${idxf}__ids}"
    echo ''
    echo mergeblocks_outputpath \${format} \${ids}
    echo ''
    echo set_images_in "${blockdir}/${dataset}" "\${ids}"
    echo ''

    echo python -W ignore "${STAPL3D}/mergeblocks.py" \
        "\${images_in[@]}" \
        "\${out_path}" \
        --blocksize "${blocksize}" \
        --blockmargin "${blockmargin}" \
        -s "${Z}" "${Y}" "${X}"

}


function get_cmd_features {

    channel_names=( 'ch00' 'ch01' 'ch02' 'ch03' 'ch04' 'ch05' 'ch06' 'ch07' )
    ids='segm/labels_memb_del_relabeled_fix'

    local ids="${features__ids}"
    local pf0="${features__segm00}"
    local pf1="${features__segm01}"
    local pf2="${features__segm02}"

    blockrange_start='$((SLURM_ARRAY_TASK_ID-1))'
    blockrange_end='$((SLURM_ARRAY_TASK_ID))'

    echo python -W ignore "${STAPL3D}/segmentation/features.py" \
        --seg_paths \
            "${datadir}/${dataset}_${ids////-}_${pf0}.h5/${ids}_${pf0}" \
            "${datadir}/${dataset}_${ids////-}_${pf1}.h5/${ids}_${pf1}" \
            "${datadir}/${dataset}_${ids////-}_${pf2}.h5/${ids}_${pf2}" \
        --seg_names ${pf0} ${pf1} ${pf2} \
        --data_path "${datadir}/${dataset}${bias_estimation__bias_postfix}.ims" \
        --data_names "${channel_names[@]}" \
        --aux_data_path "${datadir}/${dataset}${generate_mask__mask_postfix}.h5/mask_thr00000_edt" \
        --downsample_factors "1 ${dataset__dsr} ${dataset__dsr}" \
        --csv_path "${featdir}/${dataset}" \
        --blocksize "${blocksize}" \
        --blockmargin "${blockmargin}" \
        --blockrange "${blockrange_start} ${blockrange_end}" \
        --min_labelsize "${features__min_labelsize}" \
        --filter_borderlabels \
        --split_features \
        --fset_morph "${features__featset_morph}" \
        --fset_intens "${features__featset_intens}"

}


function get_py_features_postproc {

    local ids="${features__ids}"
    local pf0="${features__segm00}"
    local pf1="${features__segm01}"
    local pf2="${features__segm02}"

    echo '#!/usr/bin/env python'
    echo ''
    echo 'from stapl3d.segmentation.features import export_regionprops'
    echo "export_regionprops.postprocess_features(
        seg_paths=[
            '${datadir}/${dataset}_${ids////-}_${pf0}.h5/${ids}_${pf0}',
            '${datadir}/${dataset}_${ids////-}_${pf1}.h5/${ids}_${pf1}',
            '${datadir}/${dataset}_${ids////-}_${pf2}.h5/${ids}_${pf2}',
            ],
        blocksize=[$Z, $bs, $bs, $C, $T],
        blockmargin=[0, $bm, $bm, 0, 0],
        blockrange=[],
        csv_dir='${featdir}',
        csv_stem='${dataset}',
        feat_pf='_features',
        segm_pfs=['${pf0}', '${pf1}', '${pf2}'],
        ext='csv',
        min_size_nucl=${features__min_labelsize},
        save_border_labels=True,
        split_features=True,
        fset_morph='${features__featset_morph}',
        fset_intens='${features__featset_intens}',
        )"

}
function get_cmd_features_postproc {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo "python ${pyfile}"

}
