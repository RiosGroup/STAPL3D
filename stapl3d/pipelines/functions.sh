#!/bin/bash


###==========================================================================###
### functions to prepare processing
###==========================================================================###

function load_dataset {
    # --
    ## Go to the dataset directory.
    ##
    ## PARAMETERS:
    ##   projectdir: project directory
    ##   dataset: name of the dataset
    ##
    ## SHELL VARIABLES required:
    ##
    ## SHELL VARIABLES set:
    ##   datadir: dataset directory
    # --

    local projectdir="${1}"
    local dataset="${2}"

    mkdir -p "${projectdir}"

    datadir="${projectdir}/${dataset}"
    mkdir -p "${datadir}" && cd "${datadir}"

    echo ""
    echo " ### project directory is '${projectdir}'"
    echo " ### processing dataset '${dataset}'"
    echo " ### data directory is '${datadir}'"
    echo ""

}


function init_dataset {
    # --
    ## Copy default parameter file to dataset directory.
    ##
    ## PARAMETERS:
    ##
    ## SHELL VARIABLES required:
    ##   STAPL3D: path to STAPL3D installation
    ##   datadir: dataset directory
    ##   dataset: name of the dataset
    ##
    ## SHELL VARIABLES set:
    # --

    cp "${STAPL3D}/pipelines/params.yml" "${datadir}/${dataset}.yml"

    echo ""
    echo " ### copied default parameter file to ${datadir}/${dataset}.yml"
    echo ""

    # set_ZYXCT_ims -v "${datadir}/${dataset}.ims"
    # write_ZYXCT_to_yml ${dataset} "${datadir}/${dataset}_dims.yml"

}


function load_parameters {
    # --
    ## Load the processing parameters from a yaml parameter file.
    ##
    ## PARAMETERS:
    ##   dataset: name of the dataset
    ##   verbose: use '-v' for printing messages
    ##   parfile: path to the parameter file (default: datadir/dataset.yml)
    ##
    ## SHELL VARIABLES required:
    ## ...TODO
    ##
    ## SHELL VARIABLES set:
    ## ...TODO
    # --

    local dataset="${1}"
    local verbose="${2}"
    local parfile="${3}"

    [[ -z $parfile ]] && parfile="${datadir}/${dataset}.yml"
    eval $( parse_yaml "${parfile}" "" )

    [[ "${verbose}" = '-v' ]] && {
        echo ""
        echo " ### parameters imported from parameter file:"
        echo ""
        echo `parse_yaml "${parfile}"` | tr ' ' '\n' | awk '{print "\t", $0}'
        echo "" ; }

    dataset_shading="${dataset}${shading__params__postfix}"
    dataset_stitching="${dataset_shading}${stitching__params__postfix}"
    dataset_biasfield="${dataset_stitching}${biasfield__params__postfix}"
    dataset_preproc="${dataset_biasfield}"

    set_dims "${image_in}" "${datadir}/${dataset}_dims.yml" -v

    set_blocks "${axlab[@]}"
    [[ "${verbose}" = '-v' ]] && {
        echo ""
        echo " ### block parallelization:"
        echo ""
        printf "\t%-12s%-8s\n" "axlab: " "${axlab[*]}"
        printf "\t%-12s%-8s\n" "sizes: " "${blocksize[*]}"
        printf "\t%-12s%-8s\n" "margins: " "${blockmargin[*]}"
        printf "\t%-12s%-8s\n" "grid: " "${blockgrid[*]}"
        printf "\t%-12s%-8s\n" "Nblocks: " "${nb}"
        echo "" ; }

}


function parse_yaml {
    # --
    ## Parse a (4-space-indented) yaml-file and create variable for entries.
    ##
    ## from https://gist.github.com/pkuczynski/8665367
    ## TODO: make two-spaced indentation possible
    ##  (each double-space creates an underscore in the variable name)
    ## consider: https://github.com/0k/shyaml
    ##
    ## PARAMETERS:
    ##   filepath: path to yaml file
    ##   prefix: prefix to the variable names
    ##
    ## SHELL VARIABLES required:
    ##
    ## SHELL VARIABLES set:
    ##   all parsed yaml-entries in the format <prefix>__<key0>__<key1>__<key2>
    # --

    local filepath=$1
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')

    sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $filepath |
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


function set_dims {

    local image_in="${1}"
    local dimsfile="${2}"
    local verbose="${3}"

    local vars=(Z Y X C T M tZ tY tX elsize_z elsize_y elsize_x)
    local varname var
    local im_M im_tZ im_tY im_tX im_Z im_Y im_X im_C im_T
    local im_elsize_z im_elsize_y im_elsize_x

    [[ "${verbose}" = '-v' ]] && {
        echo ""
        echo " ### dataset dimensions:"
        echo "" ; }

    # TODO: only execute if necessary

    axlab=($(echo `get_axlab "${image_in}"` | fold -w1))
    [[ ! -f "${dimsfile}" ]] && {
        eval `get_shape "${image_in}"`
        eval `get_elsizes "${image_in}"`
        eval `get_tiled_dims "${image_in}"` ; }

    for varname in "${vars[@]}"; do
        eval var="\$dataset__${varname}"
        if ! [ -z "${var}" ]
        then  # already specified as environment variable (eg imported from parameterfile)
            src="environment variable dataset__${varname}"
        elif [ -f "${dimsfile}" ]
        then  # import via previously written dims-yml
            eval var=`cat "${dimsfile}" | grep "^$varname: " | sed "s/$varname: //g"`
            src="${dimsfile}"
        else  # import via STAPL3D python functions
            eval var="\$im_${varname}"
            src="${image_in}"
        fi
        eval $varname="${var}"
        [[ "${verbose}" = '-v' ]] && {
            printf "\t%-8s = %-20s loaded from %s \n" "$varname" "$var" "$src" ; }
    done

    echo ""

    [[ -f "${dimsfile}" ]] || write_dims_to_yml "${dimsfile}"

}


function write_dims_to_yml {

    local parfile="${1}"

    local vars=(Z Y X C T M tZ tY tX elsize_z elsize_y elsize_x)
    local varname var

    > "${parfile}"
    for varname in "${vars[@]}"; do
        eval var=\$$varname
        echo "$varname: ${var}" >> "${parfile}"
    done

}


function get_tiled_dims {

    local filepath="${1}"
    local dims

    source "${CONDA_SH}"
    conda activate ${submit_defaults__submit__conda_env}
    dims=`python -c "import os; from stapl3d.preprocessing import shading; iminfo = shading.get_image_info('${filepath}'); print('im_M={};im_tZ={};im_tY={};im_tX={};'.format(int(iminfo['nstacks']), int(iminfo['nplanes']), int(iminfo['ncols']), int(iminfo['nrows'])));"`
    conda deactivate
    echo $dims

}
function get_shape {

    local filepath="${1}"
    local dims

    source "${CONDA_SH}"
    conda activate ${submit_defaults__submit__conda_env}
    dims=`python -c "import os; from stapl3d import Image; im = Image('${filepath}', permission='r'); im.load(); print(';'.join([f'im_{al.upper()}={sh}' for sh, al in zip(im.dims, im.axlab)])); im.close();"`
    conda deactivate
    echo $dims

}
function get_elsizes {

    local filepath="${1}"

    source "${CONDA_SH}"
    conda activate ${submit_defaults__submit__conda_env}
    elsize=`python -c "import os; from stapl3d import Image; im = Image('${filepath}', permission='r'); im.load(); print(';'.join([f'im_elsize_{al}={es}' for es, al in zip(im.elsize, im.axlab)]));"`
    conda deactivate
    echo $elsize

}
function get_axlab {

    local filepath="${1}"

    source "${CONDA_SH}"
    conda activate ${submit_defaults__submit__conda_env}
    axlab=`python -c "import os; from stapl3d import Image; im = Image('${filepath}', permission='r'); im.load(); print(im.axlab); im.close();"`
    conda deactivate
    echo $axlab

}


function set_ZYXCT_ims {
    # --
    ## Set the dimension variables for an Imarisfile.
    # --

    local verbose=$1
    shift 1
    local imarisfile=${*}

    Z=`get_dim_ims Z "${imarisfile}"`
    Y=`get_dim_ims Y "${imarisfile}"`
    X=`get_dim_ims X "${imarisfile}"`
    C=`get_dim_ims C "${imarisfile}"`
    T=`get_dim_ims T "${imarisfile}"`
    if [ "$verbose" = "-v" ]; then
        echo $Z $Y $X $C $T
    fi


}
function get_dim_ims {
    # --
    ## Extract dataset dimension from an Imarisfile.
    # --

    local dim=${1}
    shift 1
    local imarisfile="${*}"

    if [[ "$dim" = 'T' ]]; then
        echo `h5ls "${imarisfile}/DataSet/ResolutionLevel 0" | wc -l`
    elif [[ "$dim" = 'C' ]]; then
        echo `h5ls "${imarisfile}/DataSet/ResolutionLevel 0/TimePoint 0" | wc -l`
    else
        echo `find_dim_from_h5 "${dim}" "${imarisfile}"`
    fi

}
function find_dim_from_h5 {
    # --
    ## Extract dataset dimension from an Imarisfile for Z/Y/X.
    # --


    local dim=${1}
    shift 1
    local imarisfile="${*}"

    local line=`h5ls -v ${imarisfile}/DataSetInfo | grep -A 2 "Attribute: ${dim}"  | tail -n 1 | grep "Data" | sed 's/ //g'`
    if [[ "$line" = "Data:" ]]; then
        line=`h5ls -v ${imarisfile}/DataSetInfo | grep -A 3 "Attribute: ${dim}" | tail -n 1  | sed 's/(0)//'`
    else
        line=`h5ls -v ${imarisfile}/DataSetInfo | grep -A 2 "Attribute: ${dim}" | tail -n 1 | tr -d 'Data:'`
    fi
    echo "$line" | tr -d ' ' | tr -d , | tr -d '"'

}
function find_axlab_from_hdf5 {
    # --
    ## Extract dataset axis labels from an STAPL3D Image HDF5.
    # --


    local hdf5path="${*}"

    local line=`h5ls -v ${hdf5path} | grep -A 2 "Attribute: DIMENSION_LABELS scalar" | tail -n 1 | grep "Data" | sed 's/ //g'`
    echo "$line" | tr -d ' ' | tr -d , | tr -d '"' | tr -d 'Data:'

}
function find_dims_from_hdf5 {
    # --
    ## Extract dataset dimensions from an STAPL3D Image HDF5.
    # --


    local hdf5path="${*}"

    echo `h5ls "${hdf5path}" | cut -d "{" -f2 | tr -d "," | tr -d '}'`

}


function set_blocks {

    local axlab=("$@")

    set_blocksizes "${axlab[@]}"  # FIXME/TODO: defaults => full dataset ranges
    set_blockdims
    set_blockstems "${dataset_preproc}"

}
function set_blocksizes {

    local axlab=("$@")

    unset blocksize
    blocksize=()

    unset blockmargin
    blockmargin=()

    unset blockgrid
    blockgrid=()

    # FIXME/TODO: defaults
    for al in "${axlab[@]}"; do
        eval bs="\$blocks__blockinfo__blocksize__${al}"
        blocksize+=( $bs )
        eval bm="\$blocks__blockinfo__blockmargin__${al}"
        blockmargin+=( $bm )
        #eval upper="\$${al^}"  # not on MAC old bash version
        eval upper="\$`echo $al | tr '[:lower:]' '[:upper:]'`"
        bg=`python -c "from math import ceil; print(int(ceil(${upper}./$bs.)))"`
        blockgrid+=( $bg )
    done

}
function multiply_array {
    local IFS='*'
    echo "$(( $* ))"
}
function set_blockdims {
    # --
    ## Set the variables holding the number of blocks, rows and columns.
    ## FIXME: evaluates to 0 on dataset smaller than blocksize
    ##
    ## PARAMETERS:
    ##
    ## SHELL VARIABLES required:
    ##
    ## SHELL VARIABLES set:
    ##
    # --

    local axlab=("$@")

    i=0
    for al in "${axlab[@]}"; do
        bg="${blockgrid[$i]}"
        eval n${al}="${blockgrid[$i]}"
        i=$(($i+1))
    done

    nb=`multiply_array "${blockgrid[@]}"`

}
function set_blockstems {

    local dataset=$1
    local verbose=$2
    local block_id dstem

    unset block_ids
    block_ids=()

    unset blockstems
    blockstems=()

    for i in `seq -f "%05g" 0 ${nb}`; do
        block_id="B$i"
        dstem="${dataset}_${block_id}"
        blockstems+=( "$dstem" )
        [[ "$verbose" = "-v" ]] && echo "$dstem"

        block_ids+=( "$block_id" )
        [[ "$verbose" = "-v" ]] && echo "$block_id"
    done

    datastems=( "${blockstems[@]}" )

}




###==========================================================================###
### some utils
###==========================================================================###


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


function get_block_id {
    # --
    ## Get a block identifier from coordinates.
    ## NOTE: format xxxxx-XXXXX_yyyyy-YYYYY_zzzzz-ZZZZZ
    # --

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


function set_filepaths {
    # --
    ## Generate an array 'filepaths' with filepaths matching a pattern.
    ## NOTE: '{f}' is matched as wildcard.
    # --

    local filepat=$1

    unset filepaths
    filepaths=()

    filepat=$( echo "${filepat}" | sed "s#{f}#\*#" )

    for fp in `ls $filepat`; do
        filepaths+=( "$fp" )
    done

}


function check_dims {
    # --
    ## Check if the variable is set and is a number.
    # --

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


function get_coords_upper {
    # --
    ## Get upper coordinate of the block.
    ## NOTE: adds blocksize and margin, bounded by the dimension's extent
    # --

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
    # --
    ## Get lower coordinate of the block.
    ## NOTE: subtracts margin, with 0-bound
    # --

    local co=$1
    local margin=$2

    co=$(( co - margin )) &&
        co=$(( co > 0 ? co : 0 ))

    echo "$co"

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
    # --
    ## Generate an array variable with paths to a set of h5-datasets.
    ## TODO: make the regex an argument/variable and merge with 'set_images_in_stardist'
    # --

    local filestem="${1}"
    local ids="${2}"
    local verbose="${3}"

    images_in=()
    for fname in `ls ${filestem}_?????-?????_?????-?????_?????-?????.h5`; do
        images_in+=( ${fname}/$ids )
    done
    if [ "${verbose}" = '-v' ]
    then
        for image_in in "${images_in[@]}"; do
            echo ${image_in}
        done
    fi

}


# UNUSED?
function set_images_in_stardist {
    # --
    ## Generate an array variable with paths to a set of h5-datasets.
    ## TODO: make the regex an argument/variable and merge with 'set_images_in'
    # --

    local filestem="${1}"
    local ids="${2}"
    local verbose="${3}"

    images_in=()
    for fname in `ls ${filestem}_block?????.h5`; do
        images_in+=( ${fname}/$ids )
    done
    if [ "${verbose}" = '-v' ]
    then
        for image_in in "${images_in[@]}"; do
            echo ${image_in}
        done
    fi
}


function find_missing_datastems {
    # --
    ## Find failed block-processing jobs by checking for generated files.
    # --

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


#UNUSED?
function set_channels_in {
    # --
    ## Generate an array with filenames for channel-separated Imaris files.
    ## TODO: generalize suffix/extension?
    # --

    local filestem="${1}"

    channels_in=( )
    for fname in `ls ${filestem}_ch??.ims`; do
        channels_in+=( ${fname} )
    done

}


function gather_maxlabels {
    # --
    ## Write a text file with the maximal label id for each h5 LabelImage.
    ## NOTE: 'maxlabel' attribute has to be set on the datasets.
    # --

    local maxlabelfile="${1}"

    > "${maxlabelfile}"
    for im in ${images_in[@]}; do
        maxlabel=`h5ls -v ${im} | grep -A 2 "maxlabel" | tail -n 1 | tr -d 'Data:' | tr -d ' '`
        echo "${maxlabel}" >> "${maxlabelfile}"
    done

}


function set_zipquads {
    # --
    ## Generate a array of zipquad linear indices.
    # --

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
    # --
    ## Set the indices of the ziplines tackled in a specific zipping job.
    # --

    local axis="${1}"
    local TASK_ID="${2}"
    local start_x="${3}"
    local start_y="${4}"
    local nseams_x="${5}"
    local nseams_y="${6}"

    if [ "${axis}" = "0" ]
    then  # zipquads
        local idx=$((TASK_ID - 1))
        set_zipquads "${start_x}" "${start_y}" 2 2 "${nseams_x}" "${nseams_y}"
        local seamnumber="${zipquads[idx]}"
	    local seam_x=$((seamnumber / nseams_y))
    	local seam_y=$((seamnumber % nseams_y))
        seamnumbers="-1 ${seam_y} ${seam_x}"
    elif [ "${axis}" = "1" ]
    then  # y-ziplines
        seamnumbers="-1 $((TASK_ID - 1)) -1"
    elif [ "${axis}" = "2" ]
    then  # x-ziplines
        seamnumbers="-1 -1 $((TASK_ID - 1))"
    else
        seamnumbers="-1 -1 -1"
    fi


}


function set_idss {
    # --
    ## Generate an array with the mergeblock volumes.
    ## (as present in bash variables parsed from the yaml parameter file).
    ## (FIXME: find a safer way to implement this)
    # --

    local pattern=$1
    local sep=$2

    unset idss
    idss=( `( set -o posix ; set ) | grep "^${pattern}" | cut -f1 -d"${sep}"` )

}


# UNUSED?
function mergeblocks_outputpath {
    # --
    ## Set a target path for block merging; copy Imaris ref file to output path.
    # --

    local format=$1
    local ids=$2

    if [ "${format}" = "ims" ]
    then
        out_path="${datadir}/${dataset}_${ids////-}.ims"
        cp ${datadir}/${dataset}${dataset__ims_ref_postfix}.ims ${out_path}
    elif [ "${format}" = "h5" ]
    then
        out_path="${datadir}/${dataset}_${ids////-}.h5/${ids}"
    else
        out_path="${datadir}/${dataset}_${ids////-}.h5/${ids}"
    fi

}


# UNUSED?
function get_blockstem_index {
    # --
    ## Locate a block identifier in an array and return it's index.
    # --

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
for i in $n_procs; do
    ./procs[${i}] &
    pids[${i}]=$!
done

function submit {
    # --
    ## Submit script to the job scheduler.
    # --

    local scriptfile="${1}"
    local dep_jid="${2}"

    case "${compute_env}" in
        'LOCAL')
            # dependency and array specifiers
            [[ -z $dep_jid ]] && dep='' || dep="$dep_jid"

            if [ "$dep_jid" = 'h' ]
            then  # dry run: print command
                echo "not submitting $scriptfile"
                echo "source $scriptfile"
            else  # run job and save pid in 'jid' variable
                source $scriptfile
                jid=$!
                echo "submitted $scriptfile as ${jid} with ${dep}"
            fi
            ;;
        'SGE')
            # dependency and array specifiers
            [[ -z $dep_jid ]] && dep='' || dep="-hold_jid $dep_jid"
            [[ -z $array ]] && arrayspec='' || arrayspec="-t $range"

            if [ "$dep_jid" = 'h' ]
            then  # dry run: print command
                echo "not submitting $scriptfile"
                echo "qsub -cwd $arrayspec $dep $scriptfile"
            else  # submit job and save job_id in 'jid' variable
                jid=$( qsub -cwd $arrayspec $dep $scriptfile )
                jid=`echo $jid | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`
                echo "submitted $scriptfile as ${jid} with ${dep}"
            fi
            ;;
        'SLURM')
            # dependency specifier
            [[ -z $dep_jid ]] && dep='' || dep="--dependency=afterok:$dep_jid"

            if [ "$dep_jid" = 'h' ]
            then  # dry run: print command
                echo "not submitting $scriptfile"
                echo "sbatch --parsable $dep $scriptfile"
            else  # submit job and save job_id in 'jid' variable
                jid=$( sbatch --parsable $dep $scriptfile )
                echo "submitted $scriptfile as ${jid} with ${dep}"
            fi
            ;;
    esac

}


function bash_directives {
    # --
    ## Generate bash directives for submission script.
    # --

    echo '#!/usr/bin/env bash'
    # echo 'set -ux'
    echo ''

}


function sbatch_directives {
    # --
    ## Generate sbatch directives for submission script.
    # --

    expand_submit_pars "$@"

    # Slurm directives.
    echo "#SBATCH --job-name=$jobname"
    echo "#SBATCH --partition=$partition"
    [[ "$partition" = 'gpu' ]] &&
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

    # Set the array job TASK_ID (1-based) and idx for python (0-based).
    echo TASK_ID="\${SLURM_ARRAY_TASK_ID}"
    echo idx="\$((TASK_ID - 1))"
    echo ''

}


function pbs_directives {
    # --
    ## Generate pbs directives for submission script.
    # --

    expand_submit_pars "$@"

    # SGE directives.
    echo "#PBS -N EM_$jobname"
    echo "#PBS -l mem=$mem"
    echo "#PBS -l walltime=$wtime"
    echo "#PBS -l nodes=$nodes:ppn=$tasks"
    [[ -z $array ]] || echo "#PBS -t $range"
    echo "#PBS -V"
    echo ''

    # Set the array job TASK_ID (1-based) and idx for python (0-based).
    echo TASK_ID="\${SGE_TASK_ID}"
    echo idx="\$((TASK_ID - 1))"
    echo ''

}


function conda_cmds {
    # --
    ## Generate conda directives for submission script.
    # --

    local action=$1

    eval conda_env="\$${stage}__${step}__submit__conda_env"
    [[ -z "${conda_env}" ]] && conda_env=${submit_defaults__submit__conda_env}
    if [ ! -z $conda_env ]
    then
        [[ "${action}" = "activate" ]] && echo source "${CONDA_SH}"
        [[ "${action}" = "activate" ]] && echo conda "${action}" "${conda_env}"
        [[ "${action}" = "deactivate" ]] && echo conda "${action}"
        echo ''
    fi

}


function base_cmds {
    # --
    ## Generate basic dataset directives.
    # --

    # Load the bash functions, dataset and parameters on the compute node.
    echo ''
    echo source "${STAPL3D}/pipelines/functions.sh"
    echo load_dataset "${projectdir}" "${dataset}"
    echo load_parameters "${dataset}" -v
    echo ''

    # # Special case for dual-mode parallelization (shading correction).
    # if [ "$array" = 'channel_plane' ]
    # then
    #     echo ch_idx="\$((idx/Z))"
    #     echo pl_idx="\$((idx%Z))"
    # fi

}


function finishing_directives {
    # --
    ## Generate directives for finishing up the job.
    # --

    # Write job accounting information.
    # echo "sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode -j \${SLURM_JOB_ID} > ${subfile}.a\${SLURM_JOB_ID}"  # TODO $PBS_JOBID

    echo ''

}


###==========================================================================###
### functions for job generation [generalized]
###==========================================================================###

function set_submit_pars {
    # --
    ## Create submit_pars array holding job specification.
    # --

    local stage=$1
    local step=$2

    # Create basic array.
    unset submit_pars
    submit_pars=()
    add_submit_par $stage $step 'array' 'no'
    add_submit_par $stage $step 'nodes' '1'
    add_submit_par $stage $step 'tasks' '1'
    add_submit_par $stage $step 'mem' '10G'
    add_submit_par $stage $step 'wtime' '01:00:00'

    # Generate array job specifier.
    #    Switch between parallelization modes.
    # TODO: parallelization modes in plural to conform to Python pipeline
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
        # 'channel_plane')
        #     array_stop="$((C*Z))"
        #     ;;
        'block')
            array_stop="${#blockstems[@]}"
            ;;
        'filepaths')
            array_stop="${#filepaths[@]}"
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
            # FIXME: # invalid for new pars -> needs to be set as stage:step:submit:stop in parameter file for now
            #set_idss "${stage}__params__ids..__ids=" '='
            #array_stop="${#idss[@]}"
            ;;
    esac

    unset array_range
    # Check if array_range is specified for $stage_$step
    eval array_range=\$${stage}__${step}__submit__array_range
    if [ -z "${array_range}" ]
    then
        # Build the specifier 'array_range' variable.
        build_array_range $stage $step 'start' '1' ''
        build_array_range $stage $step 'stop'  '1' '-'
        build_array_range $stage $step 'step'  '1' ':'
        build_array_range $stage $step 'simul' '0' '%'
    fi
    submit_pars+=( $array_range )

    # Add cpu/gpu specifications.
    add_submit_par $stage $step 'partition' 'cpu'
    add_submit_par $stage $step 'gpus_per_node' '1'

}


function add_submit_par {
    # --
    ## Add submit parameter to array (default or from shell variable if set).
    # --

    local stage="${1}"
    local step="${2}"
    local varname="${3}"
    local default="${4}"

    eval var=\$${stage}__${step}__submit__${varname}
    [[ -z "${var}" ]] && eval var=\$submit_defaults__submit__${varname}
    [[ -z "${var}" ]] && var="${default}"

    submit_pars+=( ${var} )

}


function build_array_range {
    # --
    ## Build a SLURM array job specifier (start-stop:step%simul).
    # --

    local stage="${1}"
    local step="${2}"
    local varname="${3}"
    local default="${4}"
    local divider="${5}"

    eval var=\$array_${varname}  # set in set_submit_pars
    [[ -z "${var}" ]] && eval var=\$${stage}__${step}__submit__${varname}  # specified in parameter file stage:step entry
    [[ -z "${var}" ]] && eval var=\$submit_defaults__submit__${varname}  # specified in parameter file default entry
    [[ -z "${var}" ]] && var="${default}"  # default

    [[ "${var}" != "0" ]] && array_range+="${divider}${var}"

}


function expand_submit_pars {
    # --
    ## Expand the cells from the submit_pars array to variables.
    # --

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
    #---
    ## Create a bash script for job submission.
    ## PARAMETERS:
    ##   stage:
    ##   step:
    ##   OPT: submit_pars: array nodes tasks mem wtime array_range partition gpus_per_node
    ## SHELL VARIABLES required:
    ##   dataset_alias
    ##   datadir
    ##   compute_env
    ## SHELL VARIABLES set:
    ##   jobname
    ##   subfile
    ##   submit_pars
    #---

    local stage=$1
    local step=$2
    [[ "${3}" = "LOCAL" ]] && { local compute_env=$3 ; shift 3 ; } || shift 2

    # Generate jobname and script filepath.
    jobname="go_${dataset__alias}_${stage}_${step}"
    subfile="${datadir}/${jobname}.sh"

    # Set the submission parameters.
    [[ "${stage}" = ziplines* ]] && stage='ziplines'  # Reduce zipping stage.
    [[ "${stage}" = zipquads* ]] && stage='zipquads'  # Reduce zipping stage.
    # Initialize from arguments or set from variables / defaults.
    local submit_pars=( "$@" )
    [[ ${#submit_pars[@]} -eq 0 ]] && set_submit_pars ${stage} ${step}

    # Write submission script.
    bash_directives > "${subfile}" # shebang

    case "${compute_env}" in
        'SGE')
            pbs_directives "${submit_pars[@]}" >> "${subfile}"
            base_cmds >> "${subfile}"
            ;;
        'SLURM')
            sbatch_directives "${submit_pars[@]}" >> "${subfile}"
            base_cmds >> "${subfile}"
            ;;
    esac

    conda_cmds "activate" >> "${subfile}"  # activate conda environment

    eval get_cmd_general_${compute_env} >> "${subfile}"  # processing function

    conda_cmds "deactivate" >> "${subfile}"  # deactivate conda environment

    finishing_directives >> "${subfile}"

    echo "${subfile}"  # return filepath

}


###==========================================================================###
### functions to generate bash commands
# function get_py_<module>_<step> {
#     get_py_header
#     get_py_<module>
#     echo "deshader.${step}(planes=[idx])"
# }
# function get_cmd_<module>_<step> { get_cmd "\${idx}" ; }

###==========================================================================###



function get_py_general_LOCAL {
    get_py_header
    eval get_py_${stage}
    echo "${stage}.${step}()"
}
function get_py_general_SLURM {
    get_py_header
    eval get_py_${stage}
    echo "${stage}.${step}(${submit_pars[0]}=[idx])"
}
function get_py_general_SGE {
    get_py_header
    eval get_py_${stage}
    echo "${stage}.${step}(blocks=[idx])"
}
function get_cmd_general_LOCAL { get_cmd ; }
function get_cmd_general_SLURM { get_cmd "\${idx}" ; }
function get_cmd_general_SGE { get_cmd "\${idx}" ; }


function get_py_splitter {
    echo "from stapl3d import blocks"
    echo "${stage} = blocks.Splitt3r(image_in, parameter_file)"
}



## TODO: sensible submit defaults for each step => autoselect of parallelization in blocks/channels/stacks/planes etc
# in add_submit_par: if ${stage}__${step}__submit__${varname} not specified ==> default
# now goes to global default, we want a dictionary of specifics for steps here


function get_cmd {
    write_pyfile
    echo_pyfile "$@"
}


function write_pyfile {
    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage}_${step} > "${pyfile}"
}
# function write_pyfile {
#     pyfile="${datadir}/${jobname}.py"
#     eval get_py_general_${compute_env} > "${pyfile}"
# }


function echo_pyfile {
    echo python "${pyfile}" "${image_in}" "${parfile}" "$@"
}


function get_py_header {
    # --
    ## Print initial common stanza of python script.
    ##
    ## It defines the parameters:
    ##   image_in: inputfile
    ##   parameter_file: the yaml file
    ##   TODO?: prefix:
    ##   idx: index for parallelization
    ##   idx2: aux index for dual parallelization (TODO: remove to simplify)
    ##
        # image_in : string
        #     Path to dataset.
        # parameter_file : string
        #     Path to yaml parameter file.
        # module_id : string
        #     Name of the STAPL3D module.
        # step_id: string
        #     Identifier of the yaml parameterfile entry.
        # directory : string
        #     Name of output subdirectory.
        # prefix : string
        #     Output prefix.
        # datadir : string
        #     Override for datadir [when image_in used to define alternate input].
        # max_workers : int
        #     Maximal number of cores to use for processing.
        # verbosity : int
        #     Verbosity level.
    ##
    # --

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'if len(sys.argv) > 3:'
    echo '    idx = int(sys.argv[3])'
    echo 'if len(sys.argv) > 4:'
    echo '    idx2 = int(sys.argv[4])'

}


function get_py_shading {
    echo ''
    echo "from stapl3d.preprocessing import shading"
    echo "deshad3r = shading.Deshad3r(image_in, parameter_file)"
}
function get_py_shading_estimate {
    get_py_header
    get_py_shading
    echo "deshad3r.${step}(planes=[idx])"
}
function get_cmd_shading_estimate { get_cmd "\${idx}" ; }
function get_py_shading_postprocess {
    get_py_header
    get_py_shading
    echo "deshad3r.${step}(channels=[idx])"
}
function get_cmd_shading_postprocess { get_cmd "\${idx}" ; }
function get_py_shading_postprocess {
    get_py_header
    get_py_shading
    echo "deshad3r.${step}()"
}
function get_cmd_shading_postprocess { get_cmd "" ; }

function get_py_shading_apply {
    get_py_header
    get_py_shading
    echo "deshad3r.${step}(stacks=[idx])"
}
function get_cmd_shading_apply { get_cmd "\${idx}" ; }


function get_py_stitching {
    echo ''
    echo "from stapl3d.preprocessing import stitching"
    echo "stitcher = stitching.Stitcher(image_in, parameter_file)"
    echo "stitcher.elsize = [$elsize_z, $elsize_y, $elsize_x]"
}
function get_py_stitching_prep {
    get_py_header
    get_py_stitching
    echo "stitcher.${step}()"
}
function get_cmd_stitching_prep  { get_cmd "" ; }
function get_py_stitching_load {
    get_py_header
    get_py_stitching
    echo "stitcher.${step}(channels=[idx])"
}
function get_cmd_stitching_load { get_cmd "\${idx}" ; }
function get_py_stitching_calc {
    get_py_header
    get_py_stitching
    echo "stitcher.${step}()"
}
function get_cmd_stitching_calc { get_cmd "" ; }
function get_py_stitching_fuse {
    get_py_header
    get_py_stitching
    echo "stitcher.${step}(channels=[idx])"
}
function get_cmd_stitching_fuse { get_cmd "\${idx}" ; }
function get_py_stitching_postprocess {
    get_py_header
    get_py_stitching
    echo "stitcher.${step}()"
}
function get_cmd_stitching_postprocess { get_cmd "" ; }


function get_py_mask {
    echo "from stapl3d.preprocessing import masking"
    echo "masker = masking.Masker(image_in, parameter_file)"
}
function get_py_mask_estimate {
    get_py_header
    get_py_mask
    echo "masker.${step}()"
}
function get_cmd_mask_estimate { get_cmd "" ; }


### >>> TODO
function get_py_splitchannels {

    get_py_header
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
### <<< TODO


function get_py_biasfield {
    echo "from stapl3d.preprocessing import biasfield"
    echo "homogenizer = biasfield.Homogenizer(image_in, parameter_file)"
}
function get_py_biasfield_estimate {
    get_py_header
    get_py_biasfield
    echo "homogenizer.${step}(channels=[idx])"
}
function get_cmd_biasfield_estimate { get_cmd "\${idx}" ; }
function get_py_biasfield_postprocess {
    get_py_header
    get_py_biasfield
    echo "homogenizer.${step}()"
}
function get_cmd_biasfield_postprocess { get_cmd "" ; }
function get_py_biasfield_apply {
    get_py_header
    get_py_biasfield
    echo "homogenizer.${step}(channels=[idx])"
}
function get_cmd_biasfield_apply { get_cmd "\${idx}" ; }


### >>> TODO
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
### <<< TODO


# function get_py_splitter {
#     echo "from stapl3d import blocks"
#     echo "splitt3r = blocks.Splitt3r(image_in, parameter_file)"
# }
# function get_py_splitter_split {
#     get_py_header
#     get_py_splitter
#     echo "splitt3r.${step}(blocks=[idx])"
# }
# function get_cmd_splitter_split { get_cmd "\${idx}" ; }


function get_py_membrane_enhancement {
    echo "from stapl3d.segmentation import enhance"
    echo "enhanc3r = enhance.Enhanc3r(image_in, parameter_file)"
}
function get_py_membrane_enhancement_estimate {
    get_py_header
    get_py_membrane_enhancement
    echo "enhanc3r.${step}(blocks=[idx])"
}
function get_cmd_membrane_enhancement_estimate { get_cmd "\${idx}" ; }



function get_py_segmentation {
    echo "from stapl3d.segmentation import segment"
    echo "segment3r = segment.Segment3r(image_in, parameter_file)"
}
function get_py_segmentation_estimate {
    get_py_header
    get_py_segmentation
    echo "segment3r.${step}(blocks=[idx])"
}
function get_cmd_segmentation_estimate { get_cmd "\${idx}" ; }
function get_py_segmentation_postproc {
    get_py_header
    get_py_segmentation
    echo "segment3r.${step}()"
}
function get_cmd_segmentation_postproc { get_cmd "" ; }


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
    echo 'peaks_thr = float(sys.argv[14])'
    echo 'outputstem = sys.argv[15]'
    echo 'images_in = sys.argv[16:]'
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
        peaks_thr=peaks_thr,
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
        "${zipping__params__peaks_thr}" \
        "${blockdir}/${dataset_preproc}" \
        "\${images_in[@]}"

}


# TODO: move glob inside function
function get_py_zipping_postproc {

    echo '#!/usr/bin/env python'
    echo ''
    echo "from glob import glob"
    echo "from stapl3d.reporting import merge_reports"  # TODO: change to _merge_reports from Stapl3r
    echo "pdfs = glob('${blockdir}/${dataset_preproc}_reseg_axis?-seam???-j???.pdf')"
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

    echo "rm ${blockdir}/${dataset_preproc}_reseg_axis?-seam???-j???.pdf"

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




function get_py_subsegmentation {
    echo "from stapl3d.segmentation import segment"
    echo "subsegment3r = segment.Subsegment3r(image_in, parameter_file)"
}
function get_py_subsegmentation_estimate {
    get_py_header
    get_py_subsegmentation
    echo "subsegment3r.${step}(blocks=[idx])"
}
function get_cmd_subsegmentation_estimate { get_cmd "\${idx}" ; }


function get_py_merger {
    echo "from stapl3d import blocks"
    echo "${stage} = blocks.Merg3r(image_in, parameter_file)"
}
function get_py_merger_merge {
    get_py_header
    get_py_merger
    echo "${stage}.${step}(volumes=[idx])"
}
function get_cmd_merger_merge { get_cmd "\${idx}" ; }



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


function get_py_registration {
    echo ''
    echo "from stapl3d.segmentation import stardist_nuclei"  # FIXME!!!
    echo "registrat3r = registration.Registrat3r(image_in, parameter_file)"
}
function get_py_registration_estimate {
    get_py_header
    get_py_registration
    echo "registrat3r.${step}()"
}
function get_cmd_registration_estimate { get_cmd ; }
function get_py_registration_apply {
    get_py_header
    get_py_registration
    echo "registrat3r.${step}()"
}
function get_cmd_registration_apply { get_cmd ; }
function get_py_registration_postprocess {
    get_py_header
    get_py_registration
    echo "registrat3r.${step}()"
}
function get_cmd_registration_postprocess { get_cmd ; }


function get_py_stardist {
    echo ''
    echo "from stapl3d.segmentation import stardist_nuclei"
    echo "stardist3r = stardist_nuclei.StarDist3r(image_in, parameter_file)"
}
function get_py_stardist_train {
    get_py_header
    get_py_stardist
    echo "stardist3r.${step}()"
}
function get_cmd_stardist_train { get_cmd ; }

function get_py_stardist_nblocks {
    get_py_header
    get_py_stardist
    echo "stardist3r.predict(print_nblocks=True)"
}
function get_cmd_stardist_nblocks { get_cmd ; }

function get_py_stardist_predict {
    get_py_header
    get_py_stardist
    echo "stardist3r.${step}(blocks=[idx])"
}
function get_cmd_stardist_predict { get_cmd "\${idx}" ; }

function get_py_stardist_merge {
    get_py_header
    get_py_stardist
    echo "stardist3r.${step}()"
}
function get_cmd_stardist_merge { get_cmd ; }




function get_cmd_unet3d_memb_train {

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_memb__train__unetdir?;\
         s?MODELNAME?$unet3d_memb__train__modelname?;\
         s?IDS_IMAGE?$unet3d_memb__train__ids_image?;\
         s?IDS_LABEL?$unet3d_memb__train__ids_label?g" \
         "${STAPL3D}/pipelines/unet3d_memb_train.yml" \
         > "${datadir}/${dataset}_unet3d_memb_train_$unet3d_memb__train__modelname.yml"

    echo train3dunet --config "${datadir}/${dataset}_unet3d_memb_train_$unet3d_memb__train__modelname.yml"

}


function get_cmd_unet3d_memb_predict {
    # FIXME: kidney model gives error on load via predict3dunet => needs DataParallel wrapping

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_memb__predict__unetdir?;\
         s?MODELNAME?$unet3d_memb__predict__modelname?;\
         s?IDS_IMAGE?$unet3d_memb__predict__ids_image?;\
         s?PATH_TO_THE_TEST_SET?${blockdir}?g" \
         "${STAPL3D}/pipelines/unet3d_memb_predict.yml" \
         > "${datadir}/${dataset}_unet3d_memb_predict_$unet3d_memb__predict__modelname.yml"

    echo predict3dunet --config "${datadir}/${dataset}_unet3d_memb_predict_$unet3d_memb__predict__modelname.yml"

}


function get_cmd_unet3d_nucl_train {

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_nucl__train__unetdir?;\
         s?MODELNAME?$unet3d_nucl__train__modelname?;\
         s?IDS_IMAGE?$unet3d_nucl__train__ids_image?;\
         s?IDS_LABEL?$unet3d_nucl__train__ids_label?g" \
         "${STAPL3D}/pipelines/unet3d_nucl_train.yml" \
         > "${datadir}/${dataset}_unet3d_nucl_train_$unet3d_nucl__train__modelname.yml"

    echo train3dunet --config "${datadir}/${dataset}_unet3d_nucl_train_$unet3d_nucl__train__modelname.yml"

}


function get_cmd_unet3d_nucl_predict {
    # FIXME: kidney model gives error on load via predict3dunet => needs DataParallel wrapping

    # copy config file with replacements
    sed "s?UNETDIR?$unet3d_nucl__predict__unetdir?;\
         s?MODELNAME?$unet3d_nucl__predict__modelname?;\
         s?IDS_IMAGE?$unet3d_nucl__predict__ids_image?;\
         s?PATH_TO_THE_TEST_SET?${blockdir}?g" \
         "${STAPL3D}/pipelines/unet3d_nucl_predict.yml" \
         > "${datadir}/${dataset}_unet3d_nucl_predict_$unet3d_nucl__predict__modelname.yml"

    echo predict3dunet --config "${datadir}/${dataset}_unet3d_nucl_predict_$unet3d_nucl__predict__modelname.yml"

}


function get_py_unet3d_nucl_predict { get_py_plantseg_predict ; }
function get_cmd_unet3d_nucl_predict {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"
    # TODO: make similar system as plantseg_predict: this will be messy => prefered: spec ids in plantseg yaml
    # echo python "${pyfile}" "\${blockstem}.h5" "\${blockstem}.h5" 'raw' 'nucl/mean' 'True' 'True' 'False'  # NOTE set in yaml directly

    echo prep_plantseg_predict "${plantseg_predict__params__ps_blockdir}"

    echo ""
    echo ps_path="\${datadir}/${plantseg_predict__params__ps_blockdir}/\${dataset}_\${block_id}"

    echo sed "\"s?UNETDIR?$unet3d_nucl_predict__params__unetdir?;s?MODELNAME?$unet3d_nucl_predict__params__modelname?;s?IDS_NUCLEUS?$unet3d_nucl_predict__params__ids_nucleus?;s?PATH_TO_THE_TEST_SET?\${ps_path}?g\"" "\${STAPL3D}/pipelines/unet3d_nucl_predict.yml" \> "\${ps_path}/\${dataset}_\${block_id}_unet3d_nucl_predict.yml"
    echo predict3dunet --config "\${ps_path}/\${dataset}_\${block_id}_unet3d_nucl_predict.yml"

    echo ""
    # TODO: make similar path as plantseg_predict: this will be messy
    # echo python "${pyfile}" "\${ps_path}/${plantseg_predict__params__modelname}/\${dataset}_\${block_id}_predictions.h5" "\${blockstem}.h5" 'predictions' 'nucl/3dunet' 'True' 'False' 'True'
    echo python "${pyfile}" "\${ps_path}/\${dataset}_\${block_id}_predictions.h5" "\${blockstem}.h5" 'predictions' 'nucl/3dunet' 'True' 'False' 'True'

}
# mkdir $datadir/blocks_unet3d_nucl_stapl3d
# for block_id in "${block_ids[@]}"; do
#     cp $datadir/blocks_ps/${dataset}_${block_id}/${dataset}_${block_id}_predictions.h5 $datadir/blocks_unet3d_nucl_stapl3d/
# done
# TODO: test... output_dir: '/home/adrian/Datasets/DSB2018/test/predictions'

# function get_py_plantseg_predict {
#     ### stapl3d-blocks to plantseg-input
#
#     echo '#!/usr/bin/env python'
#     echo ''
#     echo 'import sys'
#     echo 'blockdir_in = sys.argv[1]'
#     echo 'blockdir_out = sys.argv[2]'
#     echo 'dset_in = sys.argv[3]'
#     echo 'dset_out = sys.argv[4]'
#     echo ''
#     echo 'import os'
#     echo 'import h5py'
#     echo 'import glob'
#     echo "blockfiles_in = glob.glob(os.path.join(blockdir_in, '*.h5'))"
#     echo "blockfiles_in.sort()"
#     echo "blockfiles_out = glob.glob(os.path.join(blockdir_out, '*.h5'))"
#     echo "blockfiles_out.sort()"
#     echo "for blockfile_in, blockfile_out in zip(blockfiles_in, blockfiles_out):"
#     echo "    blockstem_in = os.path.splitext(os.path.basename(blockfile_in))[0]"
#     echo "    blockstem_out = os.path.splitext(os.path.basename(blockfile_out))[0]"
#     echo "    filepath_in = os.path.join(blockdir_in, '{}.h5'.format(blockstem_in))"
#     echo "    filepath_out = os.path.join(blockdir_out, '{}.h5'.format(blockstem_out))"
#     echo "    f = h5py.File(filepath_out, 'r+')"
#     echo "    if blockdir_in == blockdir_out:"
#     echo "        f[dset_out] = f[dset_in]"
#     echo "    else:"
#     echo "        f[dset_out] = h5py.ExternalLink(filepath_in, dset_in)"
#     echo "    f.close()"
#
# }
function get_py_plantseg_predict {
    ### stapl3d-blocks to plantseg-input

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'filepath_in = sys.argv[1]'
    echo 'filepath_out = sys.argv[2]'
    echo 'dset_in = sys.argv[3]'
    echo 'dset_out = sys.argv[4]'
    echo 'delete = sys.argv[5] == "True"'
    echo 'links = sys.argv[6] == "True"'
    echo 'is_unet = sys.argv[7] == "True"'
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
        # rm "${datadir}/${blocks_ps}/${filename%.h5}/${filename}"
        ln -s "${blockdir}/${filename}" "${datadir}/${blocks_ps}/${filename%.h5}/${filename}"
    done

}
function get_cmd_plantseg_predict {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"
    echo python "${pyfile}" "\${blockstem}.h5" "\${blockstem}.h5" 'memb/mean' 'raw' 'True' 'True' 'False'
    echo prep_plantseg_predict "${plantseg_predict__params__ps_blockdir}"
    echo ""
    echo ps_path="\${datadir}/${plantseg_predict__params__ps_blockdir}/\${dataset}_\${block_id}"
    echo sed "\"s?BLOCKS_PLANTSEG?\${ps_path}?;s?MODELNAME?$plantseg_predict__params__modelname?;s?DIM_Z?106?g\"" "\${STAPL3D}/pipelines/plantseg_config.yml" \> "\${ps_path}/\${dataset}_\${block_id}_plantseg.yml"
    echo plantseg --config "\${ps_path}/\${dataset}_\${block_id}_plantseg.yml"
    echo ""
    echo python "${pyfile}" "\${ps_path}/${plantseg_predict__params__modelname}/\${dataset}_\${block_id}_predictions.h5" "\${blockstem}.h5" 'predictions' 'memb/3dunet' 'True' 'False' 'True'

}





function get_py_splitblocks_ki67 {

    echo '#!/usr/bin/env python'
    echo ''
    echo 'import sys'
    echo 'image_in = sys.argv[1]'
    echo 'parameter_file = sys.argv[2]'
    echo 'idx = int(sys.argv[3])'
    echo 'outputstem = sys.argv[4]'
    echo ''
    echo 'from stapl3d import Image, wmeMPI'
    echo "im = Image(image_in, permission='r')"
    echo "im.load(load_data=False)"
    echo "mpi = wmeMPI(usempi=False)"
    echo "mpi.set_blocks(im, [$Z, $bs, $bs, $C, 1], [0, $bm, $bm, 0, 0], [])"
    echo "mpi.scatter_series()"
    echo "props = im.squeeze_props(im.get_props(), dim=4)"
    echo "block = mpi.blocks[idx]"
    echo "im.slices = block['slices']"
    echo "data = im.slice_dataset()"
    echo "props['shape'] = data.shape"
    echo "props['slices'] = None"
    echo "h5path = '{}_{}_ki67.h5/data'.format(outputstem, block['id'])"
    echo "mo = Image(h5path, **props)"
    echo "mo.create()"
    echo "mo.write(data)"
    echo "mo.close()"

}
function get_cmd_splitblocks_ki67 {

    pyfile="${datadir}/${jobname}.py"
    eval get_py_${stage} > "$pyfile"

    echo python "${pyfile}" \
        "\${biasfield_stem}.ims" \
        "${parfile}" \
        "\${idx}" \
        "\${blockdir}/${dataset_preproc}"

}

# function get_py_apply_ilastik { get_py_splitblocks_ki67 ; }
function get_cmd_apply_ilastik {

    # NOTE: NO, first generate all spliblocks because datastems and python blocks are soreted differently
    # pyfile="${datadir}/${jobname}.py"
    # eval get_py_${stage} > "$pyfile"
    #
    # echo python "${pyfile}" \
    #     "\${biasfield_stem}.ims" \
    #     "${parfile}" \
    #     "\${idx}" \
    #     "\${blockdir}/${dataset_preproc}"

    echo "export LAZYFLOW_THREADS=${apply_ilastik__submit__tasks};"
    echo "export LAZYFLOW_TOTAL_RAM_MB=100000;"
    echo "${ILASTIK} --headless --readonly 1 \\"
    echo "--preconvert_stacks \\"
    echo "--project=${datadir}/${apply_ilastik__params__pixprob_trainingset}.ilp \\"
    echo "--output_axis_order=zyxc \\"
    echo "--output_format='compressed hdf5' \\"
    echo "--output_filename_format=\${blockstem}${apply_ilastik__params__opf}.h5 \\"
    echo "--output_internal_path=${apply_ilastik__params__ods} \\"
    echo "\${blockstem}_ki67.h5/data"

}


function get_py_mergeblocks_ilastik { get_py_mergeblocks ; }
function get_cmd_mergeblocks_ilastik { get_cmd_mergeblocks ; }
# TODO: integrate step_id in arguments for all???
