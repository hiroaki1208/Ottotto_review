#! /bin/bash

BASE_DIR=$(dirname ${0})
cd ${BASE_DIR}

# defaultパラメータ設定
PARAM='param0'
SUBFOLDER=$(date '+%Y%m%d%H%M')

# パラメータのパース
while getopts p: OPT; do
    case ${OPT} in
        p) PARAM=${OPTARG} ;;
        *) usage ;;
    esac
done

source ${BASE_DIR}/configs/config.sh
set -euC
# conda activate env2

python ${BASE_DIR}/run.py -s ${SUBFOLDER} -p ${PARAM}