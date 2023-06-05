#! /bin/bash

BASE_DIR=$(dirname ${0})
cd ${BASE_DIR}

# defaultパラメータ設定
# TARGET='validation'
# IS_PARTIAL=0
# COVISIT='setting0'
PARAM='param0'
SUBFOLDER=$(date '+%Y%m%d%H%M')

# パラメータのパース
# while getopts t:p:c: OPT; do
while getopts p: OPT; do
    case ${OPT} in
        # t) TARGET=${OPTARG} ;;
        # p) IS_PARTIAL=1 ;;
        p) PARAM=${OPTARG} ;;
        # c) COVISIT=${OPTARG} ;;
        *) usage ;;
    esac
done

source ${BASE_DIR}/configs/config.sh
set -euC
# conda activate env2

python ${BASE_DIR}/run.py -s ${SUBFOLDER} -p ${PARAM}