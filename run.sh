#! /bin/bash

BASE_DIR=$(dirname ${0})
cd ${BASE_DIR}

# defaultパラメータ設定
TARGET='validation'
# IS_PARTIAL=0
COVISIT='setting0'
SUBFOLDER=$(date '+%Y%m%d%H%M')

# パラメータのパース
while getopts t:c: OPT; do
    case ${OPT} in
        t) TARGET=${OPTARG} ;;
        # p) IS_PARTIAL=1 ;;
        c) COVISIT=${OPTARG} ;;
        *) usage ;;
    esac
done

source ${BASE_DIR}/configs/config.sh
set -euC
# conda activate env2

python ${BASE_DIR}/run.py -s ${SUBFOLDER} -t ${TARGET} -c ${COVISIT}