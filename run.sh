#! /bin/bash

BASE_DIR=$(dirname ${0})
cd ${BASE_DIR}

# defaultパラメータ設定
TARGET='validation'
# echo ${TARGET}
# パラメータのパース
while getopts t: OPT; do
    case ${OPT} in
        t) TARGET=${OPTARG} ;;
        *) usage ;;
    esac
done

source ${BASE_DIR}/configs/config.sh
set -euC

LOG "TARGET: ${TARGET}"
python ${BASE_DIR}/run.py -t ${TARGET}