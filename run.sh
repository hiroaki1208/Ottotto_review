#! /bin/bash

BASE_DIR=$(dirname ${0})
cd ${BASE_DIR}

# defaultパラメータ設定
TARGET='validation'
IS_PARTIAL=0
# パラメータのパース
while getopts t:p: OPT; do
    case ${OPT} in
        t) TARGET=${OPTARG} ;;
        p) IS_PARTIAL=1 ;;
        *) usage ;;
    esac
done

source ${BASE_DIR}/configs/config.sh
set -euC


python ${BASE_DIR}/run.py -t ${TARGET} -p ${IS_PARTIAL}