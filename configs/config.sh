#!/bin/bash

export DATA_DIR="$(pwd)/data"
export OUTPUT_DIR="$(pwd)/output"
export LOG_DIR="$(pwd)/logs"
export CONFIG_DIR="$(pwd)/configs"
export SRC_DIR="$(pwd)/src"

# conda activate env0a
# export PYTHONPATH="$(pwd)/src:$(cd ../):${PYTHONPATH}"

# ログ取得
function LOG()
{
  FILENM="$(basename $0)"
  MSG="${1}"

  LOG_DATE="$(date '+%Y-%m-%d')"
  LOG_TIME="$(date '+%H:%M:%S')"
  LOGFILE="${LOG_DIR}/example.log"

  echo "${LOG_DATE}" "${LOG_TIME}" "${FILENM}" "${MSG}"
  printf "%-10s %-8s %-14s %-50s\n" \
  "${LOG_DATE}" "${LOG_TIME}" "${FILENM}" "${MSG}" >>${LOGFILE}
}
