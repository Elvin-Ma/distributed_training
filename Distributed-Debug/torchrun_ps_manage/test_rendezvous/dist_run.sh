#!/usr/bin/env bash
set -euo pipefail

QUICK_EXIT="${QUICK_EXIT:-0}"
HOSTFILE="${HOSTFILE:-./hostfile}"
RUN_DIR="${RUN_DIR:-$(pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-run.sh}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ID="${RDZV_ID:-allreduce-test}"
CURRENT_TIME="$(date "+%Y-%m-%d_%H:%M:%S")"


if [[ ! -f "${HOSTFILE}" ]]; then
  echo "hostfile not found: ${HOSTFILE}"
  exit 1
fi

hostlist="$(grep -v '^#\|^$' "${HOSTFILE}" | awk '{print $1}' | xargs)"
read -ra ip_list <<< "${hostlist}"
NNODES="${#ip_list[@]}"

if [[ "${NNODES}" -eq 0 ]]; then
  echo "no hosts found in ${HOSTFILE}"
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-${ip_list[0]}}"
mkdir -p logs/$CURRENT_TIME

echo "number of nodes: ${NNODES}"
echo "master address: ${MASTER_ADDR}"
echo "run dir: ${RUN_DIR}"
echo "log dir: logs/$CURRENT_TIME"
echo "start allreduce distributed test on:"


COUNT=0
for host in "${ip_list[@]}"; do
  echo "${host}"
  ssh -f -n "${host}" \
    "bash -c 'cd ${RUN_DIR} && nohup env PYTHONUNBUFFERED=1 QUICK_EXIT=${QUICK_EXIT} NNODES=${NNODES} NODE_RANK=${COUNT} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_BACKEND=${RDZV_BACKEND} RDZV_ID=${RDZV_ID} TORCHRUN_LOG_DIR=logs/${CURRENT_TIME}/node_${COUNT}_${host} bash ${RUN_SCRIPT} > logs/${CURRENT_TIME}/${COUNT}.log 2>&1 &'"
  ((COUNT+=1))
done
