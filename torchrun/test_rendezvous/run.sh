#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ID="${RDZV_ID:-allreduce-test}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR-"./logs"}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export QUICK_EXIT="${QUICK_EXIT:-0}"
export MUSA_VISIBLE_DEVICES='4,5'

torchrun_args=(
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --node_rank="${NODE_RANK}"
  --rdzv_backend="${RDZV_BACKEND}"
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  --rdzv_id="${RDZV_ID}"
)

if [[ -n "${TORCHRUN_LOG_DIR}" ]]; then
  mkdir -p "${TORCHRUN_LOG_DIR}"
  torchrun_args+=(
    --log_dir="${TORCHRUN_LOG_DIR}"
    --redirects=3
    --tee=3
  )
fi

printf 'PYTHONPATH: %s\n' "${PYTHONPATH}"
printf 'torchrun command:'
printf ' %q' torchrun "${torchrun_args[@]}" test_allreduce.py
printf '\n'

exec torchrun \
  "${torchrun_args[@]}" \
  test_allreduce.py
