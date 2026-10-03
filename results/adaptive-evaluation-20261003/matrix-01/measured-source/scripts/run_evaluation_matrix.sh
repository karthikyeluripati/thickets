#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || -e "$2" ]]; then
  echo "usage: $0 UPSTREAM_ROOT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=60 6300 python -u -m thicket_runtime.evaluation_matrix \
  --upstream-root "$1" --protocol experiments/adaptive_evaluation_protocol.json --out "$2"
