#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 || -e "$3" ]]; then
  echo "usage: $0 UPSTREAM_ROOT COMMITTED_SELECTION_DIRECTORY NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=60 6300 python -u -m thicket_runtime.committee_validation \
  --upstream-root "$1" --selection "$2" --out "$3"
