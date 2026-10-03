#!/usr/bin/env bash
set -euo pipefail
# Invoke from the repository root in the pinned Work 1 vLLM environment.
# Setup/model downloads precede this bounded inference job.
if [[ $# != 2 ]]; then
  echo "usage: $0 UPSTREAM_ROOT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
if [[ -e "$2" ]]; then
  echo "refusing to overwrite $2" >&2
  exit 2
fi
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=60 3600 python -u -m thicket_runtime.shared_speculative \
  --upstream-root "$1" --protocol experiments/shared_speculative_protocol.json --out "$2"
