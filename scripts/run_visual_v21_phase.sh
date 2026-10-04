#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 PERTURB_VISUAL=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=60 28800 python -u -m thicket_runtime.visual_v21_runtime "$@"
