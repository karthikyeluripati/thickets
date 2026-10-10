#!/usr/bin/env bash
cp /workspace/rate.txt /workspace/gpu.txt /workspace/pip-freeze.txt /workspace/model_sha.txt /workspace/data_hashes.txt /workspace/rv/ 2>/dev/null
touch /workspace/DONE; echo ALL_DONE
