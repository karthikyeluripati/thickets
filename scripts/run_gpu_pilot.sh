#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/gpu-pilot}"
if [[ -e "$OUT" ]]; then
  echo "Refusing existing output path: $OUT" >&2
  exit 2
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: use a CUDA-enabled PyTorch environment.")
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
python -m pytest -q | tee "$OUT/tests.txt"
# ~33M parameters, not a language model. Bounded instrumentation smoke first.
python -m thicket_runtime.cli --device cuda:0 --dtype float32 \
  --width 2048 --depth 8 --batch 8 --candidates 8 --repeats 3 --warmup 2 \
  --out "$OUT/coarse"
python -m thicket_runtime.cli --device cuda:0 --dtype float32 \
  --width 2048 --depth 8 --batch 8 --candidates 8 --repeats 3 --warmup 2 \
  --no-cuda-events --out "$OUT/wall-only-control"
python -m thicket_runtime.cli --device cuda:0 --dtype float32 \
  --width 2048 --depth 8 --batch 8 --candidates 2 --repeats 1 --warmup 0 \
  --trace --out "$OUT/diagnostic"
echo "Synthetic smoke complete. These are not LLM/vLLM throughput results."
