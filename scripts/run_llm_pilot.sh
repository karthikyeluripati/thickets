#!/usr/bin/env bash
# Bounded execution-cost microbenchmarks; the arithmetic prompts are not a
# representative task-quality evaluation. Keep the CUDA PyTorch installation.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/llm-pilot}"
if [[ -e "$OUT" ]]; then
  echo "Refusing existing output path: $OUT" >&2
  exit 2
fi
python - <<'PY'
import torch, transformers
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; use a CUDA-enabled PyTorch environment.")
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__, "GPU:", torch.cuda.get_device_name(0))
PY
mkdir -p "$OUT"
nvidia-smi -q > "$OUT/nvidia-smi-before.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
python -m pytest -q | tee "$OUT/tests.txt"
# These flags avoid requiring optional download accelerators. No model token is
# needed for this public, immutable model revision.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
python - "$OUT" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
data = Path("examples/arithmetic_smoke.jsonl")
small = root / "arithmetic-b1.jsonl"
small.write_text(data.read_text().splitlines()[0] + "\n")
base = [sys.executable, "-u", "-m", "thicket_runtime.cli",
        "--workload", "hf", "--device", "cuda:0", "--dtype", "bfloat16",
        "--model", "Qwen/Qwen2.5-0.5B",
        "--revision", "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "--fixed-length", "--candidates", "4", "--repeats", "3", "--warmup", "2"]
jobs = [("fixed-32-b4", 32, data, []),
        ("fixed-128-b4", 128, data, []),
        ("fixed-32-b1", 32, small, []),
        ("fixed-128-b1", 128, small, []),
        ("fixed-32-b4-wall-only", 32, data, ["--no-cuda-events"])]
records = []
for name, tokens, inputs, extra in jobs:
    cmd = base + ["--data", str(inputs), "--max-new-tokens", str(tokens),
                  "--out", str(root / name)] + extra
    record = {"name": name, "argv": cmd, "timeout_seconds": 900}
    records.append(record)
    commands = root / "commands.json"
    commands.write_text(json.dumps(records, indent=2) + "\n")
    print("START", name, flush=True)
    with (root / (name + ".log")).open("w") as log:
        try:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=record["timeout_seconds"])
            record["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
    commands.write_text(json.dumps(records, indent=2) + "\n")
    if record.get("returncode") != 0:
        raise SystemExit(f"{name} failed or timed out; inspect {name}.log")
    print((root / name / "summary.json").read_text(), flush=True)
PY
nvidia-smi -q > "$OUT/nvidia-smi-after.txt"
python scripts/summarize_llm_pilot.py "$OUT" > "$OUT/analysis.json"
echo "Bounded LLM pilot complete. Fixed-token HF eager baseline only; inspect exactness gates."
