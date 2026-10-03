#!/usr/bin/env bash
set -euo pipefail
cd /workspace/thickets-visual
export PATH=/workspace/venv-visual/bin:$PATH
export PYTHONPATH=/workspace/thickets-visual/src
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
root=results/visual-line-tracing-20261003
while kill -0 5843 2>/dev/null; do sleep 10; done
python scripts/decide_visual_stage.py --phase calibration --run "$root/calibration-01" --baseline-lock "$root/locks/baseline.json" --out "$root/locks/sigma.json"
git add "$root/locks/sigma.json"
git commit -m 'Freeze visual sigma decision after fixed calibration'
if ! python -c 'import json,sys; sys.exit(0 if json.load(open("results/visual-line-tracing-20261003/locks/sigma.json"))["scale_valid"] else 1)'; then
  echo 'STOP_operational_scale_failure: no search or held-out candidates will run.'
  exit 0
fi
bash scripts/run_visual_phase.sh --phase search --upstream-root /workspace/RandOpt-visual --baseline-lock "$root/locks/baseline.json" --sigma-lock "$root/locks/sigma.json" --out "$root/search-01" > "$root/search-01.log" 2>&1
python scripts/decide_visual_stage.py --phase search --run "$root/search-01" --out "$root/locks/selection.json"
git add "$root/locks/selection.json"
git commit -m 'Freeze visual best and top committees before held-out generation'
bash scripts/run_visual_phase.sh --phase heldout --upstream-root /workspace/RandOpt-visual --baseline-lock "$root/locks/baseline.json" --selection-lock "$root/locks/selection.json" --out "$root/heldout-01" > "$root/heldout-01.log" 2>&1
echo 'Frozen visual GPU study complete; terminal analysis remains.'
