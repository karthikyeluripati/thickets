#!/usr/bin/env bash
set -euo pipefail
root=${1:?New study artifact root required}
upstream=${2:?Pinned upstream checkout required}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/locks"
bash scripts/run_visual_v21_phase.sh --phase baseline --upstream-root "$upstream" --out "$root/baseline-01" > "$root/baseline-01.log" 2>&1
python scripts/decide_visual_v21_stage.py --phase baseline --run "$root/baseline-01" --out "$root/locks/baseline.json"
git add "$root/locks/baseline.json"
git commit --quiet -m 'Freeze v2.1 capability decision before RandOpt'
if ! python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["accepted"] else 1)' "$root/locks/baseline.json"; then
  echo 'STOP_benchmark_capability_gate; no RandOpt candidates generated.'
  exit 0
fi
bash scripts/run_visual_v21_phase.sh --phase calibration --upstream-root "$upstream" --baseline-lock "$root/locks/baseline.json" --out "$root/calibration-01" > "$root/calibration-01.log" 2>&1
python scripts/decide_visual_v21_stage.py --phase calibration --run "$root/calibration-01" --baseline-lock "$root/locks/baseline.json" --out "$root/locks/sigma.json"
git add "$root/locks/sigma.json"
git commit --quiet -m 'Freeze v2.1 sigma decision before final search'
if ! python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["continue_search"] else 1)' "$root/locks/sigma.json"; then
  echo 'NO_GO_no_search_signal; no final-search or held-out candidates generated.'
  exit 0
fi
bash scripts/run_visual_v21_phase.sh --phase search --upstream-root "$upstream" --baseline-lock "$root/locks/baseline.json" --sigma-lock "$root/locks/sigma.json" --out "$root/search-01" > "$root/search-01.log" 2>&1
python scripts/decide_visual_v21_stage.py --phase search --run "$root/search-01" --baseline-lock "$root/locks/baseline.json" --out "$root/locks/selection.json"
git add "$root/locks/selection.json"
git commit --quiet -m 'Freeze v2.1 best and committees before candidate held-out inference'
bash scripts/run_visual_v21_phase.sh --phase heldout --upstream-root "$upstream" --baseline-lock "$root/locks/baseline.json" --selection-lock "$root/locks/selection.json" --out "$root/heldout-01" > "$root/heldout-01.log" 2>&1
echo 'V2.1 GPU phases complete; apply the frozen terminal analysis.'
