#!/usr/bin/env bash
set -euo pipefail
root=${1:?New final-study root required}
upstream=${2:?Pinned upstream checkout required}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
test -f "$root/locks/sigma.json"
bash scripts/run_visual_final_phase.sh --phase diagnostic --sigma-lock "$root/locks/sigma.json" --upstream-root "$upstream" --out "$root/diagnostic-01" > "$root/diagnostic-01.log" 2>&1
# Diagnostic scores do not enter any branch or selection operation.
bash scripts/run_visual_final_phase.sh --phase search --sigma-lock "$root/locks/sigma.json" --upstream-root "$upstream" --out "$root/search-01" > "$root/search-01.log" 2>&1
python scripts/freeze_visual_final_selection.py --run "$root/search-01" --out "$root/locks/selection.json"
git add "$root/locks/selection.json"
git commit --quiet -m 'Freeze final v1 rank-1 and top-10 before candidate held-out inference'
git log -1 --format='%H %cI %s'
bash scripts/run_visual_final_phase.sh --phase heldout --sigma-lock "$root/locks/sigma.json" --selection-lock "$root/locks/selection.json" --upstream-root "$upstream" --out "$root/heldout-01" > "$root/heldout-01.log" 2>&1
python scripts/analyze_visual_final.py --root "$root" --out "$root/analysis"
echo 'Final v1 GPU experiment and frozen paired analysis complete.'
