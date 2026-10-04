#!/usr/bin/env bash
# Resumable: complete phases are skipped; a failed phase directory is kept
# under a .failed-N name and rerun from scratch (operational, not scientific).
set -euo pipefail
root=${1:?study root required}
upstream=${2:?pinned RandOpt checkout required}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/locks"
complete() { python -c "import json,sys;sys.exit(json.load(open('$1/run-manifest.json'))['status']!='complete')" 2>/dev/null; }
run_phase() {  # name, args...
  local name=$1; shift
  if [ -d "$root/$name" ] && complete "$root/$name"; then echo "skip $name (complete)"; return; fi
  if [ -d "$root/$name" ]; then local n=1; while [ -e "$root/$name.failed-$n" ]; do n=$((n+1)); done; mv "$root/$name" "$root/$name.failed-$n"; mv "$root/$name.log" "$root/$name.failed-$n.log" 2>/dev/null || true; fi
  echo "start $name $(date -u +%FT%TZ)"
  bash scripts/run_transfer_aware_phase.sh "$@" --locks "$root/locks" --upstream-root "$upstream" --out "$root/$name" > "$root/$name.log" 2>&1
  echo "done $name $(date -u +%FT%TZ)"
}
lock() {  # name, commit message
  if git cat-file -e "HEAD:$root/locks/$1.json" 2>/dev/null; then echo "skip lock $1 (committed)"; return; fi
  if [ -e "$root/locks/$1.json" ]; then mv "$root/locks/$1.json" "$root/locks/$1.uncommitted-$(date +%s).json"; fi
  python scripts/freeze_transfer_aware_lock.py "$1" --root "$root"
  git add "$root/locks/$1.json"
  git commit --quiet -m "$2"
  git log -1 --format='%H %cI %s'
}
run_phase baseline --phase baseline
lock baseline 'Freeze 7B base outputs and fold baselines before candidate search'
for k in 0 1 2 3 4 5 6 7; do run_phase "search-shard-0$k" --phase search --shard "$k"; done
lock ranking 'Freeze VANILLA and TRANSFER_MIN top-20 before validation inference'
run_phase validation --phase validation
lock validation 'Freeze first-bridge validation gate and final candidates before test inference'
if python -c "import json,sys;sys.exit(not json.load(open('$root/locks/validation.json'))['gate']['pass'])"; then
  run_phase test --phase test
else
  echo 'Validation gate failed: NO_GO_TRANSFER_AWARE_VISUAL_SEARCH; final test not run.'
fi
[ -d "$root/analysis" ] || python scripts/analyze_transfer_aware.py --root "$root"
echo 'Transfer-aware first bridge complete.'
