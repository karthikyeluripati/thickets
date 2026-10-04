#!/usr/bin/env bash
# Resumable OmniSpatial study. Complete phases are skipped; a failed phase is kept
# as .failed-N and rerun (operational, not scientific).
# ROLE=main (default): baseline, SHARDS, wait for all shards, search lock,
#   validation, validation lock, test, analysis.
# ROLE=helper: only SHARDS (search) and VSHARDS (validation) on this GPU; all GPUs of a
#   pod share one results root, so the main process sees every shard directly.
set -euo pipefail
root=${1:?study root required}; upstream=${2:?pinned RandOpt checkout required}; images=${3:?verified image root required}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/locks"
shards=${SHARDS:-0 1 2 3 4 5 6 7}; vshards=${VSHARDS:-0 1 2 3}; role=${ROLE:-main}
complete() { python -c "import json,sys;sys.exit(json.load(open('$1/run-manifest.json'))['status']!='complete')" 2>/dev/null; }
run_phase() {
  local name=$1; shift
  if [ -d "$root/$name" ] && complete "$root/$name"; then echo "skip $name (complete)"; return; fi
  if [ -d "$root/$name" ]; then local n=1; while [ -e "$root/$name.failed-$n" ]; do n=$((n+1)); done; mv "$root/$name" "$root/$name.failed-$n"; mv "$root/$name.log" "$root/$name.failed-$n.log" 2>/dev/null || true; fi
  echo "start $name $(date -u +%FT%TZ)"
  bash scripts/run_omnispatial_phase.sh "$@" --locks "$root/locks" --images "$images" --upstream-root "$upstream" --out "$root/$name" > "$root/$name.log" 2>&1
  echo "done $name $(date -u +%FT%TZ)"
}
lock() {
  if git cat-file -e "HEAD:$root/locks/$1.json" 2>/dev/null; then echo "skip lock $1 (committed)"; return; fi
  if [ -e "$root/locks/$1.json" ]; then mv "$root/locks/$1.json" "$root/locks/$1.uncommitted-$(date +%s).json"; fi
  python scripts/freeze_omnispatial_lock.py "$1" --root "$root"
  git add "$root/locks/$1.json"; git commit --quiet -m "$2"; git log -1 --format='%H %cI %s'
}
if [ "$role" = main ]; then
  run_phase baseline --phase baseline
  lock baseline 'Freeze OmniSpatial 7B base outputs and preprocessing before candidate search'
fi
until git cat-file -e "HEAD:$root/locks/baseline.json" 2>/dev/null; do sleep 60; done
for k in $shards; do run_phase "search-shard-0$k" --phase search --shard "$k"; done
if [ "$role" = main ]; then
  for k in 0 1 2 3 4 5 6 7; do until complete "$root/search-shard-0$k"; do sleep 60; done; done
  echo "all eight search shards complete $(date -u +%FT%TZ)"
  lock search 'Freeze OmniSpatial search ranking and top 30 before validation inference'
fi
until git cat-file -e "HEAD:$root/locks/search.json" 2>/dev/null; do sleep 30; done
for j in $vshards; do run_phase "validation-shard-0$j" --phase validation --shard "$j"; done
if [ "$role" != main ]; then echo "helper shards complete: search $shards validation $vshards"; exit 0; fi
for j in 0 1 2 3; do until complete "$root/validation-shard-0$j"; do sleep 30; done; done
lock validation 'Freeze OmniSpatial validation rank 1 and top 5 before final-test inference'
run_phase test --phase test
[ -d "$root/analysis" ] || python scripts/analyze_omnispatial.py --root "$root"
echo 'OmniSpatial existence test complete.'
