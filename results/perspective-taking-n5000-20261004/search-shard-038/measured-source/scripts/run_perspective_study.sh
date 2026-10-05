#!/usr/bin/env bash
# Resumable Perspective-Taking N=5000 study, one TP=1 process per GPU.
# Complete phases are skipped; a failed phase is kept as .failed-N (with its log)
# and rerun from scratch, so only missing candidates are recomputed and no retry
# is double-counted. ROLE=main also runs baseline, locks, test and analysis.
set -euo pipefail
root=${1:?}; upstream=${2:?}; images=${3:?}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/locks"
role=${ROLE:-main}; shards=${SHARDS:?}; vshards=${VSHARDS:-}
complete() { python -c "import json,sys;sys.exit(json.load(open('$1/run-manifest.json'))['status']!='complete')" 2>/dev/null; }
run_phase() {
  local name=$1; shift
  if [ -d "$root/$name" ] && complete "$root/$name"; then echo "skip $name (complete)"; return; fi
  if [ -d "$root/$name" ]; then local n=1; while [ -e "$root/$name.failed-$n" ]; do n=$((n+1)); done; mv "$root/$name" "$root/$name.failed-$n"; mv "$root/$name.log" "$root/$name.failed-$n.log" 2>/dev/null || true; fi
  echo "start $name $(date -u +%FT%TZ)"
  bash scripts/run_perspective_phase.sh "$@" --locks "$root/locks" --images "$images" --upstream-root "$upstream" --out "$root/$name" > "$root/$name.log" 2>&1
  echo "done $name $(date -u +%FT%TZ)"
}
lock() {
  if git cat-file -e "HEAD:$root/locks/$1.json" 2>/dev/null; then echo "skip lock $1 (committed)"; return; fi
  if [ -e "$root/locks/$1.json" ]; then mv "$root/locks/$1.json" "$root/locks/$1.uncommitted-$(date +%s).json"; fi
  python scripts/freeze_perspective_lock.py "$1" --root "$root"
  git add "$root/locks/$1.json"; git commit --quiet -m "$2"; git log -1 --format='%H %cI %s'
}
nshards=$(python -c "import json;p=json.load(open('experiments/perspective_taking_n5000_protocol.json'))['candidates'];print(p['count']//p['shard_size'])")
if [ "$role" = main ]; then
  run_phase baseline --phase baseline
  lock baseline 'Freeze Qwen3-VL-8B Perspective-Taking base outputs before candidate search'
fi
until git cat-file -e "HEAD:$root/locks/baseline.json" 2>/dev/null; do sleep 30; done
for k in $shards; do n=$(printf %03d $k); run_phase "search-shard-$n" --phase search --shard "$k"; done
if [ "$role" = main ]; then
  for k in $(seq 0 $((nshards - 1))); do n=$(printf %03d $k); until complete "$root/search-shard-$n"; do sleep 60; done; done
  echo "all $nshards search shards complete $(date -u +%FT%TZ)"
  lock search 'Freeze 5000-candidate search ranking and top 50 before validation inference'
fi
until git cat-file -e "HEAD:$root/locks/search.json" 2>/dev/null; do sleep 30; done
nv=$(python -c "
import json,sys;sys.path.insert(0,'src')
from thicket_runtime.perspective_runtime import validation_plan, VALIDATION_SHARD_SIZE
from thicket_runtime.visual_runtime import read
p=read('experiments/perspective_taking_n5000_protocol.json');l={'search':read('$root/locks/search.json')}
print(-(-len(validation_plan(l,p))//VALIDATION_SHARD_SIZE))")
for j in $vshards; do [ "$j" -lt "$nv" ] || continue; n=$(printf %02d $j); run_phase "validation-shard-$n" --phase validation --shard "$j"; done
if [ "$role" != main ]; then echo "helper complete: search [$shards] validation [$vshards]"; exit 0; fi
for j in $(seq 0 $((nv - 1))); do n=$(printf %02d $j); until complete "$root/validation-shard-$n"; do sleep 30; done; done
lock validation 'Freeze validation rank 1, top 5 and top 10 before final-test inference'
run_phase test --phase test
[ -d "$root/analysis" ] || python scripts/analyze_perspective.py --root "$root"
echo 'Perspective-Taking N=5000 existence test complete.'
