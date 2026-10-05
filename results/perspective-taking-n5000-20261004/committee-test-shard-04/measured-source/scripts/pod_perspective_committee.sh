#!/usr/bin/env bash
# RandOpt top-50 re-evaluation: committee TEST shard j runs on GPU j % N (one TP=1 process
# per GPU). Then stop the pod once results are retrieved (RETRIEVED) or after a limit.
cd "$(dirname "$0")/.."
root=${1:?}; upstream=${2:?}; images=${3:?}; wait_minutes=${4:-60}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
n=$(nvidia-smi -L | wc -l)
shards=$(python -c "import json;c=json.load(open('$root/locks/committee.json'));print(-(-len(c['to_generate'])//c['shard_size']))")
pids=()
for g in $(seq 0 $((n - 1))); do
  ( for j in $(seq $g $n $((shards - 1))); do
      d=$root/committee-test-shard-$(printf %02d $j)
      if [ -d $d ] && python -c "import json,sys;sys.exit(json.load(open('$d/run-manifest.json'))['status']!='complete')" 2>/dev/null; then continue; fi
      if [ -d $d ]; then k=1; while [ -e $d.failed-$k ]; do k=$((k+1)); done; mv $d $d.failed-$k; mv $d.log $d.failed-$k.log 2>/dev/null; fi
      CUDA_VISIBLE_DEVICES=$g bash scripts/run_perspective_phase.sh --phase committee --shard $j --locks $root/locks \
        --images $images --upstream-root $upstream --out $d > $d.log 2>&1 || exit 1
    done ) &
  pids+=($!)
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "exit=$status $(date -u +%FT%TZ)" >> /workspace/STUDY_FINISHED
for _ in $(seq "$wait_minutes"); do [ -e /workspace/RETRIEVED ] && break; sleep 60; done
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
runpodctl stop pod "$RUNPOD_POD_ID"
