#!/usr/bin/env bash
# One TP=1 orchestrator per GPU on this pod. GPU 0 is main (baseline, locks,
# test, analysis); search shard k and validation shard j run on GPU k % N and j % N.
# Then stop the pod once results are retrieved (RETRIEVED marker) or after a limit.
# The RunPod key comes from the container environment (detached shells lack it).
cd "$(dirname "$0")/.."
root=${1:?}; upstream=${2:?}; images=${3:?}; wait_minutes=${4:-60}
n=$(nvidia-smi -L | wc -l)
pids=()
for g in $(seq 0 $((n - 1))); do
  sh=""; for k in 0 1 2 3 4 5 6 7; do [ $((k % n)) = $g ] && sh="$sh $k"; done
  vs=""; for j in 0 1 2 3; do [ $((j % n)) = $g ] && vs="$vs $j"; done
  role=helper; [ $g = 0 ] && role=main
  CUDA_VISIBLE_DEVICES=$g RAY_TMPDIR=/tmp/ray-g$g ROLE=$role SHARDS="$sh" VSHARDS="$vs" \
    bash scripts/run_omnispatial_study.sh "$root" "$upstream" "$images" > "$root.gpu$g.log" 2>&1 &
  pids+=($!)
  echo "gpu $g role=$role search:$sh validation:$vs pid=$!"
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "exit=$status $(date -u +%FT%TZ)" >> /workspace/STUDY_FINISHED
for _ in $(seq "$wait_minutes"); do [ -e /workspace/RETRIEVED ] && break; sleep 60; done
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
runpodctl stop pod "$RUNPOD_POD_ID"
