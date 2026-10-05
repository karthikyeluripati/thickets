#!/usr/bin/env bash
# One TP=1 orchestrator per GPU. GPU g runs search shards k with k % N == g and
# validation shards j with j % N == g; GPU 0 is main. Afterwards stop the pod once
# results are retrieved (RETRIEVED marker) or after a limit. The RunPod key is read
# from the container environment because detached shells do not inherit it.
cd "$(dirname "$0")/.."
root=${1:?}; upstream=${2:?}; images=${3:?}; wait_minutes=${4:-60}
n=$(nvidia-smi -L | wc -l)
nshards=$(python -c "import json;p=json.load(open('experiments/perspective_taking_n5000_protocol.json'))['candidates'];print(p['count']//p['shard_size'])")
pids=()
for g in $(seq 0 $((n - 1))); do
  sh=$(seq $g $n $((nshards - 1)) | tr '\n' ' '); vs=$(seq $g $n 15 | tr '\n' ' ')
  role=helper; [ $g = 0 ] && role=main
  CUDA_VISIBLE_DEVICES=$g RAY_TMPDIR=/tmp/ray-g$g ROLE=$role SHARDS="$sh" VSHARDS="$vs" \
    bash scripts/run_perspective_study.sh "$root" "$upstream" "$images" > "$root.gpu$g.log" 2>&1 &
  pids+=($!); echo "gpu $g role=$role search:[$sh] validation:[$vs] pid=$!"
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "exit=$status $(date -u +%FT%TZ)" >> /workspace/STUDY_FINISHED
for _ in $(seq "$wait_minutes"); do [ -e /workspace/RETRIEVED ] && break; sleep 60; done
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
runpodctl stop pod "$RUNPOD_POD_ID"
