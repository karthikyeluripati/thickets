#!/usr/bin/env bash
# Run the study, record the outcome, then stop the pod when the results have been
# retrieved (RETRIEVED marker) or after a time limit. The RunPod key is loaded from
# the container environment because detached SSH shells do not inherit it.
cd "$(dirname "$0")/.."
root=${1:?}; upstream=${2:?}; images=${3:?}; wait_minutes=${4:-60}
bash scripts/run_omnispatial_study.sh "$root" "$upstream" "$images" >> "$root.study.log" 2>&1
echo "exit=$? $(date -u +%FT%TZ)" >> /workspace/STUDY_FINISHED
for _ in $(seq "$wait_minutes"); do [ -e /workspace/RETRIEVED ] && break; sleep 60; done
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
runpodctl stop pod "$RUNPOD_POD_ID"
