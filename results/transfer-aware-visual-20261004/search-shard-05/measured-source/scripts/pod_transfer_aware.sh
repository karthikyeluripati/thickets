#!/usr/bin/env bash
# Pod wrapper: run the study, record the outcome, then wait for the local
# retrieval acknowledgement (or a time limit) and stop the pod.
cd "$(dirname "$0")/.."
root=${1:?study root required}; upstream=${2:?upstream required}; wait_minutes=${3:-180}
bash scripts/run_transfer_aware_study.sh "$root" "$upstream" >> "$root.study.log" 2>&1
echo "exit=$? $(date -u +%FT%TZ)" > /workspace/STUDY_FINISHED
for _ in $(seq "$wait_minutes"); do [ -e /workspace/RETRIEVED ] && break; sleep 60; done
runpodctl stop pod "$RUNPOD_POD_ID"
