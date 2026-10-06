#!/usr/bin/env bash
# Causal diagnostic of 9504111, GPU session runner (1x H100). Usage on the pod:
#   bash pod_causal_diag.sh <session1> <cap_usd>
# Reads the real hourly rate from the RunPod API, starts a guard that stops the pod at the hard cap (or 20 min after
# DIAG_FINISHED unless /workspace/RETRIEVED appears), installs the pinned stack, extracts and verifies the 761 images,
# and runs the session with an in-script budget guard that saves partial results.
set -u
stage=${1:?}; cap=${2:?}
cd /workspace
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 PIP_BREAK_SYSTEM_PACKAGES=1
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
RATE=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'content-type: application/json' \
  -d "{\"query\":\"query { pod(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { costPerHr } }\"}" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['pod']['costPerHr'])" 2>/dev/null)
[ -z "$RATE" ] && RATE=4.0
START=$(stat -c %Y /proc/1)
echo "RATE=$RATE START=$START CAP=$cap" | tee /workspace/rate.txt
CAPTS=$(python3 -c "print(int($START + ($cap - 0.10)/$RATE*3600))")
( fin=""
  while true; do
    now=$(date +%s)
    [ "$now" -ge "$CAPTS" ] && { echo "HARD_CAP $(date -u +%FT%TZ)"; break; }
    [ -e /workspace/RETRIEVED ] && { echo "RETRIEVED $(date -u +%FT%TZ)"; break; }
    if [ -e /workspace/DIAG_FINISHED ]; then [ -z "$fin" ] && fin=$now; [ $((now - fin)) -ge 1200 ] && { echo FINISHED_TIMEOUT; break; }; fi
    sleep 20
  done
  runpodctl stop pod "$RUNPOD_POD_ID" ) > /workspace/guard.log 2>&1 &
set -x
python -m pip install --no-input vllm==0.11.0 transformers==4.57.1 numpy==2.1.2 Pillow==12.3.0 tokenizers==0.22.2 huggingface-hub==0.36.2 torch==2.8.0 hf_transfer 2>&1 | tail -1
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download;print('MODEL',snapshot_download('Qwen/Qwen3-VL-8B-Instruct',revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))"
git clone -q https://github.com/sunrainyg/RandOpt.git /workspace/RandOpt; git -C /workspace/RandOpt checkout -q 4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca; git -C /workspace/RandOpt rev-parse HEAD
mkdir -p /workspace/repro; cd /workspace/repro; tar xzf /workspace/diag.tgz --no-same-owner; python /workspace/extract-q.py
pip freeze > /workspace/repro/pip-freeze.txt; nvidia-smi -L > /workspace/repro/gpu.txt
echo SETUP_DONE
cd /workspace/repro && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python -u scripts/causal_diag_9504111.py "$stage" --images /workspace/pt-images \
  --upstream /workspace/RandOpt --out /workspace/repro/results/paper-analysis/causal-diagnostic/$stage --usd-per-hour "$RATE" \
  --pod-start-epoch "$START" --cap-usd "$cap" > /workspace/run.log 2>&1
echo "exit=$? $(date -u +%FT%TZ)" > /workspace/DIAG_FINISHED
