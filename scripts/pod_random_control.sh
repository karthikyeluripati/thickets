#!/usr/bin/env bash
# Random-control transfer for 9504111: pod runner. Setup runs immediately; INFERENCE starts only after
# /workspace/GO exists (created only after explicit user approval). Guard: hard cap (cap - 0.10 USD), auto-stop if
# no GO within 40 min of SETUP_DONE, and stop 20 min after RC_FINISHED unless /workspace/RETRIEVED appears first.
set -u
cap=${1:?}
cd /workspace
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 PIP_BREAK_SYSTEM_PACKAGES=1
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
RATE=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'content-type: application/json' \
  -d "{\"query\":\"query { pod(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { costPerHr } }\"}" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['pod']['costPerHr'])" 2>/dev/null)
[ -z "$RATE" ] && RATE=4.0
START=$(stat -c %Y /proc/1)
echo "RATE=$RATE START=$START CAP=$cap" | tee /workspace/rate.txt
CAPTS=$(python3 -c "print(int($START + ($cap - 0.10)/$RATE*3600))")
( fin=""; sd=""
  while true; do
    now=$(date +%s)
    [ "$now" -ge "$CAPTS" ] && { echo "HARD_CAP $(date -u +%FT%TZ)"; break; }
    [ -e /workspace/RETRIEVED ] && { echo "RETRIEVED $(date -u +%FT%TZ)"; break; }
    if [ -e /workspace/SETUP_DONE ] && [ ! -e /workspace/GO ]; then [ -z "$sd" ] && sd=$now; [ $((now - sd)) -ge 2400 ] && { echo "NO_GO_TIMEOUT"; break; }; fi
    if [ -e /workspace/RC_FINISHED ]; then [ -z "$fin" ] && fin=$now; [ $((now - fin)) -ge 1200 ] && { echo FINISHED_TIMEOUT; break; }; fi
    sleep 20
  done
  runpodctl stop pod "$RUNPOD_POD_ID" ) > /workspace/guard.log 2>&1 &
set -x
python -m pip install --no-input vllm==0.11.0 transformers==4.57.1 numpy==2.1.2 Pillow==12.3.0 tokenizers==0.22.2 huggingface-hub==0.36.2 torch==2.8.0 hf_transfer 2>&1 | tail -1
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download;print('MODEL',snapshot_download('Qwen/Qwen3-VL-8B-Instruct',revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))"
git clone -q https://github.com/sunrainyg/RandOpt.git /workspace/RandOpt; git -C /workspace/RandOpt checkout -q 4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca; git -C /workspace/RandOpt rev-parse HEAD
mkdir -p /workspace/repro; cd /workspace/repro; tar xzf /workspace/rcimg.tgz --no-same-owner; python /workspace/extract-q.py
pip freeze > /workspace/pip-freeze.txt; nvidia-smi -L > /workspace/gpu.txt
touch /workspace/SETUP_DONE; echo SETUP_DONE
set +x
while [ ! -e /workspace/GO ]; do sleep 10; done
cd /workspace/repro && tar xzf /workspace/rc.tgz --no-same-owner && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python -u scripts/random_control_transfer.py \
  --images /workspace/pt-images --upstream /workspace/RandOpt --out /workspace/repro/results/paper-analysis/random-control-transfer/gpu \
  --usd-per-hour "$RATE" --pod-start-epoch "$START" --cap-usd "$cap" > /workspace/run.log 2>&1
echo "exit=$? $(date -u +%FT%TZ)" > /workspace/RC_FINISHED
