#!/usr/bin/env bash
# Multi-job pod session: guard (hard cap; idle stop 30 min after the last activity when no job is queued/running;
# stop on RETRIEVED), setup, then runs /workspace/jobs/NN-name.sh in order as they appear; each writes jobs/NN-name.rc.
set -u
cap=${1:?}; model=${2:?}
cd /workspace; mkdir -p jobs
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 PIP_BREAK_SYSTEM_PACKAGES=1
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
RATE=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'content-type: application/json' \
  -d "{\"query\":\"query { pod(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { costPerHr } }\"}" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['pod']['costPerHr'])" 2>/dev/null)
[ -z "$RATE" ] && RATE=4.0
START=$(stat -c %Y /proc/1)
echo "RATE=$RATE START=$START CAP=$cap" | tee /workspace/rate.txt
CAPTS=$(python3 -c "print(int($START + ($cap - 0.10)/$RATE*3600))")
touch /workspace/LAST_ACTIVITY
( while true; do
    now=$(date +%s)
    [ "$now" -ge "$CAPTS" ] && { echo "HARD_CAP $(date -u +%FT%TZ)"; break; }
    [ -e /workspace/RETRIEVED ] && { echo "RETRIEVED $(date -u +%FT%TZ)"; break; }
    if [ ! -e /workspace/JOB_RUNNING ]; then
      la=$(stat -c %Y /workspace/LAST_ACTIVITY); [ $((now - la)) -ge 1800 ] && { echo "IDLE $(date -u +%FT%TZ)"; break; }
    fi
    sleep 20
  done
  runpodctl stop pod "$RUNPOD_POD_ID" ) > /workspace/guard.log 2>&1 &
set -x
touch /workspace/JOB_RUNNING
python -m pip install --no-input vllm==0.11.0 transformers==4.57.1 numpy==2.1.2 Pillow==12.3.0 tokenizers==0.22.2 huggingface-hub==0.36.2 torch==2.8.0 hf_transfer 2>&1 | tail -1
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download;print('MODEL',snapshot_download('$model',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))"
python -c "from huggingface_hub import HfApi;print('MODEL_SHA',HfApi().model_info('$model').sha)" | tee /workspace/model_sha.txt
git clone -q https://github.com/sunrainyg/RandOpt.git /workspace/RandOpt; git -C /workspace/RandOpt checkout -q 4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca; git -C /workspace/RandOpt rev-parse HEAD
pip freeze > /workspace/pip-freeze.txt; nvidia-smi -L > /workspace/gpu.txt
rm -f /workspace/JOB_RUNNING; touch /workspace/LAST_ACTIVITY /workspace/SETUP_DONE; echo SETUP_DONE
while true; do
  for j in $(ls /workspace/jobs/*.sh 2>/dev/null | sort); do
    [ -e "${j%.sh}.rc" ] && continue
    touch /workspace/JOB_RUNNING; bash "$j" > "${j%.sh}.log" 2>&1; echo "rc=$? $(date -u +%FT%TZ)" > "${j%.sh}.rc"
    rm -f /workspace/JOB_RUNNING; touch /workspace/LAST_ACTIVITY
  done
  sleep 10
done
