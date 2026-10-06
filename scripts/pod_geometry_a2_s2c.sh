#!/usr/bin/env bash
# GPU-A run 2, S2c-only continuation (HF only): guard with hard cap; stops 20 min after a successful finish, 5 min after a
# failed one, or on RETRIEVED. Inputs: ga2s2c.tgz (repo files + gpu2/base_contrast_fp32.json from the earlier session).
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
( fin=""
  while true; do
    now=$(date +%s)
    [ "$now" -ge "$CAPTS" ] && { echo "HARD_CAP $(date -u +%FT%TZ)"; break; }
    [ -e /workspace/RETRIEVED ] && { echo "RETRIEVED $(date -u +%FT%TZ)"; break; }
    if [ -e /workspace/GA_FINISHED ]; then
      [ -z "$fin" ] && fin=$now
      lim=1200; grep -q 'hf=0' /workspace/GA_FINISHED || lim=300
      [ $((now - fin)) -ge $lim ] && { echo "FINISHED_TIMEOUT $lim"; break; }
    fi
    sleep 20
  done
  runpodctl stop pod "$RUNPOD_POD_ID" ) > /workspace/guard.log 2>&1 &
set -x
python -m pip install --no-input transformers==4.57.1 numpy==2.1.2 Pillow==12.3.0 tokenizers==0.22.2 huggingface-hub==0.36.2 torch==2.8.0 torchvision==0.23.0 hf_transfer 2>&1 | tail -1
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download;print('MODEL',snapshot_download('Qwen/Qwen3-VL-8B-Instruct',revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))"
mkdir -p /workspace/repro; cd /workspace/repro; tar xzf /workspace/ga2s2c.tgz --no-same-owner; python scripts/geometry_extract_images.py
pip freeze > /workspace/pip-freeze.txt; nvidia-smi -L > /workspace/gpu.txt
echo SETUP_DONE
O=/workspace/repro/results/paper-analysis/geometry-gpu-a/gpu2; mkdir -p $O
cp results/paper-analysis/geometry-gpu-a/pod2/repro/results/paper-analysis/geometry-gpu-a/gpu2/base_contrast_fp32.json $O/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python -u scripts/geometry_gpu_a2.py --resume-s2c --images /workspace/pt-images --out $O \
  --vllm-check results/paper-analysis/geometry-gpu-a/gpu/vllm_stream_check.json \
  --usd-per-hour "$RATE" --pod-start-epoch "$START" --cap-usd "$cap" > /workspace/run.log 2>&1
h=$?
echo "hf=$h $(date -u +%FT%TZ)" > /workspace/GA_FINISHED
