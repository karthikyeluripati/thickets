#!/usr/bin/env bash
# G1 8-GPU pod: guard (hard cap; stop if no run 10 min after setup; stop 15 min after success / 5 min after failure;
# RETRIEVED), setup, then GPU0: check -> gradient -> eval worker; GPUs1-7: eval workers (shared claim queue).
set -u
cap=${1:?}
cd /workspace
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 PIP_BREAK_SYSTEM_PACKAGES=1
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
RATE=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'content-type: application/json' \
  -d "{\"query\":\"query { pod(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { costPerHr } }\"}" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['pod']['costPerHr'])" 2>/dev/null)
[ -z "$RATE" ] && RATE=30.0
START=$(stat -c %Y /proc/1)
echo "RATE=$RATE START=$START CAP=$cap" | tee /workspace/rate.txt
CAPTS=$(python3 -c "print(int($START + ($cap - 0.30)/$RATE*3600))")
( fin=""; sd=""
  while true; do
    now=$(date +%s)
    [ "$now" -ge "$CAPTS" ] && { echo "HARD_CAP $(date -u +%FT%TZ)"; break; }
    [ -e /workspace/RETRIEVED ] && { echo "RETRIEVED $(date -u +%FT%TZ)"; break; }
    if [ -e /workspace/SETUP_DONE ] && [ ! -e /workspace/RUN_STARTED ]; then
      [ -z "$sd" ] && sd=$now; [ $((now - sd)) -ge 600 ] && { echo "IDLE_NO_RUN $(date -u +%FT%TZ)"; break; }
    fi
    if [ -e /workspace/GA_FINISHED ]; then
      [ -z "$fin" ] && fin=$now
      lim=900; grep -q 'rc=0' /workspace/GA_FINISHED || lim=300
      [ $((now - fin)) -ge $lim ] && { echo "FINISHED_TIMEOUT $lim"; break; }
    fi
    sleep 15
  done
  runpodctl stop pod "$RUNPOD_POD_ID" ) > /workspace/guard.log 2>&1 &
set -x
python -m pip install --no-input vllm==0.11.0 transformers==4.57.1 numpy==2.1.2 Pillow==12.3.0 tokenizers==0.22.2 huggingface-hub==0.36.2 torch==2.8.0 hf_transfer pandas pyarrow accelerate 2>&1 | tail -1
MP=$(HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download;print(snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct',revision='66285546d2b821cf421d4f5eb2576359d3770cd3',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))" | tail -1)
echo "MODEL_PATH=$MP" | tee /workspace/model_path.txt
git clone -q https://github.com/sunrainyg/RandOpt.git /workspace/RandOpt; git -C /workspace/RandOpt checkout -q 4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca
mkdir -p /workspace/repro; cd /workspace/repro; tar xzf /workspace/g1.tgz --no-same-owner
python scripts/g1_images.py /workspace/gqa-images || { echo "rc=11 images" > /workspace/GA_FINISHED; exit 11; }
pip freeze > /workspace/pip-freeze.txt; nvidia-smi -L > /workspace/gpu.txt
touch /workspace/SETUP_DONE /workspace/RUN_STARTED
eval $(sed 's/ /\n/g' /workspace/rate.txt | grep -E '^(RATE|START|CAP)=')
O=/workspace/repro/results/paper-analysis/g1/gpu; mkdir -p $O
EV="--images /workspace/gqa-images --upstream /workspace/RandOpt --out $O --model-path $MP --usd-per-hour $RATE --pod-start-epoch $START --cap-usd $CAP"
NG=$(nvidia-smi -L | wc -l)
pids=()
for g in $(seq 1 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 python -u scripts/g1_eval.py --worker gpu$g $EV > /workspace/eval_gpu$g.log 2>&1 & pids+=($!)
done
( export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
  python -u scripts/g1_check.py --upstream /workspace/RandOpt --out $O --model-path $MP > /workspace/check.log 2>&1; echo "check rc=$?" >> /workspace/gpu0_status.txt
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/g1_gradient.py --images /workspace/gqa-images --out $O --check $O/vllm_stream_check.json --model-path $MP > /workspace/gradient.log 2>&1; echo "gradient rc=$?" >> /workspace/gpu0_status.txt
  python -u scripts/g1_eval.py --worker gpu0 $EV > /workspace/eval_gpu0.log 2>&1; echo "eval0 rc=$?" >> /workspace/gpu0_status.txt ) & pids+=($!)
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "rc=$rc $(date -u +%FT%TZ) $(tr '\n' ' ' < /workspace/gpu0_status.txt)" > /workspace/GA_FINISHED
