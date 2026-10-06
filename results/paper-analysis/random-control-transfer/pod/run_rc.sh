#!/usr/bin/env bash
# Approved run: cap 3.50 USD total pod spend. New guard (stop at cap-0.10), inference, RC_FINISHED.
cap=3.50
cd /workspace
set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a
RATE=$(grep -o 'RATE=[0-9.]*' /workspace/rate.txt | cut -d= -f2); START=$(stat -c %Y /proc/1)
CAPTS=$(python3 -c "print(int($START + ($cap - 0.10)/$RATE*3600))")
echo "APPROVED cap=$cap RATE=$RATE START=$START CAPTS=$CAPTS" >> /workspace/rate.txt
( fin=""
  while true; do
    now=$(date +%s)
    [ "$now" -ge "$CAPTS" ] && { echo "HARD_CAP $(date -u +%FT%TZ)"; break; }
    [ -e /workspace/RETRIEVED ] && { echo "RETRIEVED $(date -u +%FT%TZ)"; break; }
    if [ -e /workspace/RC_FINISHED ]; then [ -z "$fin" ] && fin=$now; [ $((now - fin)) -ge 1200 ] && { echo FINISHED_TIMEOUT; break; }; fi
    sleep 20
  done
  runpodctl stop pod "$RUNPOD_POD_ID" ) > /workspace/guard2.log 2>&1 &
touch /workspace/GO
cd /workspace/repro && tar xzf /workspace/rc.tgz --no-same-owner && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python -u scripts/random_control_transfer.py \
  --images /workspace/pt-images --upstream /workspace/RandOpt --out /workspace/repro/results/paper-analysis/random-control-transfer/gpu \
  --usd-per-hour "$RATE" --pod-start-epoch "$START" --cap-usd "$cap" > /workspace/run.log 2>&1
echo "exit=$? $(date -u +%FT%TZ)" > /workspace/RC_FINISHED
