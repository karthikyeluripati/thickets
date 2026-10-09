#!/usr/bin/env bash
# Finish RV-2a's selection: 15 sub-workers (W = 36) on 8 slots, 2 per GPU (amendment_1.md).
set -ux
grep -q '^rc=0 ' /workspace/jobs/01-prep.rc
export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING; cd /workspace/repo
MP=$(cat /workspace/mp_q3.txt); ROOT=/workspace/rv/q3; OUT=$ROOT/out
python scripts/rv_resume_q3.py --out $OUT --gpus 4 | tee $ROOT/resume_plan.txt
chain() { local g=$1; shift; local s
  for s in "$@"; do
    CUDA_VISIBLE_DEVICES=$g python -u scripts/gsm_randopt_fast.py --phase select --worker $s --workers 36 --root $ROOT --out out \
      --model-path $MP --gpu-mem 0.38 --prompt boxed --pop-seed 42 > $ROOT/resume_select_$s.log 2>&1 || { echo "SUBWORKER_FAILED $s"; return 1; }
  done; }
mapfile -t SL < <(grep '^slot ' $ROOT/resume_plan.txt)
pids=()
for i in 0 1 2 3; do set -- ${SL[$i]}; g=$2; shift 3; first=$1; chain $g "$@" & pids+=($!)
  until grep -q ENGINE_READY $ROOT/resume_select_$first.log 2>/dev/null; do sleep 3; kill -0 ${pids[-1]} 2>/dev/null || break; done; done
for i in 4 5 6 7; do set -- ${SL[$i]}; g=$2; shift 3; chain $g "$@" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
python - <<'PY'
import json, glob
k = set()
for f in glob.glob('/workspace/rv/q3/out/select_*.jsonl'):
    for l in open(f):
        if l.strip(): k.add(json.loads(l)['k'])
print('UNIQUE_SCORED', len(k)); assert len(k) == 5000
PY
exit $rc
