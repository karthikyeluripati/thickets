#!/usr/bin/env bash
# RV-2a test phase: top 50 + BASE on the 1319 test questions, 8 workers (2 per GPU).
set -ux
grep -q '^rc=0 ' /workspace/jobs/02-select.rc
export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING; cd /workspace/repo
MP=$(cat /workspace/mp_q3.txt); ROOT=/workspace/rv/q3; pids=()
for layer in 0 1; do for g in 0 1 2 3; do w=$((layer*4+g))
  CUDA_VISIBLE_DEVICES=$g python -u scripts/gsm_randopt_fast.py --phase test --worker $w --workers 8 --root $ROOT --out out \
    --model-path $MP --gpu-mem 0.38 --prompt boxed --pop-seed 42 > $ROOT/out_test_$w.log 2>&1 & pids[$w]=$!; done
  [ $layer = 0 ] && for g in 0 1 2 3; do until grep -q ENGINE_READY $ROOT/out_test_$g.log 2>/dev/null; do sleep 3; kill -0 ${pids[$g]} || break; done; done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
ls $ROOT/out/test_rank*.json | wc -l; [ -e $ROOT/out/test_rank49.json ] && [ -e $ROOT/out/test_base.json ] || rc=1; exit $rc
