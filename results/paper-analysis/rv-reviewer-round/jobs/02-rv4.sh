#!/usr/bin/env bash
# RV-4: does randopt.py reproduce its own printed BASE selection reward (73.00, C) for Qwen2.5-1.5B on a fresh pod?
# A: randopt.py smoke exactly as C's smoke but with this pod's 4 engines; B: the same with one engine; C: our runner's
# base + perturbations k < 24 under RandOpt's prompt (reused by 06 as its fidelity pass).
set -ux
grep -q '^rc=0 ' /workspace/jobs/01-prep.rc
. /workspace/repo/scripts/pod_rv_common.sh
MP=$(cat /workspace/mp_q15.txt); R=/workspace/rv/rv4; mkdir -p $R
export VLLM_NO_USAGE_STATS=1 VLLM_DISABLE_COMPILE_SAMPLER=1
for v in "A 4 0,1,2,3" "B 1 0"; do set -- $v
  (cd /workspace/RandOpt && python -u randopt.py --dataset gsm8k --model_name $MP --num_engines $2 --tp 1 --cuda_devices $3 --train_samples 200 \
     --test_samples 30 --precision bfloat16 --max_tokens 1024 --sigma_values 0.0005,0.001,0.002 --global_seed 42 --population_size 24 \
     --top_k_ratios 0.25 --experiment_dir $R/smoke_$1 2>&1 | grep -v "RayWorkerWrapper" > $R/randopt_$1.log)
  grep -E "Train reward|Test accuracy|Batch [0-9]+ \|" $R/randopt_$1.log | head -8
done
launch select /workspace/rv/q15 out_fid 1 $MP --prompt randopt --first 24 --pop-seed 42
python - <<'PY'
import json, glob, re
ours = 100 * json.load(open('/workspace/rv/q15/out_fid/base_select.json'))['reward']
pr = {v: re.search(r'Train reward: ([\d.]+)%', open(f'/workspace/rv/rv4/randopt_{v}.log').read()).group(1) for v in 'AB'}
print(f'RV4 randopt.py_A={pr["A"]} randopt.py_B={pr["B"]} ours={ours:.2f} C_printed=73.00 PS=68.5 GB2=68.00')
PY
echo RV4_DONE
