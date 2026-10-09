#!/usr/bin/env bash
# RV-1 (public templates, chosen on the 200 selection questions among the NEW templates only) and RV-5 (SC@50 at
# T = 0.5 and 1.0 under the PS-chosen prompt, Qwen-3B boxed and GQA direct). One pipeline per GPU, in parallel.
set -ux
grep -q '^rc=0 ' /workspace/jobs/01-prep.rc
. /workspace/repo/scripts/pod_rv_common.sh
TOP=$A/o2-olmo-prompt/top50_reconstructed.json  # required by o2_eval.py; unused by the base/sc arms
gsm() { local tag=$1 gpu=$2 mp out p ch; mp=$(cat /workspace/mp_$tag.txt); out=/workspace/rv/rv1/$tag; mkdir -p $out
  for p in harness_cot harness_plain simple_evals; do
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/o2_eval.py --prompt $p --arms base --split select --top50 $TOP --model-path $mp --out $out || return 1
  done
  ch=$(python scripts/rv_choose.py --task gsm8k --out $out) || return 1; echo "RV1_CHOSEN $tag $ch"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/o2_eval.py --prompt $ch --arms base,sc --top50 $TOP --model-path $mp --out $out || return 1
  if [ $tag = q3 ]; then for T in 0.5 1.0; do
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/o2_eval.py --prompt boxed --arms sc --temperature $T --top50 $TOP --model-path $mp --out /workspace/rv/rv5/q3 || return 1
  done; fi
  echo "PIPE_DONE $tag"
}
gqa() { local gpu=$1 mp out p ch; mp=$(cat /workspace/mp_gqa.txt); out=/workspace/rv/rv1/gqa; mkdir -p $out
  for p in llava blip; do
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/g3_eval.py --split selection --budgets 256 --arms base --prompt $p --model-path $mp --out $out/$p || return 1
  done
  ch=$(python scripts/rv_choose.py --task gqa --out $out) || return 1; echo "RV1_CHOSEN gqa $ch"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/g3_eval.py --budgets 256 --arms base,sc --prompt $ch --model-path $mp --out $out/$ch || return 1
  for T in 0.5 1.0; do
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/g3_eval.py --budgets 256 --arms sc --prompt direct --temperature $T --model-path $mp --out /workspace/rv/rv5/gqa/direct || return 1
  done
  echo "PIPE_DONE gqa"
}
gsm q15 0 > /workspace/rv/pipe_q15.log 2>&1 & p0=$!
gsm q3 1 > /workspace/rv/pipe_q3.log 2>&1 & p1=$!
gsm olmo 2 > /workspace/rv/pipe_olmo.log 2>&1 & p2=$!
gqa 3 > /workspace/rv/pipe_gqa.log 2>&1 & p3=$!
rc=0; for p in $p0 $p1 $p2 $p3; do wait $p || rc=1; done
grep -h "RV1_CHOSEN\|PIPE_DONE" /workspace/rv/pipe_*.log; exit $rc
