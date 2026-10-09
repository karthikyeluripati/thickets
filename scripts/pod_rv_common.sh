#!/usr/bin/env bash
# Shared functions for the RV pod session (results/paper-analysis/rv-reviewer-round/plan_lock.md). Sourced by the job
# files in that folder's jobs/. Earlier sessions kept this file on the pod only; it is committed now so the runs reproduce.
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 VLLM_LOGGING_LEVEL=WARNING
cd /workspace/repo
G=${G:-4}  # GPUs on the pod
A=results/paper-analysis
RV=$A/rv-reviewer-round

dl() {  # dl <repo> <revision> <tag>: snapshot the model, record its local path in /workspace/mp_<tag>.txt
  python -c "from huggingface_hub import snapshot_download as s;print(s('$1',revision='$2',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))" \
    | tail -1 > /workspace/mp_$3.txt; cat /workspace/mp_$3.txt
}

spent() { python3 -c "import time;r=dict(x.split('=') for x in open('/workspace/rate.txt').read().split());print(round((time.time()-int(r['START']))/3600*float(r['RATE']),2))"; }

# launch <select|test> <root> <out> <workers-per-gpu> <model-path> [runner args...]
# W = wpg*G workers of gsm_randopt_fast.py; each next worker on a GPU starts only after the previous one printed
# ENGINE_READY (vLLM's startup memory check fails otherwise). Failed workers are re-run once, in parallel (resumable).
launch() {
  local ph=$1 root=$2 o=$3 wpg=$4 mp=$5; shift 5
  local W=$((wpg * G)); local -a pids=(); mkdir -p "$root"
  local mem; mem=$(python3 -c "print(round(0.85/$wpg, 2))")
  for layer in $(seq 0 $((wpg - 1))); do
    for g in $(seq 0 $((G - 1))); do
      local w=$((layer * G + g))
      CUDA_VISIBLE_DEVICES=$g python -u scripts/gsm_randopt_fast.py --phase "$ph" --worker "$w" --workers "$W" --root "$root" --out "$o" \
        --model-path "$mp" --gpu-mem "$mem" "$@" > "$root/${o}_${ph}_$w.log" 2>&1 &
      pids[$w]=$!
    done
    for g in $(seq 0 $((G - 1))); do
      local w=$((layer * G + g))
      until grep -q ENGINE_READY "$root/${o}_${ph}_$w.log"; do kill -0 "${pids[$w]}" 2>/dev/null || break; sleep 3; done
    done
  done
  local bad=""
  for w in $(seq 0 $((W - 1))); do wait "${pids[$w]}" || bad="$bad $w"; done
  if [ -n "$bad" ]; then  # one retry round, one worker per GPU at a time
    echo "RETRY workers:$bad"; local -a rp=(); local i=0 rc=0 p
    for w in $bad; do
      local g=$((i % G)); i=$((i + 1))
      CUDA_VISIBLE_DEVICES=$g python -u scripts/gsm_randopt_fast.py --phase "$ph" --worker "$w" --workers "$W" --root "$root" --out "$o" \
        --model-path "$mp" --gpu-mem 0.85 "$@" > "$root/${o}_${ph}_${w}_retry.log" 2>&1 &
      rp+=($!)
      if [ $((i % G)) -eq 0 ]; then for p in "${rp[@]}"; do wait "$p" || rc=1; done; rp=(); fi
    done
    for p in "${rp[@]}"; do wait "$p" || rc=1; done
    [ $rc -eq 0 ] || { echo "LAUNCH_FAILED $ph $o"; return 1; }
  fi
  echo "LAUNCH_DONE $ph $o W=$W"
}

# search_row <tag> <root> <model-path> <prompt> <pop-seed> <randopt-prompt selection ref> <search-prompt selection ref>
#            <search-prompt test-base ref> <randopt.py log for fidelity | none> <cap>
# Gates (lock "Gates"), smoke projection, then the full selection (3 workers per GPU) and test (1 per GPU).
search_row() {
  local tag=$1 root=$2 mp=$3 prompt=$4 seed=$5 roref=$6 spref=$7 tbref=$8 lg=$9 cap=${10}
  mkdir -p "$root"
  # (a) environment + fidelity under RandOpt's prompt: base + perturbations k < 24 of THIS population seed
  launch select "$root" out_fid 1 "$mp" --prompt randopt --first 24 --pop-seed "$seed" || return 1
  # (b) search-prompt base on the test set, same engine settings as Q2/O2 (o2_eval.py)
  CUDA_VISIBLE_DEVICES=0 python -u scripts/o2_eval.py --prompt "$prompt" --arms base --top50 $A/o2-olmo-prompt/top50_reconstructed.json \
    --model-path "$mp" --out "$root/envtest" > "$root/envtest.log" 2>&1 || return 1
  # (c) smoke under the search prompt: 15 perturbations per worker, 12 workers
  launch select "$root" out 3 "$mp" --prompt "$prompt" --first $((15 * 3 * G)) --pop-seed "$seed" || return 1
  python scripts/rv_gates.py --root "$root" --prompt "$prompt" --ro-ref "$roref" --sp-ref "$spref" --tb-ref "$tbref" --log "$lg" \
    --spent "$(spent)" --cap "$cap" --workers $((3 * G)) --gpus $G | tee "$root/gates.txt"
  grep -q '^ENV .* PASS$' "$root/gates.txt" || { echo "ROW_INVALID_ENV $tag"; return 2; }
  grep -q '^PROJECTION .* GO$' "$root/gates.txt" || { echo "ROW_NOT_RUN_BUDGET $tag"; return 3; }
  launch select "$root" out 3 "$mp" --prompt "$prompt" --pop-seed "$seed" || return 1
  launch test "$root" out 1 "$mp" --prompt "$prompt" --pop-seed "$seed" || return 1
  touch "$root/DONE"; echo "ROW_DONE $tag spent=$(spent)"
}
