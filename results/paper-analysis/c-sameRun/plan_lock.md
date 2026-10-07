# Session C plan lock: same-run RandOpt vs self-consistency on GSM8K / Qwen2.5-1.5B-Instruct

Committed before any Session C output. One H100, user-provided pod, **$8 hard cap** (pod guard). Context:
P2 (`../p2/P2_RESULT.md`) compared our SC@50 against the paper's *reported* RandOpt (76.4) across papers; reviewers
correctly note that is not a same-run comparison. Session C runs both arms on the same pod, model, prompts, scorer
and test set.

## Fixed setup
- Model `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; vLLM 0.11.0, pins as `scripts/pod_jobqueue.sh`.
- Data: `openai/gsm8k` main via datasets 3.6.0, converted to RandOpt's verl parquet format by `scripts/c_prep_gsm8k.py`
  (prompt = question + ' Let\'s think step by step and output the final answer after "####".'; ground truth = text
  after '####', commas removed). RandOpt train set = first 200 train rows (its loader); test = all 1319 test rows.
- **RandOpt arm:** RandOpt's own `randopt.py` @ 4000d34, unmodified except one added logging line that dumps the
  per-model extracted test answers (`all_answers`) before they are deleted (no behavioural change). Arguments:
  `--dataset gsm8k --model_name <pinned path> --num_engines 1 --tp 1 --cuda_devices 0 --train_samples 200
  --precision bfloat16 --max_tokens 1024 --sigma_values 0.0005,0.001,0.002 --global_seed 42
  --top_k_ratios 50/N,10/N` (σ set and seed from RandOpt's "paper" run script). Greedy decoding, RandOpt's own
  subtract-to-restore weights, RandOpt's own vote and scorer.
- **Population N (budget rule, data-independent):** a smoke run (12 perturbations, 200 train items, 30 test items)
  measures seconds per perturbation `s`. N = the largest of {2000, 1500, 1000} with
  spent + N·s·1.10 + $1.60 (test ensemble + SC) + $0.40 margin ≤ $7.90 at the pod rate. If even 1000 does not fit,
  the RandOpt arm is not run and the session reports SC only. (Paper: N = 5000; K = 50 is then top 1% vs our top
  2.5–5%; stated as a caveat.)
- **SC arm:** `scripts/p2_sc.py --task gsm8k --temps 0.7 --n 50` (greedy + SC at T = 0.7, top_p 1.0, seed 20261007),
  same handler. SC votes use exactly RandOpt's vote rule and scorer (most_common over non-empty extracted answers;
  `is_answer_correct(format_answer_for_check(v), gt)`) on the first 50 / 10 samples. P2's rule (empties counted) is
  reported as a secondary.

## Primary (C-1)
Paired per-item difference D = acc(RandOpt K = 50 vote) − acc(SC@50, T = 0.7) on the 1319 test items, with a paired
item bootstrap 95% CI (10,000 resamples, seed 0). RandOpt vote correctness uses RandOpt's own rule (Counter
most_common over non-empty answers; `format_answer_for_check` + `is_answer_correct`), recomputed from the dumped
answers and required to reproduce randopt.py's printed K = 50 accuracy exactly (else C-1 is reported INVALID).
- **RANDOPT AHEAD** if CI lower bound > 0; **SC AHEAD** if CI upper bound < 0; otherwise **NO DIFFERENCE DETECTED**;
  additionally **EQUIVALENT (±2 pp)** if the whole CI lies within [−2, +2].

## Secondary (reported)
- Same rule at K = 10 (RandOpt top-10 vote vs SC@10).
- Base greedy accuracy (both randopt.py's base evaluation and p2_sc greedy) vs paper Base 58.8 (gate ±2.0 pp; a gate
  failure is reported, it does not change C-1).
- RandOpt K = 50 vs the paper's 76.4, and SC@50 vs P2's SC@50 (79.8, other pod) as reproducibility checks.
- Selection: mean train reward of the top-50 vs all N; per-σ means.

## Caveats stated in advance
N < 5000 (budget); one RandOpt run (paper averages 3); 1.5B only. Training-time compute of RandOpt (N × 200
generations) is not charged to it; both arms use 50 generations per test question at test time.
