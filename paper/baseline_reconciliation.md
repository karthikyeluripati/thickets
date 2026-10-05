# OmniSpatial baseline reconciliation: Qwen3-VL-8B, Perspective Taking

Our unperturbed baseline scores **259/561 = 46.17 %** on the official
Perspective-Taking test. That number is the micro accuracy under the official
direct-letter prompt. This note checks it against published numbers, using
sources only.

## 1. What is published

| Source | Model label | Protocol | Ego | Allo | Hypo | PT (macro) | PT (micro) |
|---|---|---|---:|---:|---:|---:|---:|
| OmniSpatial paper (arXiv 2506.03135, ICLR 2026), Tables 2/11 | — | no Qwen3-VL row | — | — | — | — | — |
| OmniSpatial repo README and project leaderboard | — | no Qwen3-VL row (Qwen2.5-VL at most) | — | — | — | — | — |
| EASI, *Holistic Evaluation of Multimodal LLMs on Spatial Intelligence* (arXiv 2508.13142v5, 30 Dec 2025), Table 13 "OmniSpatial (Official Protocol)" | "Qwen3-8B-Instruct [65]" | Manual-CoT, official prompt; truncated generations counted as wrong | 68.63 | 31.91 | 43.37 | 47.97 | 40.29 (226/561) |
| EASI Table 14 (EASI protocol) | same | unified zero-shot CoT; CAA metric (chance-adjusted) | — | — | — | — | not comparable |
| **This work, base (official direct prompt)** | Qwen/Qwen3-VL-8B-Instruct @ `0c351dd` | direct letter, greedy, 16 tokens, first-character scorer | **77.45** (79/102) | **37.50** (141/376) | **46.99** (39/83) | **53.98** | **46.17** (259/561) |

Other papers checked had no Qwen3-VL-8B OmniSpatial Perspective-Taking row:

- **OmniView-Space** (2607.00881): it lists Qwen3-VL-8B-Instruct, but only on
  MindCube, MMSI and SPAR. A search-engine summary attributed those numbers
  to OmniSpatial. That attribution was wrong.
- **SpatialScore** (2505.17012): it uses 205 OmniSpatial items mixed into its
  own categories.
- **G2VLM** (2511.21688): Qwen2.5-VL only.
- **AlloEgo-VLM** (2608.15605): no OmniSpatial table.

The only published Qwen3-family OmniSpatial Perspective-Taking result found is
the EASI Table 13 row.

## 2. Is the EASI row the same test set?

**Yes.** Each EASI subtask score is an exact fraction of our TEST561
denominators:

- Ego: 70/102 = 68.63
- Allo: 120/376 = 31.91
- Hypo: 36/83 = 43.37

Our TEST561 is the full official Perspective-Taking test, with nothing
filtered:

| Subtask | Questions |
|---|---:|
| Allocentric | 376 |
| Egocentric | 102 |
| Hypothetical | 83 |

The exact fractions are also consistent with **one pass**, not an average of 5
sampled repeats. The OmniSpatial paper averages 5 runs, but EASI does not state a
repeat count for Table 13.

## 3. Differences between the EASI setting and ours

| Factor | EASI Table 13 / official OmniSpatial code | This work | Source |
|---|---|---|---|
| Model identity | Labelled "Qwen3-8B-Instruct". Ref. [65] is the Qwen3 (text LLM) technical report, and the string "Qwen3-VL" does not occur in the paper. The Ego score (68.6 vs 22.6 blind / 24.8 random) implies the model saw images, so it is almost certainly a Qwen3-VL-8B checkpoint, but the exact checkpoint and revision are **not stated**. | Qwen3-VL-8B-Instruct @ `0c351dd01ed8…` (model, processor, tokenizer, template pinned) | EASI p. Tab. 13 and ref list; our protocol |
| Prompt | Manual-CoT system prompt + `re` format ("Answer: X") | `SYS_PROMPTS['none']` + `FORMAT_PROMPTS['direct']` (official direct mode) | official `system_prompts.py` blob `38fa3eb`; our protocol |
| Answer extraction | Official `re`: `Answer\s*:\s*([A-D])\b`, falling back to "A" if no match. EASI uses its own two-stage extraction (rules, then LLM judge) and counts truncated outputs as wrong. | First non-space character; invalid ⇒ wrong; no fallback | official `qwenvl_eval.py` at `208bac2`; EASI §3 |
| Generation length | Official default `max_new_tokens=8192` (CoT) | 16 tokens (direct letter) | as above |
| Decoding | Official script: HF `generate` with checkpoint defaults (Qwen3-VL: T=0.7, top-p 0.8, top-k 20). EASI: not stated for Table 13. | Greedy (T=0) | checkpoint `generation_config.json`; our protocol |
| Repeats | Official: 5 (paper Table 2 caption). EASI: not stated; fractions consistent with 1. | 1 deterministic pass | — |
| Engine | Official script pins `transformers==4.49.0` and loads `Qwen2_5_VLForConditionalGeneration`, so it **cannot load Qwen3-VL unmodified**. EASI used its own harness (VLMEvalKit-style). | vLLM 0.11.0, BF16 | official README; our protocol |
| Image handling | Official: `qwen_vl_utils` defaults. EASI: not stated. | Original images, processor's own resize; RGB hashes recorded per image | our protocol |
| Coarse PT metric | **Macro** mean of Ego/Allo/Hypo (OmniSpatial Table 11 caption) | We report micro (259/561); macro is 53.98 | OmniSpatial p. 30 |

## 4. Size and direction of the gap

| Comparison | EASI | Ours | Δ (ours − EASI) |
|---|---:|---:|---:|
| PT micro | 40.29 | 46.17 | +5.88 pp (+33 questions) |
| PT macro (OmniSpatial convention) | 47.97 | 53.98 | +6.01 pp |
| Ego | 68.63 | 77.45 | +8.8 pp (+9) |
| Allo | 31.91 | 37.50 | +5.6 pp (+21) |
| Hypo | 43.37 | 46.99 | +3.6 pp (+3) |

Our direct-letter base is **higher** than the only published Qwen3-family
manual-CoT number in every subtask. It is not lower. The direction does not
suggest a broken pipeline on our side; for example, a failing parser would push
our score down, not up. Quoting the paper's convention (macro) next to our
micro number exaggerates the apparent gap. Like for like, the two metrics differ
by about 6 pp.

## 5. What explains the gap?

The source-level evidence above leaves several candidate causes, which the
sources alone can't separate:

1. **Prompt and output format.** Manual-CoT with 8,192 tokens versus a single
   direct letter. Truncation and regex misses count as wrong in EASI.
2. **Decoding.** Sampling versus greedy.
3. **Checkpoint identity and revision.** Not stated by EASI.
4. **Harness and image preprocessing.**

The sources alone do not attribute the gap to any one of these; the controlled
run below tests the prompt and decoding factors directly.

## 6. Controlled check (Part N, 1× H100, 2026-10-05)

The run used the pinned base checkpoint on all 561 TEST questions. Model,
engine (vLLM 0.11.0) and images (SHA256-verified) were held fixed. Only the
prompt, answer extraction and length changed, to:

- the official OmniSpatial manual-CoT + `re` prompt, rebuilt from the vendored
  official `system_prompts.py`;
- the official parser, copied from `qwenvl_eval.py@208bac2`: last
  `Answer: X`, case-insensitive, fallback "A";
- `max_new_tokens = 8192`.

**Engine check.** In the same process, our original direct-letter prompt
reproduced the stored base outputs byte for byte: 259/561, 0 mismatches.

| Base, official manual-CoT | Ego /102 | Allo /376 | Hypo /83 | Total /561 | Micro % | Truncated at 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| Greedy (T = 0) | 75 | 140 | 36 | 251 | 44.74 | 29 |
| Sampled, checkpoint defaults (T 0.7, top-p 0.8, top-k 20), seed 1 | 82 | 141 | 37 | 260 | 46.35 | 27 |
| … seed 2 | 77 | 133 | 36 | 246 | 43.85 | 23 |
| … seed 3 | 73 | 129 | 33 | 235 | 41.89 | 35 |
| … seed 4 | 74 | 115 | 31 | 220 | 39.22 | 35 |
| Sampled mean (SD) | 76.5 (4.0) | 129.5 (10.9) | 34.2 (2.8) | 240.2 (16.9) | 42.83 (3.0) | — |
| **EASI Table 13** | 70 | 120 | 36 | 226 | 40.29 | — |
| Ours, direct letter (study protocol) | 79 | 141 | 39 | 259 | 46.17 | — |

The sampled runs use the decoding of the official script, which calls HF
`generate` with checkpoint defaults. Two sampled runs agree on only 74.8% of
answers. A single sampled pass of the official protocol therefore moves by
±3 pp on these 561 items.

## 7. What the controlled run shows

1. **The prompt and protocol difference explains the gap.** Under the official
   protocol our own pinned checkpoint scores 220–260 in single sampled passes
   (greedy 251). The published single-pass EASI value of 226 lies inside that
   range, as do its Allocentric (120) and Hypothetical (36) counts. Only
   Egocentric (70) falls slightly below our 4-run range (73–82). That is
   1.6 SD under our mean, which is unremarkable for 4 runs.
2. **The protocol is high-variance.** Free-form CoT plus "A" fallback on
   truncation (23–35 runaway generations per pass) produces a single-pass SD of
   about 3 pp. Our direct-letter greedy protocol has no truncation or sampling
   noise and sits at the top of that range (46.17%).
3. **No checkpoint difference has to be invoked.** We cannot verify which
   checkpoint EASI used ("Qwen3-8B-Instruct", citing the Qwen3 text report). But
   its number is reproduced within noise by Qwen3-VL-8B-Instruct @ `0c351dd`
   under the official protocol.

## Status

**`BASELINE_DISCREPANCY_EXPLAINED`**

**Exact cause.** The published number comes from a different evaluation
protocol from ours: the official manual-CoT prompt with `re` extraction and
"A" fallback, a single pass, and the model's default sampling (inferred from
the official script). Ours is the official direct-letter prompt with greedy
decoding. Under the published protocol, our pinned checkpoint scores 220–260/561
(SD 16.9) in single passes, which contains the published 226/561. The
remaining difference (our direct-letter 259 against their 226) is therefore
protocol choice plus single-pass sampling variance, not a pipeline error.

**Caveats for the paper:**

- EASI does not state its checkpoint revision, decoding or repeat count. "Single
  pass" is inferred from its exact k/n subtask fractions.
- We ran 4 sampled passes, not the OmniSpatial paper's 5. Our sampled-run mean
  is 42.8% (micro), or 50.2% macro.
- All study claims are relative (candidate versus base under one frozen
  protocol), so they don't depend on the absolute level.

## GPU cost

1× H100 80GB at $3.49/h, from pod start 16:31:34 to DONE 17:13:27 UTC, then
automatic stop (≈ 1 min). **≈ $2.50 total, under the $5 cap.**

- Setup and model load: $0.25.
- Integrity check and preflight: $0.22.
- Priority 1: $0.25.
- Priority 2: $0.58.
- 4 sampled repeats: $1.14.

Per-step log: `results/paper-analysis/prompt-robustness/cost_log.json`.

### Sources

- OmniSpatial: arXiv 2506.03135 (ICLR 2026), Tables 2 and 11; github.com/qizekun/OmniSpatial README; qizekun.github.io/omnispatial
- EASI: arXiv 2508.13142v5, Tables 13–14 and §3
- Checked, no relevant row: arXiv 2607.00881, 2505.17012, 2511.21688, 2608.15605
