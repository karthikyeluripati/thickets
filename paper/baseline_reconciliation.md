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

We do not attribute the gap to any one of these without a controlled run.

## 6. Planned controlled check (Part N, ≤ $5 GPU)

Run the pinned base checkpoint on all 561 TEST questions with the official
OmniSpatial manual-CoT + `re` prompt and parser. The run holds model, engine and
images fixed and changes only the prompt, extraction and length. Section 7 is
filled in from that run.

## 7. Controlled run result

*Pending (Part N).*

## Status

**`BASELINE_DISCREPANCY_UNRESOLVED`** (provisional; to be updated after Part N).

**Limitation to keep in the paper.** Our base differs from the published EASI
Qwen3-family manual-CoT number on the identical 561 items by +5.9 pp (micro).
The published row does not identify its checkpoint, decoding or repeat count.
All our claims are relative (candidate versus base under one frozen protocol),
so they don't depend on matching the absolute level.

### Sources

- OmniSpatial: arXiv 2506.03135 (ICLR 2026), Tables 2 and 11; github.com/qizekun/OmniSpatial README; qizekun.github.io/omnispatial
- EASI: arXiv 2508.13142v5, Tables 13–14 and §3
- Checked, no relevant row: arXiv 2607.00881, 2505.17012, 2511.21688, 2608.15605
