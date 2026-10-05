# Expert Mirages in Weight Space: paper analysis summary

Branch: `research/expert-mirages-paper`. Every analysis here is built from the
committed artifacts of the Perspective-Taking N = 5,000 study. No old artifact
was modified, and no candidate, search or σ was added. The only new inference
is Part N, the post-hoc official-prompt diagnostic, run on 1× H100 under a $5
cap; it is listed in the GPU cost log.

Stage names: **SEARCH** (SEARCH200), **RERANK** (historical VALIDATION200, a
second-stage selection set) and **TEST** (official, 561). File names are
unchanged.

## Final verdict

**`PAPER_READY_WORKSHOP`**

The evidence package for the primary claim is complete. Every number reproduces
from committed artifacts; the precommitted chain is git-timestamped
(`table_analysis_provenance.tex`); the baseline discrepancy is explained
(`BASELINE_DISCREPANCY_EXPLAINED`); and no further experiment is needed.

The verdict holds only if the draft adopts the corrections in
`paper_claim_audit.md`:

1. Call RERANK a second-stage selection set, not validation.
2. Drop the "Egocentric share caused the collapse" and template-overlap
   explanations. The subtask mix explains under half of the drop (Part E), and
   unseen question forms gain as much as seen ones (Part F).
3. Report the oracle only together with its symmetric churn.
4. State prompt sensitivity as a limitation (Part N): the +8 pp exists only
   under the direct-letter prompt used for selection.
5. Label every post-hoc analysis as post hoc.

Writing the paper from the old report wording would make it
`PAPER_NEEDS_FIXES`.

## 1. Deliverables

| Part | Output |
|---|---|
| A master dataset | `results/paper-analysis/paper_master_predictions.parquet` (1,137,611 rows: candidate × example × phase); checks in `paper_master_checks.json` (all pass) |
| B main table | `paper/tables/table_main_results.tex` |
| C–K figures | `paper/figures/fig1…fig8.{pdf,svg,png}`; data beside each in `paper/figures/data/`; code in `paper/figures/scripts/` |
| E decomposition | `fig3_shift_decomposition`, `table_shift_decomposition.tex` |
| F form overlap | `fig4_form_overlap_gain`, `table_question_form_overlap.tex` |
| G fragile items | `fig5_fragile_items` |
| H extreme-selection null | `fig6_extreme_selection_null` |
| I committee | `fig7_committee_convergence`, `table_committee_diversity.tex` |
| J ties | `table_tie_instability.tex` |
| K σ trade-off | `fig8_sigma_tradeoff` |
| L provenance | `table_analysis_provenance.tex` |
| M baseline | `paper/baseline_reconciliation.md` |
| N/O prompt robustness | `results/paper-analysis/prompt-robustness/`, `table_prompt_robustness.tex`, `fig9_prompt_robustness` |
| R other tables | `table_setup_differences.tex`, `table_forensic_integrity.tex` |
| S claim audit | `paper/paper_claim_audit.md` |

Reproduce with:

```
python paper/figures/scripts/build_master.py
python paper/figures/scripts/analyses.py
python paper/figures/scripts/provenance.py
python paper/figures/scripts/tables_static.py
```

Each analysis part has its own seeded random stream, so all numeric outputs
(data files, tables) and figure files regenerate byte-identically; figures carry no
embedded timestamps. The tables need `booktabs` and `adjustbox`, and all of them
compile.

## 2. Results

| | Base | 9504111 | Gain |
|---|---:|---:|---:|
| SEARCH (200) | 72 | 77 | +2.5 pp |
| RERANK (200) | 79 | 95 | **+8.0 pp** [3.5, 12.5] |
| TEST (561) | 259 | 244 | **−2.67 pp** [−5.3, 0.0] |
| SEARCH top-50 majority vote, TEST | 259 | 259 | +0.00 pp |

### Findings that support the mirage account

- **Selection from a pool of near-copies (Part H).** Picking the best of 50 random
  candidates with the SEARCH-selected σ mix already gives an expected RERANK
  maximum of **+5.8 pp** (95% range +3.5 to +7.5). The observed +8.0 is one
  question above the largest of all 500 precommitted random candidates (+7.5).
  An independent-flip null gives only +3.7, so correlated flipping matters, not
  just chance.
- **Shared fragile items (Part G).** Single RERANK questions are corrected by up
  to 80% of random σ = 0.002 candidates. 9504111's wins sit on items random
  candidates fix 32% of the time, against 4.4% for other base-wrong items.
- **Selection instability (Part J).** 9504111 entered the top 50 through a 33-way
  tie at 77/200, and 25 of the top 50 were decided by the hash tie-break.
- **No transfer (Part D).** Across the 50 SEARCH-selected candidates, RERANK gain
  does not predict TEST gain (r = −0.27, p = 0.06).
- **Committee = base (Part I).**
  - The base answer is the committee majority on 96.6% of TEST questions.
  - Members disagree pairwise on 12% of questions (error Jaccard 0.84).
  - The 64.3% "oracle" is symmetric churn: 102 base-wrong questions fixed vs 114
    base-right broken by at least one member (38 vs 37 at ≥ 10 members), and the
    majority fixes 7 and breaks 7. This is not complementary expertise.
- **σ trade-off (Part K).** Larger σ raises the RERANK tail (23/125 random
  σ = 0.002 candidates ≥ +3 pp) while lowering mean SEARCH (−2.0 pp) and TEST
  accuracy.

### Findings that contradict the previous narrative (reported, not hidden)

- **Subtask mix does not explain most of the collapse (Part E).** With RERANK
  per-subtask behaviour reweighted to the TEST subtask mix, the predicted TEST
  gain is **+3.88 pp** (bootstrap [−1.5, +9.3]), not a loss. The 10.67-pp drop
  splits into −4.12 pp from the mixture and **−6.56 pp within subtasks**:
  Egocentric goes from +11.8 to −4.9 pp. The forensic report's "Egocentric
  share" explanation is incomplete.
- **Question-form overlap is not the mechanism (Part F).** RERANK gain is
  +11.5 pp on questions whose exact text appeared in SEARCH and **+14.6 pp on
  unseen forms**. "Templated items seen during search" cannot carry the
  explanation.
- **Baseline level (Parts M/N).** Our direct-letter base (259/561) is +5.9 pp
  above the only published Qwen3-family OmniSpatial Perspective-Taking number
  (EASI Table 13: 226/561, same items).
  - Under the official manual-CoT protocol, our pinned checkpoint scores 220–260
    in single sampled passes (greedy 251), which contains 226.
  - **`BASELINE_DISCREPANCY_EXPLAINED`**: protocol choice plus single-pass
    sampling variance (SD ≈ 3 pp). Details in `baseline_reconciliation.md`.

## 3. Part N: official-prompt robustness (`POST_HOC_PROMPT_ROBUSTNESS_DIAGNOSTIC`)

Setup: the same pinned weights and engine, with the candidate state hash
verified, greedy decoding, and the official OmniSpatial manual-CoT prompt and
`re` parser. Paired bootstrap CIs. Files are in
`results/paper-analysis/prompt-robustness/`, `table_prompt_robustness.tex` and
`fig9_prompt_robustness`.

| Prompt | Split | Base | Candidate 9504111 | Gain (pp) [95% CI] |
|---|---|---:|---:|---:|
| Direct-letter original | RERANK | 79/200 | 95/200 | **+8.00** [+3.5, +13.0] |
| Direct-letter original | TEST | 259/561 | 244/561 | −2.67 [−5.3, 0.0] |
| Official OmniSpatial-style | RERANK | 97/200 | 94/200 | **−1.50** [−7.0, +4.0] |
| Official OmniSpatial-style | TEST | 251/561 | 227/561 | **−4.28** [−7.8, −0.7] |

With the strict parser (no "A" fallback) the gains are −0.5 (RERANK) and −4.28
(TEST). Truncations at 8,192 tokens are similar for base and candidate: 9 vs 5
on RERANK, 29 vs 28 on TEST.

**Outcome.** No collapse can be measured under the official prompt, because
there is no RERANK gain to collapse from. The +8 pp advantage of the selected
candidate exists only under the direct-letter prompt it was selected with.
Under the official prompt it is −1.5 pp on the same 200 questions. The TEST
harm persists and is larger (−4.28 pp, CI excludes 0). As the spec requires:

> **the expert-mirage phenomenon is prompt-sensitive.**

The *selected advantage* does not survive a change of prompt even on the
selection set itself. That is consistent with the mirage account, since the
"expertise" is specific to the exact condition it was selected under. But it is
a limitation, and the paper must state it: we tested one alternative prompt,
on one candidate.

## 4. Evidence hierarchy (Part T)

1. **Precommitted and frozen** (git-locked before the data existed): splits;
   protocol; base outputs; SEARCH ranking; RERANK selection; the standalone TEST
   result (+8.0 → −2.67). This is the primary evidence.
2. **Pre-frozen addendum:** the SEARCH top-50 committee, frozen before its TEST
   inference. Its vote equals the base.
3. **Forensic verification:** independent re-scoring, byte-identical GPU re-run,
   no leakage, no bug. This rules out implementation explanations.
4. **Post-hoc descriptive analyses:** the nulls (H), fragile items (G), ties (J),
   decomposition (E), form overlap (F), committee diversity (I) and σ (K). They
   explain the result but were not precommitted, and are labelled so.
5. **Post-hoc single-condition diagnostics:** the mask diagnostic (one seed) and
   prompt robustness (one prompt pair). Appendix; no general claims.

## 5. Paper story

**Title:** *Expert Mirages in Weight Space: Two-Stage Selection over Random
Perturbations of a Vision-Language Model*

**Thesis:** In a random weight-perturbation population, the selected "expert"
can be an artefact of selecting the maximum over a large pool of correlated
near-copies that flip the same fragile items. A precommitted three-stage design
exposes it: the +8 pp RERANK winner loses 2.67 pp on the official test, and the
RandOpt top-50 vote collapses to the base. The winner's advantage is also
specific to the prompt it was selected under: with the official OmniSpatial
prompt it vanishes even on the selection set (−1.5 pp).

**Abstract structure:**

1. Setting: RandOpt/Neural Thickets claims that useful experts are dense near
   pretrained weights.
2. What we did: a precommitted N = 5,000 all-parameter search on Qwen3-VL-8B,
   OmniSpatial Perspective Taking, with SEARCH → RERANK → TEST and a 500-candidate
   random audit.
3. Result: +8.0 → −2.67 pp; vote = base.
4. Why: selection maximum, shared fragile items, tie instability; not leakage and
   not a bug.
5. Prompt sensitivity: under the official OmniSpatial prompt the winner's
   RERANK advantage vanishes (−1.5 pp) and TEST harm remains (−4.3 pp).
6. Scope: one model, one task, one population, one alternative prompt.

**Contributions:**

1. A precommitted three-stage protocol with a random-candidate audit that
   separates selection effects from expertise.
2. A documented expert mirage: frozen, reproduced, and forensically verified.
3. Diagnostics any perturbation-search paper can run:
   - an extreme-selection null;
   - fragile-item flip rates;
   - tie-break sensitivity;
   - an oracle-versus-churn decomposition for committees.

## 6. Figures (Part Q)

**Main text, 4 figures:**

1. `fig1_mirage_three_stage`: the phenomenon.
2. `fig6_extreme_selection_null`: why +8 is close to what selection alone
   gives.
3. `fig5_fragile_items`: the mechanism, shared flips of a few items.
4. `fig7_committee_convergence`: ensembling returns the base.

**Appendix:**

- `fig2` (transfer scatter)
- `fig3` (decomposition; its table goes in the main text)
- `fig4` (form overlap)
- `fig8` (σ)
- `fig9` (prompt robustness)

## 7. Tables

**Main text:**

- `table_main_results`
- `table_shift_decomposition`
- `table_analysis_provenance`

**Appendix:**

- `table_setup_differences`
- `table_forensic_integrity`
- `table_committee_diversity`
- `table_tie_instability`
- `table_question_form_overlap`
- `table_prompt_robustness`

## 8. Delete or move

- **Delete** the forensic report's causal sentence ("+8 occurred because …
  Egocentric … easier kind") and the main report's "the larger the validation
  gain, the worse the test" (see claim audit 7–8).
- **Move to the appendix:**
  - the mask diagnostic;
  - the Allocentric follow-up (a null; one paragraph);
  - engineering incidents (Ray leak, pod races).
- **Do not quote** the oracle 64% without the churn numbers, or any "p = 0"
  from the empirical null.

## 9. GPU cost log

Part N was the only new GPU work: 1× H100 80GB at $3.49/h (rate read from the
RunPod API). The pod ran from 16:31:34 to 17:13:27 UTC, then stopped
automatically after the results were pulled.

| Step | Cumulative $ |
|---|---:|
| Setup and model load | 0.25 |
| Integrity: direct-prompt base reproduces stored text (259, 0 mismatches) | 0.36 |
| Preflight: 64 questions; projected P1 + P2 $3.58 including setup | 0.47 |
| Priority 1: base TEST, official prompt | 0.73 |
| Priority 2: base RERANK; candidate RERANK + TEST | 1.30 |
| 4 sampled base repeats on TEST (budget permitted) | 2.44 |
| Stop, about 1 min | **≈ 2.50** |

A hard-cap guard on the pod would have stopped it at $4.85. No other GPU
work was run: no seeds, no N, no GQA, no masks, no σ, no other model.
