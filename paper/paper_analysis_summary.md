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

*Pending Part N; see the end of this file.*

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
- **Baseline level (Part M).** Our direct-letter base is **+5.9 pp above** the
  only published Qwen3-family OmniSpatial Perspective-Taking number (EASI
  Table 13, manual-CoT) on the identical 561 items. That row is labelled
  "Qwen3-8B-Instruct" and doesn't give its checkpoint, decoding or repeat count.
  See Part N.

## 3. Part N: official-prompt robustness

*Pending GPU run.*

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
RandOpt top-50 vote collapses to the base.

**Abstract structure:**

1. Setting: RandOpt/Neural Thickets claims that useful experts are dense near
   pretrained weights.
2. What we did: a precommitted N = 5,000 all-parameter search on Qwen3-VL-8B,
   OmniSpatial Perspective Taking, with SEARCH → RERANK → TEST and a 500-candidate
   random audit.
3. Result: +8.0 → −2.67 pp; vote = base.
4. Why: selection maximum, shared fragile items, tie instability; not leakage and
   not a bug.
5. Scope: one model, one task, one population.

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

*Pending Part N.*
