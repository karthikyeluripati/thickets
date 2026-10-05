# Claim audit: "Expert Mirages in Weight Space"

**Scope.** No LaTeX draft of the paper exists in this repository or on this
machine, so this audit covers the claims in the four result reports the paper
will draw on, checked against the frozen artifacts and the new paper analyses
(`paper/figures/data/*`). The reports are:

- `PERSPECTIVE_TAKING_VISUAL_NEURAL_THICKETS_N5000` (main)
- `PERSPECTIVE_TAKING_RANDOPT_TOP50_REEVALUATION` (committee)
- `PERSPECTIVE_TAKING_N5000_FORENSIC_AUDIT` (forensic)
- `EXISTING_N5000_ALLOCENTRIC_SPECIALIST_SEARCH` (allocentric)

The same verdicts apply to any draft written from them.

**Terminology for the paper.** The stages are SEARCH (SEARCH200), RERANK and
TEST. RERANK is the historical "VALIDATION200": a *second-stage selection* set,
not an independent validation, because the reported +8 pp is the maximum over
the 50 search-selected candidates. TEST is the official test, 561 questions.
Historical file names are unchanged.

Verdicts:

- **OK** — supported as written.
- **SCOPE** — supported only with the stated scope or qualifier.
- **REWORD** — the evidence supports a weaker statement.
- **CONTRADICTED** — the new analyses disagree with it.

## 1. Headline numbers

| # | Claim (source) | Evidence | Verdict | Paper wording |
|---|---|---|---|---|
| 1 | 9504111 gains +8.0 pp on "validation", 95/200, paired 95% CI [+3.5, +13.0] (main) | Reproduced by independent re-scoring and byte-identical GPU re-run; stratified bootstrap gives [+3.5, +12.5] | **REWORD** | "+8.0 pp on RERANK, the second-stage selection set (max over the SEARCH top 50)". Never call it validation accuracy or held-out. |
| 2 | Same candidate −2.67 pp on TEST, 244 vs 259/561 (main) | Reproduced exactly | **OK** | — |
| 3 | SEARCH-top-50 majority vote = base on TEST, 259/561, CI [−1.25, +1.25]; no prefix K ∈ {1,5,10,25,50} exceeds base (committee) | `table_committee_diversity.json`; fig7 | **OK** | — |
| 4 | "Genuine expert found? No" (main) | No top-10 member passes +5 pp/bootstrap/two-subtask checks; best of 50 on TEST +1.25 pp | **SCOPE** | "No transferable ≥ +5 pp standalone candidate was found in this population." |
| 5 | Density audit: of 125 random σ = 0.002 candidates, 60 / 23 / 7 reach ≥ +1 / +3 / +5 pp on RERANK (main) | `fig8_sigma_tradeoff.csv` identical | **OK** | — |
| 6 | σ = 0.002 population averages −4 correct on SEARCH200 (main) | mean −1.98 pp = −3.97 questions | **OK** | — |

## 2. Statements that must change

| # | Claim (source) | Evidence | Verdict | Paper wording |
|---|---|---|---|---|
| 7 | "The larger the validation gain, the worse the test result tended to be" (main, from the top 10) | Across all 50 SEARCH-selected: RERANK vs TEST Pearson −0.27 (p = 0.059), Spearman −0.18 (p = 0.21). SEARCH vs TEST −0.35 (p = 0.012), but σ confounds it. | **REWORD** | "RERANK gain did not predict TEST gain (r = −0.27, p = 0.06, n = 50)." Do not claim an anti-correlation. |
| 8 | "The +8 pp occurred because σ = 0.002 tips borderline, templated Egocentric items… the test's Egocentric questions are a different, easier kind… costs 14 net Allocentric answers" (forensic summary) | **Part E:** with RERANK per-subtask behaviour reweighted to the TEST subtask mix, the predicted TEST gain is **+3.88 pp** (bootstrap [−1.4, +9.4]), not a loss. The mixture change explains −4.12 pp of the 10.67-pp drop; the **within-subtask change explains −6.56 pp**, the larger part. Per subtask, Egocentric goes from +11.8 pp (RERANK) to −4.9 pp (TEST) and Allocentric from −1.5 to −3.7. **Part F:** RERANK gain is +11.5 pp on questions whose exact text appeared in SEARCH and **+14.6 pp on unseen question forms**, so the template-overlap explanation is not supported. | **CONTRADICTED (in part)** | "The subtask mix shift explains under half of the collapse; most of it is a change of behaviour within the same subtasks. The RERANK gain is not explained by question-form overlap with SEARCH." The fragile-item finding (claim 9) stays. |
| 9 | "Some single questions are flipped by ~80% of random σ = 0.002 candidates" (forensic) | Max wrong→correct rates 0.800, 0.784 (fig5). 9504111's wins sit on items random candidates fix 32% of the time, against 4.4% for other base-wrong items. | **OK** | Keep; this is the strongest mechanism evidence. |
| 10 | "Seed 9504111 is the maximum of that shared shift plus candidate-specific luck" (forensic) | Part H: expected max of 50 random σ-mix-matched candidates is +5.8 pp (95% range +3.5 to +7.5); the observed +8.0 is above all 500 audit candidates (audit max +7.5), i.e. about one question beyond the empirical range. Part J: entered top 50 via a 33-way tie at the SEARCH cut-off; 50% of the top 50 was decided by tie-break. | **REWORD** | Give the numbers. "Luck" is fair only as "about +2 pp beyond the σ-matched selection expectation". The empirical null cannot exceed the audit max, so do not report "p = 0". |
| 11 | Mask diagnostic: language-only gives most of the RERANK gain; the combination is "super-additive" and "causes most of the test loss" (forensic Phase R) | One seed: language-only 91/255, vision-only 82/259, all 95/244 | **SCOPE** | Appendix only, labelled "single-seed post-hoc diagnostic". No causal or general claim. |
| 12 | "The observed right tail is fully explained by candidates independently tipping the same few fragile questions" (allocentric) | Structured null matches max and ≥ +5/+7 counts, but ≥ +3 pp is in excess (471 vs 435, p = 0.03) | **REWORD** | "The extreme tail matches the structured null; a small excess at +3 pp (two questions) remains." |
| 13 | "The Neural-Thickets effect is absent under both the standalone and the top-K ensemble standard" (committee) | Correct inside its stated scope | **SCOPE** | Keep the scope sentence (Qwen3-VL-8B, OmniSpatial Perspective Taking, all-parameter N = 5,000). Never "Neural Thickets does not hold" or "RandOpt fails". |

## 3. Claims the new analyses add

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 14 | The top-50 oracle (any member right) reaches 64.3% on TEST | Oracle 361/561 | **SCOPE.** Report it together with symmetric churn: 102 base-wrong questions are fixed by ≥ 1 member, while 114 base-right ones are broken by ≥ 1 member (38 vs 37 at ≥ 10 members). The majority fixes 7 and breaks 7. It is **not** evidence of complementary expertise, and the abstract must not quote the oracle alone. |
| 15 | The committee converges to the base | Base answer is the majority on 96.6% of questions; questions where the majority differs from base fall from 60 (K = 1) to 19 (K = 50) | **OK** |
| 16 | Members are near-copies | Mean pairwise prediction disagreement 12.2%; mean error Jaccard 0.84 | **OK** |
| 17 | Base accuracy matches published OmniSpatial numbers | Not true: our base is +5.9 pp (micro) above the only published Qwen3-family manual-CoT row (EASI) on identical items; the checkpoint label there is ambiguous | **SCOPE.** State it as a limitation (`baseline_reconciliation.md`). The final status depends on Part N. |

## 4. Never claim

- That Neural Thickets or RandOpt is false or refuted in general; that this
  replicates or fails to replicate the GQA result. Our setup differs (VLM,
  all-parameter mask, multiple-choice direct letter, two-stage selection).
- That 9504111 or any member is a perspective-taking "expert" or "specialist".
- That the collapse is "caused" by subtask mix alone, or by template overlap.
- Anything about other models, other N, other σ, or other benchmarks.
- Any p-value from the empirical random-candidate null beyond "exceeds all 500
  audit candidates".
- That the paper analyses were precommitted. Only the stages up to the
  standalone TEST result were frozen (`table_analysis_provenance.tex`).

## 5. Allowed primary claim

> In a precommitted, 5,000-candidate random weight-perturbation search on
> Qwen3-VL-8B (OmniSpatial Perspective Taking), the best candidate after
> two-stage selection appeared +8.0 pp better than the base model on the
> selection set and was −2.67 pp worse on the official test. Its RERANK gain is
> about what selecting from a pool of correlated near-copies flipping the same
> fragile items would produce. A RandOpt-style top-50 majority vote returns
> exactly the base accuracy.

## Status of this audit

**Every quantitative claim in the reports reproduces. Four statements need
rewording (claims 1, 7, 10, 12). One causal explanation (claim 8) is partly
contradicted by the new analyses and must be replaced.** Without these fixes the
paper would not meet its own standard.
