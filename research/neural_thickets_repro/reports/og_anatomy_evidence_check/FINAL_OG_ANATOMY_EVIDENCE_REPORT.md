# Final Report — Paper-Viability Screening: "Where Do Visual Experts Live?"

**This is a read-only screening analysis over already-collected 3B/7B artifacts at commit
`9305cc8`. No GPU inference, no new perturbation, no new prediction, no 32B S2 execution
occurred at any point in this task.**

**Mid-task correction (disclosed, not buried):** the first pass of this analysis checked
only git-committed content and reported Phases 4/5 as NOT_MEASURABLE. While diagnosing an
unrelated full-test-suite failure, real per-candidate stage8 3B data (`results.jsonl`) was
found to exist locally on this machine — gitignored, never committed (this project's own
established convention: raw per-example results stay local, only aggregates are
versioned), but genuinely present in the sibling main worktree. It was restored
(checksum-verified, plain local file copy, no git operation, no branch touched) into this
isolated worktree, and Phases 3–5 were recomputed against it. The verdicts below are the
corrected, final ones.

## Decision

**STOP_OR_REFRAME**

## One-sentence reason

None of the three testable evidence blocks clears its own frozen bar: anatomy is not
stable across 3B→7B, the observed cross-capability transfer structure does not exceed a
label-permutation null, and anatomy-guided search beats fair random-region search in only
2 of 6 capabilities (need ≥3) — all now measured against real per-candidate 3B data, not
inferred.

## Four-block verdict table

| Evidence block | Verdict |
|---|---|
| Stable anatomy | **FAIL** |
| Structured transfer | **FAIL** |
| Guided-search value | **FAIL** |
| 7B–32B consistency | **NOT_MEASURABLE** |

## Exact existing evidence

**Stable anatomy** — 3B coarse atlas (all 6 capabilities × 3 regions × 3 radii, N=64/cell)
crossed against 3B-vs-7B interim anatomy data: only 1 of 6 capabilities
(`spatial_reasoning`) has a 3B region preference that is both stable across radii and
statistically significant (BH q<0.05) — and it does not reproduce at 7B (flips from
`language` to `vision`). 0 of 6 clear the ≥3-capability bar. The existing prior-work
summary independently corroborates this: 16/18 capability×radius cells are
`diffuse_no_clear_preference` at both scales.

**Structured transfer** — computed from the real stage8 3B `results.jsonl` (3,456 rows,
576 perturbation directions each scored on all 6 capabilities by design). For every
(source, target) capability pair, `transfer_effect` = mean target-capability delta among
the top-10 source-capability experts, minus the cell's own baseline mean, averaged over
the 9 region×radius cells. Result: 1 of 30 pairs shows |effect| > 0.005 (the strongest,
`counting`→`spatial_reasoning`, is +0.0063); split-half reproducibility (even vs. odd
direction index) is weak (Pearson r = 0.27, below the 0.3 bar); and the observed
cross-pair variance (3.5×10⁻⁶) is *smaller* than the 95th percentile of a
label-permutation null (4.4×10⁻⁵) — i.e., real capability identity does not explain more
structure than randomly relabeling which capability plays "source."

**Guided-search value** — also computed from the real stage8 3B data, with a
circularity-safe protocol: each region's 64 directions split 50/50 by direction-index
parity (even=train, odd=held-out, fixed before any region is selected). The
train-fold-preferred region, evaluated only on its held-out half at budget k=10, beats a
whole-model random-region search (pooled held-out directions from all 3 regions) in a
majority of radii for only 2 of 6 capabilities (`spatial_reasoning`, `relational_reasoning`)
— below the ≥3 threshold.

**Weaker, already-supported claim** — Stage 8/9's own claim gates (C1 "nearby visual
specialists exist", C2a "expert density/strength depends on coarse anatomy") remain
`strongly_supported` at 3B. This is a claim about density depending on anatomy, not about
capabilities having stable, transferable, searchable locations — the weaker claim survives;
the paper's actual fixed title and question do not.

**32B** — zero committed or locally-present result rows anywhere (confirmed twice: once
via `git ls-tree`, again while auditing every locally-gitignored `results/` directory on
this machine during the mid-task correction — `results/stage11_whole_model_scaling/` has
3B and 7B subdirectories only, no 32B).

## Exact missing evidence

1. Any raw per-candidate 7B or 32B data (confirmed absent, committed or local, anywhere
   on this machine) — without it, transfer/guided-search/CI-bootstrap cannot be extended
   past 3B, and cross-scale consistency cannot be checked against 32B at all.
2. Depth-resolved (Stage 9) raw data exists locally but was out of scope for this task's
   coarse-region (Stage 8) protocol — not used, not needed given the negative 3B result.

## Claims currently supported

- Coarse anatomical region structures thicket *density/magnitude* in a capability-dependent
  way, at 3B (Stage 8/9's own claim gates).

## Claims currently unsupported

- "Different visual capabilities have **stable** locations" — fails at 3B→7B.
- "Those locations predict transfer" — tested directly on real data; fails (no
  above-null, reproducible transfer structure).
- "Those locations guide expert search" — tested directly on real data; fails (guided
  search beats fair random search in only 2/6 capabilities).

## Is 32B S2 scientifically justified?

**No.** All three testable legs of the paper's claim fail at 3B/7B, where real evidence
exists. 32B would extend an unestablished claim to a third, far more expensive scale.

## Is any additional GPU spending justified?

**No**, not for this fixed title/question as specified.

## Limitations and comparability concerns

- Transfer and guided-search are 3B-only (Stage 8's coarse atlas) — no raw 7B/32B
  per-candidate data exists to extend either analysis.
- The CI reported per cell in `anatomy_results.json` is a **true nonparametric percentile
  bootstrap** (10,000 resamples) wherever raw per-direction data is available (all of
  stage8 3B, after the mid-task correction); it falls back to a parametric approximation
  only where raw data genuinely isn't available. Phase 3's own PASS/FAIL logic is decided
  by `density_ge_0.02` region agreement and the pre-existing BH-corrected contrasts, not
  by this per-cell CI directly.
- `stage7b_anatomical_calibration`'s baselines do not match stage11's 7B baselines for
  the same capability names — flagged, not resolved.
- Stage 9's depth-resolved analysis independently found depth conditioning did **not**
  sharpen localization — consistent with, not contradicted by, this report.
- The top-10-experts / 9-cell-average / label-permutation-null (Phase 4) and
  k=10-budget / even-odd-fold (Phase 5) protocol choices were frozen in
  `analysis_specification.json` before being run, and never re-tuned after seeing the
  resulting verdict.

## Recommendation

On the existing evidence — now including a real, directly-tested transfer and
guided-search analysis rather than an absence-of-data placeholder — the fixed title and
its three-step intellectual chain (anatomy → transfer → guided search) are not supported
at any of the three steps this data can test. This report does not propose an
alternative direction; that decision is reserved for a separate discussion.

---

## Reproduction

```
cd research/neural_thickets_repro
PYTHONPATH=src python -m neural_thickets_repro.run_og_anatomy_evidence_check \
    --output-dir reports/og_anatomy_evidence_check
```

Phases 4/5 require the gitignored raw `results/stage8_coarse_anatomical_atlas/.../results.jsonl`
to be present locally (restore from a backup/pod copy); if absent, they degrade
gracefully to NOT_MEASURABLE rather than erroring. All other phases read only committed
CSV/Markdown files. Deterministic given the same local artifacts — verified by
`tests/test_run_og_anatomy_evidence_check.py::test_compute_anatomy_results_against_real_committed_data_is_deterministic`
and `test_bootstrap_ci_mean_is_deterministic_given_seed`.
