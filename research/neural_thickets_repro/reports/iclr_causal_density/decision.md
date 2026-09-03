# Decision — Isolated 7B Causal-Density Pilot

**This document supersedes the earlier `INCONCLUSIVE — EXECUTION BLOCKED` draft** written when
this pilot was CPU-only infrastructure with no GPU access (see git history for that version).
GPU execution subsequently completed in full — this is the real, final decision, computed from
the actual 600-candidate result data.

## Decision

**INCONCLUSIVE**

Produced by `run_iclr_causal_density_analysis.py` (Step 12) — the frozen, unmodified
`decision_gate.evaluate_decision_gate` returned this via its own pre-existing precedence rule 2:
*"expected exactly 5 capabilities with valid results, got 4."* `visual_grounding` could not be
scored under the frozen Phase 7 formula (see `preregistration_amendment_2026-09-03.md` §1) — a
structural incompatibility between RefCOCO grounding (no meaningful text-only condition) and
`metrics.ConditionScores`'s equal-length, three-condition requirement, not a data gap fixable
with more GPU time.

**This is a formal INCONCLUSIVE, not a near-miss CONFIRMED.** Even setting the capability-count
gap aside, the underlying evidence does not support CONFIRMED: criterion 3 (search-budget
divergence in ≥4/5 capabilities) holds for only 1 of the 4 scored capabilities; criterion 4
(grounded retention ≥80%) fails in both capabilities where it is defined (47% and 62%). A
hypothetical 5th capability would have needed to single-handedly satisfy both already-failing
criteria to flip the outcome.

## What was completed

All of Phases 0–10, on real hardware, with real data:

- **Phases 0–4** (CPU-only infra) — as before: audit, preregistration, subset/shuffle manifests,
  candidate population, evaluator, checkpoint/resume driver. 91 tests.
- **Phase 5** (base-control gate) — real 7B, all 5 capabilities, both subsets, all applicable
  conditions. `BASE CONTROL GATE: PASS`.
- **Phase 6** (decisive pilot) — 600/600 candidates, audit-set pass AND selection-set pass (the
  latter added after discovering Phase 9 requires selection-set scores the first pass never
  collected — see `preregistration_amendment_2026-09-03.md`), 0 failures in final form. Six
  transient infrastructure failures along the way (image-token overflow, encoder-cache-reset
  wiring, RayEngineLLMAdapter wiring, norm-verification field bug, resume-duplication bug, and a
  GPU memory-fragmentation OOM) were each diagnosed from real tracebacks, fixed with a regression
  test, and retried successfully — see commit history on `iclr-causal-density-pilot`. None
  involved changing a preregistered scientific parameter.
- **Phase 7–10** — real, final computation against the real data. See Evidence below.

## Integrity summary

| | |
|---|---|
| Source commit (model) | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Branch / worktree | `iclr-causal-density-pilot` only; `main` never touched |
| Expected perturbations | 600 | Completed | 600 (both passes) |
| Expected result rows | 1,680,000 per pass (600 × 2800; visual_grounding's missing text_only means 14, not 15, (capability,condition) pairs per candidate) | Completed | 1,680,000 / 1,680,000 (both passes) |
| Restoration / isolation / norm / provenance | 100% verified `True` on every row, both passes (independently validated, not merely asserted — see `artifact_provenance.md`) |
| Duplicate rows | 0 (both passes) |
| Candidate-ID identity across passes | Byte-identical 600-ID sets, cross-checked directly |

Full checksum manifest and reproduction instructions: `artifact_provenance.md`.
Full per-capability, per-cell numeric detail: `decision.json`, `analysis_full_output.json`.

## Headline evidence

| Capability | ρ_standard | ρ_visual | D (point) | D 95% CI | Search-budget divergence | Grounded G improved (top-10) | Grounded retention (top-10) |
|---|---|---|---|---|---|---|---|
| counting | 0.152 | 0.005 | 30.3 | [1.01, 1.59] | False (0/6 cells) | True | undefined |
| ocr_text_recognition | 0.453 | 0.005 | 90.7 | [1.01, 1.24] | False (0/6 cells) | True | undefined |
| spatial_reasoning | 0.290 | 0.010 | 29.0 | [1.01, 1.61] | False (1/6 cells) | True | 46.7% (fails 80%) |
| relational_reasoning | 0.538 | 0.050 | 10.8 | [1.01, 1.52] | **True** (4/6 cells) | True | 61.7% (fails 80%) |
| visual_grounding | — | — | — | — | — | — | Excluded (see amendment §1) |

**On the D point-estimate vs. CI divergence** (audited, not a bug): `design.
PREREGISTERED_BOOTSTRAP_METHOD_NOTE` specifies the population-`D` CI uses a laxer per-resample
classification than the strict, individually-bootstrapped classification the point estimate
uses — explicitly to avoid a computationally infeasible nested bootstrap. Full detail:
`preregistration_amendment_2026-09-03.md` §3.

## Interpretation

The frozen hypothesis was tested in full and is **not confirmed** by this pilot. Formally
INCONCLUSIVE (visual_grounding unclassifiable under the frozen formula); substantively, the
evidence available from the four scored capabilities does not support CONFIRMED either —
search-budget divergence (the signature the hypothesis predicts) holds in only 1 of 4
capabilities, and grounded selection's retention of standard selection's real-image gain falls
well short of the frozen 80% bar in both capabilities where it is defined.

## Recommendation

On this evidence, 3B–72B scaling should **not** be authorized for this specific claim. This
pilot preserves the evidence exactly as produced; the paper-direction decision is reserved for a
separate discussion, not made or implied here.

## Exact next action (none required by this pilot)

None. This pilot is complete: infrastructure was built, GPU execution ran to completion twice
(audit-set and selection-set passes), completeness was independently validated, the frozen
analysis ran against real data, and the decision above is final. Any further action (a
follow-up experiment, a different capability set, an amended hypothesis) is a new, separate
decision outside this pilot's scope.
