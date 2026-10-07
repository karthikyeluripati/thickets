# Correction: selection and test splits are different question formats

> **Note (2026-10-07 cleanup):** earlier study write-ups referenced below were removed from the working tree because
> they are superseded or outside the paper's story. They remain available at git tag `pre-cleanup`
> (`git show pre-cleanup:paper/<FILE>.md`). Their raw data, where the paper still uses it, is kept under `results/`.

- **Date:** 2026-10-06. Found post hoc in `WHY_INVESTIGATION_9504111.md`.
- **This note supersedes** the affected statements in every report listed below. Those reports are not edited, so
  their history is preserved; read them together with this note.

## The fact
SEARCH200 and RERANK200 were drawn from **OmniSpatial-train** Perspective_Taking. TEST561 is the **official
OmniSpatial-test** split.

| | SEARCH / RERANK | TEST |
|---|---|---|
| 8-way compass options ("front-left", "back", …) | 185 / 189 of 200 | **2 of 561** |
| left / right / "can not determine" | 15 / 11 | 266 |
| counting / other formats | 0 | 159 / 134 |
| gold answer contains "front" | 94 / 92 | 16 |
| subtasks (Ego / Allo / Hypo) | 119 / 65 / 16 | 102 / 376 / 83 |

The winner's entire RERANK gain is on compass items (19 repairs, 3 regressions out of 189). **TEST cannot test the
behaviour that was selected.** Any RERANK→TEST or SEARCH→TEST comparison is therefore a cross-format comparison, not
a generalization test.

Selecting on a train-derived set and reporting on the official test split is standard practice. The miss was not
auditing whether the two have the same composition.

## Status of earlier conclusions
**INVALID** means the claim must not be used as evidence. **REINTERPRET** means the measurement stands but its meaning
changes. **UNAFFECTED** means it never relied on the cross-format comparison.

| Report | Statement | Status | Note |
|---|---|---|---|
| `paper_analysis_summary.md` §5; `paper_claim_audit.md` §5 | "Expert mirage": +8.0 pp selected → −2.67 pp on the official test, read as failed transfer | **INVALID as a transfer claim** | The numbers are correct. "The selected advantage does not generalize" is not shown by them. |
| `paper_claim_audit.md` claim 7 | Larger RERANK gain goes with worse TEST | **INVALID** | Cross-format correlation |
| `paper_analysis_summary.md` Part E | Subtask mix explains under half of the collapse | **INVALID (superseded)** | The format mismatch is the larger difference; "collapse" is not a transfer quantity |
| `paper_analysis_summary.md` Part F | Unseen question forms gain as much as seen ones | UNAFFECTED | Within RERANK |
| `paper_analysis_summary.md` Parts G, H, J (fragile items, extreme-selection null, selection instability) | | UNAFFECTED | SEARCH/RERANK only |
| `paper_analysis_summary.md` Part I; `BEHAVIORAL_DIVERSITY_AUDIT.md` Part 8 | Top-50 vote = base on TEST | **REINTERPRET** | True on TEST formats; says nothing about the compass format the candidates were selected on |
| `paper_analysis_summary.md` Part N (official prompt) | Under the official prompt the RERANK advantage vanishes (−1.5 pp) | UNAFFECTED | Within RERANK; consistent with an answer-prior shift that depends on the prompt |
| `MARGIN_AND_ADDITIVITY_FOLLOWUP.md` Question A | Score shifts differ between RERANK and TEST after opportunity matching | **REINTERPRET** | The "unassigned residual" has a known candidate: different formats |
| `MARGIN_AND_ADDITIVITY_FOLLOWUP.md` Question B | Group insertions are additive in score space | UNAFFECTED | |
| `RANDOM_CONTROL_TRANSFER_9504111.md` | A shared phase response (10/12 controls RERANK > TEST); winner −9 pp beyond it | **REINTERPRET** | The "shared response" is consistent with a set/format difference that any perturbation exposes, not transfer failure |
| `SELECTION_VS_SHARED_RESPONSE_9504111.md` Analysis B | Group-wide RERANK-vs-TEST accounting | **REINTERPRET** | Cross-format |
| `SELECTION_VS_SHARED_RESPONSE_9504111.md` Analysis C (RERANK-only reselection) | | UNAFFECTED | |
| `SELECTION_VS_SPECIFICITY_9504111.md` A/B (held-out RERANK) | | UNAFFECTED | |
| `SELECTION_VS_SPECIFICITY_9504111.md` C/E | The winner's TEST deficit is candidate-specific (worst of 61) | **REINTERPRET** | Fact stands; it describes damage on other formats, with no identified cause |
| `MECHANISM_RESEARCH_PLAN_9504111.md` H★ | Mirage predicted by cos(F_RERANK, F_TEST) | **MOOT** | H★ was falsified anyway; the cross-set geometry compared different formats |
| GPU-A (`geometry-gpu-a`) F1/S2d | First-order prediction falsified | UNAFFECTED | Per-example validity, not transfer |
| GPU-A F3-TEST and TEST cosines | | **MOOT** | Cross-format |
| `CAUSAL_DIAGNOSTIC_9504111.md` | Group insertions and removals; additivity; no single owner | UNAFFECTED | Phase-difference remarks: REINTERPRET |
| `VISION_NONLINEARITY_9504111.md` | Vision nonlinear, not the why | UNAFFECTED | |
| `TRANSFER_DENSITY_AUDIT.md` | Weak transfer across SEARCH halves and the 400-question train pool | UNAFFECTED, and now the **valid** transfer evidence | Within format |
| `RADIUS_AUDIT.md` | Holdout flat then steps; abandon the radius framing | UNAFFECTED | Train-pool holdout |
| `BEHAVIORAL_DIVERSITY_AUDIT.md` Parts 2–7, 9 | | UNAFFECTED | SEARCH/RERANK |
| `baseline_reconciliation.md` | | UNAFFECTED | |
| `WHY_INVESTIGATION_9504111.md`, Stage 1+2, Stage 3, R1, R2/R2b | Answer-prior shift; predictable from noise; mid-layer | UNAFFECTED | Train-split items only (3A's TEST correlation was reported, not gating) |

## Consequences for the paper
- **Do not** frame the paper as "selected experts fail to transfer to the test set".
- **Valid transfer evidence is within format only.** That is `TRANSFER_DENSITY_AUDIT.md` (weak transfer across
  train-pool halves) and the RERANK split-half and crossfit analyses.
- **The open question the project actually needs answered** is whether the selected advantage holds on *fresh items
  of the same format*. That is a matched-split test, not yet run.
- **Disclose the mismatch as a finding:** split-format mismatch can manufacture apparent non-transfer.
