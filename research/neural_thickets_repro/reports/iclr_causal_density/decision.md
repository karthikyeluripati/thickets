# Decision — Isolated 7B Causal-Density Pilot

## Decision

**`INCONCLUSIVE — EXECUTION BLOCKED`**

## Why

This session runs on a local Windows machine with no GPU. Direct probes at the start of this
task confirmed:

```
$ ls /workspace         -> No such file or directory
$ nvidia-smi             -> command not found
```

The task's stated working directory (`/workspace/thickets/research/neural_thickets_repro`)
does not exist here. This is not the RunPod environment the task describes executing on.
Phase 5 (base-control gate) and Phase 6 (decisive pilot) both require a live, TP-capable
`vllm`/`ray` GPU engine to evaluate the unperturbed model and all 600 perturbation candidates
across 5 capabilities × 3 visual conditions — neither can run here.

Per the task's own Phase 6 contingency (*"If GPU execution is blocked, preserve the tested
implementation and exact resumable command, then return `INCONCLUSIVE — EXECUTION BLOCKED`.
Never fabricate results."*), no result row, metric, or figure in this deliverable is
fabricated or estimated. Every metrics/analysis/decision-gate module was built and verified
against **synthetic** data specifically so the code's correctness is established independent
of, and prior to, any real GPU run.

## What was completed (Phases 0–4, 7–10 as tested infrastructure)

- **Phase 0** — audit and branch isolation. Branch `iclr-causal-density-pilot` created from
  commit `9305cc8` in an isolated git worktree; main branch/worktree untouched.
  `reports/iclr_causal_density/artifact_audit.{json,md}` document, with evidence, that **no
  existing artifact in this repository** satisfies the reuse bar for this pilot's design — only
  validated, unexecuted code (scope taxonomy, image-sanity primitives, capability adapters) is
  reusable.
- **Phase 1** — preregistration (`reports/iclr_causal_density/preregistration.md`), frozen
  before any result exists, generated from the single source of truth
  `iclr_causal_density/design.py`.
- **Phases 2–4** — subset/shuffle-manifest construction, the 600-candidate population, the
  long-form result schema, the paired-condition evaluator, and the checkpoint/resume driver are
  all implemented and unit-tested (84 new tests, all CPU-only, via injected fakes — the
  established convention throughout this repository).
- **Phases 7–10** — the paired-bootstrap metrics (`Δ^R`/`Δ^T`/`Δ^S`/`G_i`, `ρ_standard`/
  `ρ_visual`/`D` with confidence intervals), the search-budget Monte Carlo analysis, the
  standard-vs-grounded selection comparison, and the **decision gate itself** (implemented in
  code, never assigned subjectively) are all implemented and unit-tested against synthetic
  `CONFIRMED`/`REJECTED`/`INCONCLUSIVE` cases.

## What was not attempted

- Any GPU execution — no candidate was perturbed, no capability was evaluated, no base-model
  row exists.
- The real engine-launch/dataset-loading wiring layer (binding the evaluator's injected
  callables to real `launch_stage6_engine`/`resolve_model_snapshot`/`benchmarks.runner.
  run_benchmark`/`scoped_apply_perturbation` calls) — this genuinely requires a live pod to
  write and verify safely, and is the concrete next step once one is available.
- `diagnostics/scope_isolation_gpu_check.py` (the *one* GPU mechanical isolation validation
  this pilot's evaluator treats as a required precondition) has never been run in this
  repository at all — confirmed by the artifact audit. It must run and pass for all three
  scopes before Phase 6 candidates in any scope can be evaluated.
- Figures — no real data exists to plot; generating them now would mean plotting fabricated
  numbers, which this task explicitly forbids.

## Integrity summary

| | |
|---|---|
| Source commit | `9305cc8824b2c16a8f73befed3978c0cf96ff7ec` (verified present, is `HEAD` of `iclr-causal-density-pilot`) |
| Branch / worktree | `iclr-causal-density-pilot`, isolated worktree; main branch/worktree untouched |
| New tests | 91 passed, 0 failed (84 + 7 driver tests) |
| Full CPU suite baseline (before this pilot) | 2355 passed, 2 skipped, 0 failed |
| Full CPU suite (after this pilot, isolated worktree) | 2446 passed, 2 skipped, 0 failed — exactly 2355 + 91, zero regressions |
| Expected perturbations | 600 | Completed | 0 |
| Expected result rows | 9,000 (candidates) | Completed | 0 |
| Restoration / isolation / norm / provenance status | not applicable — no candidate was ever evaluated |

## Headline evidence

Not computed. Every `ρ_standard`, `ρ_visual`, `D`, confidence interval, budget-divergence
verdict, and grounded-selection verdict for all five capabilities is `null` in
`decision.json` — none fabricated, none estimated.

## Interpretation

The frozen hypothesis was **not tested** — not confirmed, not rejected. The infrastructure and
statistical/decision-gate logic needed to test it are built and verified; the GPU execution
needed to generate the data the gate operates on could not happen in this environment.

## Exact next action

The blocker is the absence of a GPU/pod in this environment — nothing else. To proceed: run
`diagnostics/scope_isolation_gpu_check.py` on real TP hardware for `vision_encoder`/`full_lm`/
`full_vlm` (confirming the one required precondition this repository has never actually
validated), then wire and run this pilot's Phase 5/6 (base-control gate, 600-candidate decisive
pilot) using the tested `iclr_causal_density` infrastructure on that hardware. No scope is
expanded, no alternative hypothesis is proposed, and 32B work is not resumed.
