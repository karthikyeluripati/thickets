# Artifact Audit — Isolated 7B Causal-Density Pilot

Audited at commit `9305cc8` (branch `iclr-causal-density-pilot`, isolated worktree), before any
GPU execution. Machine-readable version: `artifact_audit.json`.

## Environment (verified, not assumed)

This session runs on a local Windows machine (`win32`/MINGW64). Direct probes:

```
$ ls /workspace          -> No such file or directory
$ nvidia-smi              -> command not found
```

The task's stated working directory (`/workspace/thickets/research/neural_thickets_repro`) does
not exist here, and no GPU is present. This is **not** the RunPod environment the task
describes executing on. Per the task's own Phase 6 contingency ("If GPU execution is blocked,
preserve the tested implementation and exact resumable command, then return `INCONCLUSIVE —
EXECUTION BLOCKED`"), this audit proceeds to build and test the pilot infrastructure, but no
GPU execution is attempted in this environment.

## Existing 3B/7B/32B perturbation outputs — none reusable

| Directory | Reusable? | Why not |
|---|---|---|
| `results/stage7b_anatomical_calibration` | No | Different scope taxonomy, no 3-condition evaluation. |
| `results/stage8_coarse_anatomical_atlas` | No | `anatomical_relative_l2` over L1 anatomy regions, not `scopes.py`'s vision_encoder/full_lm/full_vlm. Different radii. Correct-image only. |
| `results/stage9_hierarchical_anatomical_atlas` | No | Same mismatch as Stage 8, extended to depth bands. |
| `results/stage11_visual_thicket_scaling_analysis` | No | Same anatomical-region taxonomy mismatch; cross-scale, not scope-based. |
| `results/visual_thicket_global_3b_pilot` | No | 3B-only (this pilot is 7B-only); `global_gaussian_upstream` semantics, already disqualified as a whole-model anchor by this repo's own `WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE`. |

None of these use the preregistered scope taxonomy, radii, or (correct/shuffled/text-only)
paired-condition design — none pass the reuse bar (exact candidate identity, model revision,
seeds, samples, prompts, decoding, scope, radius, schema).

## Existing scoped-perturbation infrastructure — code only, never executed

- **`run_scoped_randopt.py`** exists and implements the correct scope taxonomy
  (`vision_encoder`/`full_lm`/`full_vlm`) with `relative_l2` scale mode and the exact
  perturb→evaluate→restore RPC pattern this pilot's evaluator reuses. Its own docstring states:
  *"PREPARES ONLY — not executed by us here (no GPU on this machine)... this milestone
  authorizes only diagnostics/scope_isolation_gpu_check.py, never this script's own candidate
  search."* No `results/*scoped*` directory exists anywhere in the repository, confirming it
  has never run. Even if it had, it only evaluates one capability per invocation with
  correct-image scoring only — no audit/selection split, no shuffled/text-only conditions — so
  it could never have produced reusable rows for this pilot's design regardless.
- **`scopes.py`** — the `vision_encoder`/`full_lm`/`full_vlm` scopes are already canonically
  defined (`build_scope_manifest`, `compute_relative_l2_sigma`). Reused unmodified; this pilot
  never redefines scope membership.
- **`benchmarks/image_sanity.py`** — `make_shuffled_variant`/`make_text_only_variant`/
  `run_image_sanity_check` already exist and are validated as a small-N (default 40) wiring
  check. `find results/ -iname '*image_sanity*'` returns nothing — no prior run of any kind was
  ever persisted, for any capability, at any scale. Reused unmodified by this pilot's
  `shuffle_manifest.py`/`evaluator.py`, but contributes zero reusable data.
- **`diagnostics/scope_isolation_gpu_check.py`** — the *one* GPU mechanical validation the task
  spec describes as authorized for the scoped-perturbation milestone. `find results/ -iname
  '*scope_isolation*' -o -iname '*scope_diagnosis*'` returns nothing — **this has never been
  run either.** This pilot's evaluator treats a PASS from this check as a required precondition
  before any candidate in a given scope may be evaluated (never re-diffing every out-of-scope
  parameter on all 600 candidates — see `evaluator.py`'s own docstring for why that would be
  both prohibitively expensive and redundant with the structural + one-time-mechanical
  guarantee). **This diagnostic must be run and confirmed PASS for all three scopes on real TP
  hardware before Phase 6 can begin.**

## Capability configs — all five present

`configs/benchmarks/{visual_grounding,counting,ocr_text_recognition,spatial_reasoning,relational_reasoning}.yaml`
all exist. Exact live pool sizes (needed for the 400-disjoint-example gate) require a live
dataset load — a Phase-2 (build-time) step this offline audit does not attempt.

## Base-model image sanity vs. per-perturbation causal evaluation

No artifacts of either kind exist. Distinguishing them matters for future audits: base-model
image sanity (image_sanity.py's own wiring check) answers "does the image reach the model at
all," a necessary-but-not-sufficient precondition; per-perturbation causal evaluation (this
pilot's own design) answers "does THIS candidate's real-image gain survive removing the image
entirely / feeding a wrong image" — a materially different, per-candidate question this repo
has never previously asked.

## Conclusion

**No existing artifact in this repository satisfies the reuse bar for this pilot.** Every
reusable asset is validated, unexecuted *code* (the scope taxonomy, the image-sanity
primitives, the RPC dispatch pattern, the five capability adapters). Zero reusable *result
data* exists. This pilot's infrastructure (Phases 1–4) builds a new, isolated candidate
population, subset manifest, shuffle manifest, and result schema from scratch — reusing only
the code-level primitives cataloged above, never any prior result row.
