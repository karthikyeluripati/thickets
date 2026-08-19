# Scoped RandOpt design — component-localization infrastructure (first WACV milestone)

Estimates, per task `t` and model component `m`, `rho_{m,t} = P[S(theta+Delta_m) > S(theta)]`
by perturbing only a selected subset of Qwen2.5-VL-3B-Instruct's parameters instead of the
whole LM, with magnitude made comparable across components of very different size via
relative-L2 scaling, always restoring to the exact stored pretrained base. This document is
the durable record of the investigation this design is based on — not just a chat response.

## Phase 1 investigation — what was actually checked, not assumed

### HF checkpoint metadata (via WebFetch, no weights downloaded)

`Qwen/Qwen2.5-VL-3B-Instruct` config.json + `model.safetensors.index.json`, pinned revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`:
- Vision: 1 patch embed, 32 transformer blocks (`depth: 32`), 1 merger.
- LM: 36 decoder layers (`num_hidden_layers: 36`).
- Flat safetensors keys: `visual.patch_embed.*` / `visual.blocks.{0-31}.*` / `visual.merger.*`
  / `model.embed_tokens.*` / `model.layers.{0-35}.*` / `model.norm.*`. No separate `lm_head.*`
  tensor saved (tied embeddings).

**This describes what's saved to disk, not necessarily what a loaded/wrapped runtime module's
`named_parameters()` yields.**

### Runtime namespace correction (blocking, from review of prior GPU logs)

The actual vLLM-loaded runtime module nests the LM under `language_model.model.*`, not the
checkpoint's flat `model.*`, and exposes a *separate* `language_model.lm_head.*` entry even
though the checkpoint ties the weight:
```
visual.*
language_model.model.embed_tokens.*
language_model.model.layers.{0-35}.*
language_model.model.norm.*
language_model.lm_head.*
```
`visual.*` itself is **not** nested under `language_model.` in either form — consistent with
upstream `_should_perturb` already anticipating two visual prefixes (`("visual.",
"model.visual.")`).

Since scope selection operates on `worker_self.model_runner.model.named_parameters()`, the
runtime names are the only namespace that matters. `scopes.py` therefore *discovers* which
convention is present at call time (`detect_lm_namespace_convention`) rather than assuming
either the checkpoint's flat form or the observed runtime form — both are explicitly
recognized (`LM_NAMESPACE_CONVENTIONS`), and detection hard-fails if zero or more than one
convention matches, or if the discovered layer indices aren't a complete contiguous range.

### WorkerExtension investigation (fetched `utils/worker_extn.py` + `core/engine.py`,
pinned commit `536df0a308f3990b6270c991fbb96bd0b779a58e`, described here, not copied)

- `_should_perturb`: `_VISUAL_PREFIXES = ("visual.", "model.visual.")`, skipped by default,
  perturbed only if env var `PERTURB_VISUAL=1` — matches `REPRO_SPEC.md`'s existing row.
- `store_base_weights()`: clones **all** current parameters unconditionally (vision + merger
  + LM), called automatically by `core/engine.py:launch_engines()` right after each engine
  starts. A full, unscoped exact-base snapshot already exists the moment `launch_engines()`
  runs — nothing extra needed to obtain one.
- `apply_perturbation(seed, sigma)`: restores from that stored base, *then* applies fresh
  `_should_perturb`-gated noise. `reset_to_base_weights()`: copies the entire stored base back
  unconditionally.
- **Neither method, nor any other WorkerExtension method, takes a scope/component argument.**
  The upstream mechanism can only express "LM-only" (default) or "everything"
  (`PERTURB_VISUAL=1`) — never vision-encoder-only, merger-only, or per-third LM.

**Conclusion**: scoped perturbation is a local, package-side extension
(`scoped_perturbation.py`), never a modification to `external/RandOpt`, per the reproduction-
integrity rule established throughout this project. Our own perturb function calls
`worker_self.reset_to_base_weights()` — the real, unmodified upstream method, invoked as an
ordinary Python method call on the same worker object — to defensively land on the exact
stored base (mirroring `apply_perturbation`'s own restore-then-perturb discipline), then
applies noise to only the scope-selected tensors. Restoration after evaluation reuses the
existing, unmodified, string-dispatched `reset_to_base_weights` call — exactly what
`fixed_base` already does today. `PERTURB_VISUAL` is never touched by the scoped path.

## Scope definitions (from real runtime parameter names, not hardcoded)

| Scope | Selection rule | Excludes |
|---|---|---|
| `full_lm` | `not name.startswith(visual prefixes)` (identical to upstream `_should_perturb`'s own semantics — never an enumerated union of embed/layers/norm, which would need to know the LM naming convention at all) | all `visual.*` |
| `vision_encoder` | `visual.*` minus `visual.merger.*` (patch embed + blocks) | merger, all LM |
| `vision_merger` | `visual.merger.*` only | encoder, all LM |
| `lm_early` / `lm_middle` / `lm_late` | decoder layers partitioned into three contiguous, equal thirds by **discovered** layer count (36 real layers → 0–11 / 12–23 / 24–35) | vision, embeddings/norm, other thirds |
| `full_vlm` | everything | nothing |

Every scope hard-fails (raises `ScopeSelectionError`) rather than falling back to full-model
perturbation if it selects zero parameters, and scope-specific exclusion assertions run after
selection (e.g. `vision_encoder`'s selection is checked to contain zero `merger`/LM names).

## Tied-parameter / storage-dedup handling

A tied `nn.Parameter` (e.g. a `lm_head` sharing storage with `embed_tokens`) can be yielded
under two different names by `named_parameters(remove_duplicate=False)`, but PyTorch's own
`named_parameters()` (default `remove_duplicate=True`) already deduplicates shared storage —
so in ordinary use this is close to a non-issue. `build_scope_manifest` still deduplicates by
`tensor.data_ptr()` itself as a defensive guarantee, regardless of what it's handed: the
manifest that drives noise application and the relative-L2 formula can never contain the same
underlying storage twice, and `d_m`/`||theta_m||_2` are computed over that same deduplicated
set. An optional, diagnostic-only `named_parameters(remove_duplicate=False)` pass (attempted,
never required — not available on every torch/transformers version) surfaces any discovered
alias name in the manifest's `aliases` field for visibility, never used to build the perturbed
set itself.

## Relative-L2 scaling

For scope `m` with `d_m` = deduplicated selected element count and `||theta_m||_2` = L2 norm
of the deduplicated selected base weights, requested relative level `r`:

```
sigma_m = r * ||theta_m||_2 / sqrt(d_m)
```

so `E[||Delta_m||_2^2] = d_m * sigma_m^2 = r^2 * ||theta_m||_2^2`, i.e.
`E[||Delta_m||_2] ~= r * ||theta_m||_2` in expectation.

**Noise semantics are unchanged from upstream — this was a hard requirement, not a design
choice left open.** Each selected tensor gets a **fresh `torch.Generator().manual_seed(seed)`
reseeded from the same candidate seed** (the exact per-tensor-reseed convention
`perturb_cpu._generate_noise`/upstream `perturb_self_weights` already use), not a continuous
RNG stream across the concatenated scope, and not an independent/isotropic-across-tensors
Gaussian mode. The relative-L2 formula above stays valid under this scheme because every
selected coordinate still has variance `sigma_m^2` individually — but the resulting joint
perturbation across the concatenated scope is **not** claimed to be strictly isotropic, since
reusing one seed per tensor means same-shaped tensors receive identical noise patterns rather
than independent draws. This is recorded explicitly as `noise_semantics =
"upstream_per_tensor_reseed"` in every scoped candidate's ledger record and run metadata, so
it is never ambiguous later. The actual sampled `||Delta_m||_2` is also recorded per candidate
(`actual_perturbation_l2`), letting the requested relative-L2 normalization be verified
empirically against what was actually sampled.

## Candidate sampling: raw_sigma vs relative_l2

- `raw_sigma`: candidates come from `run_randopt_image_aware.py`'s own `sample_candidates(N,
  sigma_values, global_seed)` (imported, read-only reuse — that file is not modified in any
  way, not even additively). `(seed, sigma)` pairs, `--sigma-candidate` required.
- `relative_l2`: candidates vary by **seed only** — `candidate_sampling.sample_candidate_seeds
  (N, global_seed)` (an independent implementation reproducing exactly `sample_candidates`'s
  own seed draw: `np.random.default_rng(seed).choice(2**31, size=n, replace=False)`, proven
  equivalent by test across several `(N, seed)` combinations, never by editing or importing
  the seed-drawing logic itself from `run_randopt_image_aware.py`) paired with the one **fixed**
  requested `r` from `--relative-l2`. No sigma_candidate value is ever drawn or used in this
  mode — `--sigma-candidate` is rejected outright, not silently ignored. The per-scope derived
  sigma is a run-level constant (the scope manifest doesn't depend on candidate seed), computed
  once per candidate from that candidate's own perturbation call and recorded on its ledger
  entry — never presented as "the candidate sigma" when it's actually the unused
  sigma_candidate value, because no such value is ever drawn to begin with.

## Restoration mode: fixed_base only

Scoped scientific experiments require `--restoration-mode fixed_base`
(`scoped_apply_perturbation` → evaluate → `reset_to_base_weights`, exact per-candidate restore
to the stored base). `released_compat` may drift across repeated perturbation cycles
(`REPRO_SPEC.md` "Gate 2 restoration semantics" — max abs parameter drift 3.125e-02 across 10
repeated same-seed cycles in the A/B diagnostic) — exactly what scope-isolation science cannot
tolerate. `run_scoped_randopt.py` accepts `--restoration-mode` at the CLI level (so rejecting
`released_compat` is an explicit, readable message) but hard-fails immediately if anything
other than `fixed_base` is passed.

## What this milestone does NOT do

No N=20/50/5000 candidate search, no all-seven-scope sweep, no full GQA test evaluation — the
only GPU execution this milestone authorizes is
`diagnostics/scope_isolation_gpu_check.py`'s Test A (`vision_encoder`) / Test B (`lm_middle`)
mechanical isolation check (extended to Tests C/D/E for `vision_early`/`vision_middle`/
`vision_late` -- see "Vision-encoder sub-scopes" addendum below). No per-layer, attention-vs-MLP, routing, transfer, or construction-
dataset work. `external/RandOpt` is never edited; `run_randopt_image_aware.py` is never
modified (not even additively); `released_compat`/`fixed_base`/`sample_candidates`/candidate
seed generation/top-K selection/majority voting are all reused unmodified.

## Addendum: vLLM multimodal encoder-output cache (found during the coarse sweep, fixed)

The 7×4-run coarse sweep (`full_lm` × 4 radii complete; stopped at `vision_encoder r=.005`)
found every one of 100 independently-perturbed `vision_encoder` candidates scoring EXACTLY
the base score — 0 experts, 100 ties, 0 regressions. `vision_encoder r=.02` began showing the
same pattern before the sweep was intentionally stopped.

**Root cause**: vLLM's per-worker multimodal encoder-OUTPUT cache (distinct from
`mm_processor_cache_gb`, the image *preprocessing* cache — disabling that does not touch
this) serves a cached vision embedding for a repeated image regardless of whether model
weights changed since that embedding was computed. `selection_requests`/`test_requests` are
built once and reused across every candidate, so every visual-scope candidate's images hash
to the same cache key as the base model's — and every generation after the first was served
stale, pre-perturbation (or pre-restoration) embeddings.

**Investigation, verified directly against the pinned `vllm==0.27.1` source** (tag `v0.27.1`,
commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` — not assumed from another version):

- `vllm.entrypoints.llm.LLM` (what our unmodified, upstream `RandOptNcclLLM` subclasses) has
  **no `reset_encoder_cache()` method** — confirmed by grepping `entrypoints/llm.py` and all
  three of its mixin base classes (`BeamSearchOfflineMixin`, `PoolingOfflineMixin`,
  `OfflineInferenceMixin`), zero matches. Contrast with `reset_mm_cache`/`reset_prefix_cache`,
  both of which ARE defined on `LLM`.
- `LLMEngine.reset_encoder_cache()` (`vllm/v1/engine/llm_engine.py`) exists and calls
  `self.engine_core.reset_encoder_cache()`, which clears BOTH halves:
  `Scheduler.reset_encoder_cache()` (`vllm/v1/core/sched/scheduler.py:2487`, driver-process
  bookkeeping — `self.encoder_cache_manager.reset()`) and the executor's worker-side reset.
  `LLM` never forwards to this method, unlike its two siblings.
- Since our `engine` is a Ray actor wrapping the whole `RandOptNcclLLM`/`LLM` object, and
  Ray's actor RPC only exposes methods that exist directly on the wrapped class (confirmed:
  no `__getattr__` forwarding on `LLM`), `LLMEngine.reset_encoder_cache()` — and therefore
  the scheduler half — is not reachable from our driver code, and cannot be made reachable
  without either modifying `external/RandOpt`'s `RandOptNcclLLM` (forbidden) or vLLM itself.
- What IS reachable: `LLM.collective_rpc(method, args)` is itself a real, public `LLM`
  method. `Executor.reset_encoder_cache()` (`vllm/v1/executor/abstract.py:314`) is
  implemented as exactly `self.collective_rpc("reset_encoder_cache")`, dispatching to a
  real, built-in `vllm.v1.worker.gpu_worker.Worker.reset_encoder_cache()` method (line 858 —
  not something `WorkerExtension` adds) → `self.model_runner.reset_encoder_cache()`.
  `vlm_adapter.reset_vllm_encoder_cache(engine)` calls this identical string-dispatched
  `collective_rpc` directly — the same mechanism already used throughout this project.

**Known limitation, stated plainly, not glossed over**: `Scheduler._try_schedule_encoder_inputs`
consults `self.encoder_cache_manager.check_and_update_cache(request, i)` — confirmed, from the
scheduler source directly, to be **pure in-memory bookkeeping with no worker consultation**.
Resetting only the worker-side cache does not, by source inspection alone, guarantee the
scheduler will ask the worker to recompute rather than continue believing a (now-cleared)
cache entry is still valid. This is a genuine, sourced finding, not speculation — and it is
exactly why the fix ships alongside a GPU A/B validation rather than being declared correct
from source reading alone: the empirical test (does a `vision_encoder` candidate's raw output
now actually differ from base?) is what determines whether the reachable half is sufficient
in practice.

**First fix attempt (worker-only) — confirmed INSUFFICIENT on GPU**: `scopes.
scope_requires_encoder_cache_reset(scope)` gated a call to `vlm_adapter.
reset_vllm_encoder_cache(engine)` (the worker-only `collective_rpc("reset_encoder_cache")`
reset) after `scoped_apply_perturbation` and before `engine.generate`. The N=2 GPU
validation crashed: `RuntimeError: Encoder cache miss for <mm_hash>`, with scheduler output
immediately before the crash showing `scheduled_encoder_inputs={}` — exactly the predicted
failure mode above, now empirically confirmed rather than merely theorized: the scheduler
still believed the embedding was cached (so it never scheduled recomputation), while the
worker's copy had already been cleared, producing a lookup miss.

**Second fix (full engine-level reset) — implemented, CPU-tested, awaiting GPU
re-validation**: `LLMEngine.reset_encoder_cache()` clears both halves together
(`self.engine_core.reset_encoder_cache()` → both `Scheduler.reset_encoder_cache()` and the
executor's worker-side reset), but is unreachable from `LLM`'s public surface as established
above. Resolved by reading `external/RandOpt/core/engine.py:launch_engines` directly (pinned
commit `536df0a308f3990b6270c991fbb96bd0b779a58e`): it applies `ray.remote(num_cpus=0,
num_gpus=0, scheduling_strategy=strategy)(RandOptNcclLLM).remote(**engine_kwargs)` **fresh,
inside the function body, on every call** — not a class-level `@ray.remote` decorator
applied once at import time. `RandOptNcclLLM` is therefore an ordinary, mutable Python class
object right up until the moment `launch_engines()` wraps it. `vlm_adapter.
ensure_full_encoder_cache_reset_exposed()` adds a `reset_encoder_cache_full` instance method
to that already-imported class object — a runtime attribute addition to an in-memory Python
object, executed entirely in our own package's code — **before** `launch_engines()` is
called, so Ray's actor-method registry (built when `ray.remote(...)` inspects the class)
includes it. This is not a file edit to `external/RandOpt` (nothing on disk changes), not a
subclass (Ray still wraps the exact same `RandOptNcclLLM` class object `launch_engines()`
always would have), and `launch_engines()` itself is called completely unmodified.

`vlm_adapter.reset_vllm_encoder_cache_full(engine)` then calls the resulting method as a
genuine Ray actor method — `engine.reset_encoder_cache_full.remote()`, the same calling
convention as `engine.generate.remote(...)`, **never** `collective_rpc` (which only ever
reaches worker processes and cannot reach driver-process scheduler state at all) — reaching
`self.llm_engine.reset_encoder_cache()` on the real running instance, the full
scheduler-plus-worker reset. This supersedes `reset_vllm_encoder_cache` (the worker-only
function) as the scientific visual-scope reset in `run_scoped_randopt.py`'s sampling and
ensemble phases; the old function is kept in `vlm_adapter.py` only for reference/forensic
value and is proven, by test, to never be dispatched from the candidate loop anymore.
Hard-fails (raises) if the actor doesn't expose the method at all, or if the call itself
errors — never silently continues with a possibly-incoherent cache state.

Nothing about candidate seeds, scopes, radii, relative-L2 math, perturbation math, scoring,
prompts, dataset, image construction, restoration semantics, `N`, or candidate selection was
changed by either fix attempt.

`vision_encoder r=.005`'s pre-fix result AND the worker-only-reset crash log are preserved
on disk as forensic artifacts and are explicitly excluded from the final coarse map — the
completed `full_lm` × 4-radii results are valid and preserved unchanged, not rerun.

## Addendum: vision-encoder sub-scopes (fine-localization inside vision_encoder)

Motivated by the completed, validated 7×4 coarse map: `vision_encoder` showed the highest
expert density of any scope at `r=.04` (0.87) and `r=.07` (0.89). The next question is
*where inside the 32-block vision encoder* that density lives — not a new dataset, model, or
task, only a finer partition of the existing `vision_encoder` scope.

**Partition** (fixed, non-uniform, requested explicitly — 32 is not divisible by 3, so
`partition_layers_into_thirds`'s equal-thirds rule does not apply):

| Scope | Selection | Block count |
|---|---|---|
| `vision_early` | `visual.patch_embed.*`, `visual.blocks.0`..`visual.blocks.10`, and `visual.rotary_pos_emb.*` if any trainable parameters exist there | 11 blocks + patch_embed (+ rotary_pos_emb, if present) |
| `vision_middle` | `visual.blocks.11`..`visual.blocks.21` | 11 blocks |
| `vision_late` | `visual.blocks.22`..`visual.blocks.31` | 10 blocks |

`visual.merger.*` stays excluded from all three, same as `vision_encoder` today. The union of
the three exactly equals `vision_encoder`'s own selection, with zero overlap — proven by test
(`tests/test_scopes.py::test_vision_thirds_union_equals_vision_encoder_exactly`,
`::test_vision_thirds_pairwise_disjoint`) against a real named-parameters set, not merely
inspected.

**Discovery, not hardcoding**: `scopes.discover_vision_block_indices` matches the single
recognized `visual.blocks.(\d+).` pattern (confirmed — Phase 1 above — that `visual.*` is
never re-nested under `language_model./model.` in either LM convention, so unlike LM layers
no convention-discovery step is needed here, only a completeness check) and hard-fails unless
the found indices are exactly the complete `{0, ..., 31}` set the fixed 11/11/10 boundaries
depend on. `rotary_pos_emb` is typically a registered buffer (no trainable `nn.Parameter`),
in which case `named_parameters()` never yields it and `vision_early`'s selector — which
includes the `visual.rotary_pos_emb.` prefix unconditionally, exactly like the "if it has
trainable parameters, assign them to `vision_early`" instruction — simply never matches
anything under that prefix; the resulting manifest's `selected_param_names` is itself the
documentation of whether any existed, no separate flag needed.

**Relative-L2 sigma** is derived independently per sub-scope from its own manifest
(`base_l2_norm`, `total_element_count`) via the same unmodified `compute_relative_l2_sigma`
used by every other scope — `scoped_perturbation.scoped_apply_perturbation` is fully
scope-agnostic and required no changes at all for this milestone.

**Scientific protocol, unchanged**: same model revision, GQA, 200-example selection subset,
scorer/prompt, `fixed_base` restoration, `upstream_per_tensor_reseed` noise semantics,
relative-L2 normalization, candidate seed generation, full encoder-cache reset (all three new
scopes are visual-affecting — added to `_VISUAL_AFFECTING_SCOPES` — so
`scope_requires_encoder_cache_reset` returns `True` for all three), N=100, K=1,
test_samples=5. Only `r ∈ {0.04, 0.07}` (the two radii where `vision_encoder` density peaked)
are used — no new radii. 3 scopes × 2 radii × 100 candidates = 600 candidates total, not run
by this milestone.

**GPU isolation check**: `diagnostics/scope_isolation_gpu_check.py` gained Tests C/D/E,
reusing the identical `_run_isolation_test` helper Test A/B already used — no new diagnostic
framework. `_diag_report_all_scopes` picks up the three new scopes automatically (it iterates
`scopes.PERTURBATION_SCOPES`), so its pre-perturbation report now also prints each new
scope's real selected-tensor count, element count, and base L2 norm before any weight is
touched — the actual numbers depend on the real checkpoint and are not fabricated here.

## Addendum: vision_late sub-scopes (finer localization inside vision_late)

Motivated by the completed, validated 6-cell vision-localization sweep's **paired seed-level
analysis** (all six cells share the identical 100 candidate seeds, enabling exact within-seed
comparisons rather than relying on Wilson-interval overlap): at `r=.04`, `vision_late`'s
expert density was significantly higher than `vision_early`'s (exact two-sided McNemar
p=.0227) and `vision_middle`'s (p=.0192, paired mean delta −.0065, 95% bootstrap CI
[−.0118,−.0012]). At `r=.07`, no pairwise McNemar comparison reached significance. The next
question is *where inside vision_late's 10 blocks (22–31)* the signal concentrates — only at
`r=.04`, only inside `vision_late`, no new dataset/model/task.

**Partition** (fixed 5/5 split of `vision_late`'s own block range):

| Scope | Selection | Block count |
|---|---|---|
| `vision_late_a` | `visual.blocks.22`..`visual.blocks.26` | 5 blocks |
| `vision_late_b` | `visual.blocks.27`..`visual.blocks.31` | 5 blocks |

`vision_late_a ∪ vision_late_b == vision_late` exactly, pairwise disjoint, no merger
parameters, no parameters outside blocks 22–31 — proven by test
(`tests/test_scopes.py::test_vision_late_halves_union_equals_vision_late_exactly`,
`::test_vision_late_halves_pairwise_disjoint`, `::test_vision_late_halves_exclude_merger_and_lm`)
against a real named-parameters set. `partition_vision_late_into_halves` requires the
complete `{0,...,31}` block set, same completeness discipline as `partition_vision_blocks` —
never partitions a partial/gapped set.

Both scopes added to `_VISUAL_AFFECTING_SCOPES` (full encoder-cache reset required, same
mechanism as every other visual scope — no change to the reset logic itself). Relative-L2
sigma is derived independently per half from its own manifest via the same unmodified
`compute_relative_l2_sigma` — `scoped_perturbation.py` again required zero changes, being
fully scope-agnostic. `run_scoped_randopt.py`'s CLI again required zero changes
(`--perturbation-scope` reads its choices from `scopes.PERTURBATION_SCOPES` dynamically).
`vision_early`/`vision_middle`/`vision_late` and all other existing scopes' own selection
logic is untouched — proven by the full existing test suite remaining green (312 passed, 1
skipped, up from 294, zero regressions) alongside a dedicated regression test
(`test_existing_vision_third_scopes_unaffected_by_late_halves`).

**GPU isolation check**: Tests F/G added to `diagnostics/scope_isolation_gpu_check.py`,
reusing the identical `_run_isolation_test` helper again — still no new diagnostic framework.

**Scientific protocol, unchanged**: same model/GQA/subset/scorer, `fixed_base` restoration,
`upstream_per_tensor_reseed` noise semantics, relative-L2 normalization, candidate seed
generation, N=100, K=1, test_samples=5. Only `r=0.04` (the radius where the paired signal was
found) — no `r=.07`, no other scopes revisited. 2 scopes × 1 radius × 100 candidates = 200
candidates, not run by this milestone.
