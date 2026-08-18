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
mechanical isolation check. No per-layer, attention-vs-MLP, routing, transfer, or construction-
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

**Fix**: `scopes.scope_requires_encoder_cache_reset(scope)` (pure, `vision_encoder`/
`vision_merger`/`full_vlm` → `True`, the four LM-only scopes → `False`) gates a call to
`vlm_adapter.reset_vllm_encoder_cache(engine)` in `run_scoped_randopt.py`'s sampling and
ensemble phases, inserted after `scoped_apply_perturbation` and before `engine.generate`.
Hard-fails (raises) if the `collective_rpc` call itself errors or returns an unexpected
shape — never silently continues with a possibly-stale cache. Nothing about candidate seeds,
scopes, radii, relative-L2 math, perturbation math, scoring, prompts, dataset, image
construction, restoration semantics, `N`, or candidate selection was changed.

`vision_encoder r=.005`'s pre-fix result is preserved on disk as a forensic artifact
(0/100/0 expert/tie/regression counts, all candidates scoring 0.5600) and is explicitly
excluded from the final coarse map — the completed `full_lm` × 4-radii results are valid and
preserved unchanged.
