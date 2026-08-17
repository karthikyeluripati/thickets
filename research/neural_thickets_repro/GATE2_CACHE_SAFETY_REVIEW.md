# Gate 2 cache-safety review

Question: RandOpt perturbs LM weights in-place and repeatedly evaluates the same 200
GQA selection-set prompts across N candidates. Could vLLM's prefix/KV caching or
multimodal-encoder caching let a candidate's generation reuse cached state computed under
a *different* weight state, silently corrupting results?

**Verified from the actual pinned external source (`536df0a308f3990b6270c991fbb96bd0b779a58e`),
not from memory:**

## 1. Prefix/KV cache — disabled, and this is what actually protects us

`core/engine.py:launch_engines`'s own Python signature:
```
def launch_engines(..., enable_prefix_caching: bool = False, ...):
```
and it's threaded straight through: `engine_kwargs["enable_prefix_caching"] = enable_prefix_caching`,
passed to vLLM's `LLM(**engine_kwargs)`. `run_randopt_image_aware.py` calls
`launch_engines(1, model_path, precision=cfg.model.precision, tensor_parallel_size=1, multimodal=True)`
without overriding this argument, so it uses the wrapper's own default: **False**.

Critically, `utils/worker_extn.py`'s `perturb_self_weights`/`restore_self_weights` (verified,
full bodies read) call **only** `torch.cuda.synchronize()` and `torch.cuda.empty_cache()` —
the latter is PyTorch's own allocator reclaiming free memory, unrelated to vLLM's internal
KV-cache/prefix-cache bookkeeping (a separate radix-tree/hash structure vLLM maintains
itself). **There is no cache-invalidation call anywhere in the released weight-perturbation
path.** If prefix caching were enabled, nothing in the released code would stop a later
request from reusing KV blocks computed under a stale weight state — a real, unaddressed
gap in the upstream design. We are safe only because the default is `False` and our Gate 2
code doesn't turn it on.

## 2. Multimodal/vision-encoder caching — safe by construction, not by disabling anything

No separate multimodal-processor-cache setting is configured anywhere in `engine_kwargs`
(checked — grepped for `mm_processor_cache`/`processor_cache` across both files, zero
hits), so vLLM's own default applies, whatever it is. This doesn't matter: the visual
encoder is never perturbed (`WorkerExtension._should_perturb` excludes every `visual.`/
`model.visual.`-prefixed parameter, confirmed against the real checkpoint's actual tensor
names earlier in this project — REPRO_SPEC.md "Perturbation scope (exact)"), so any cached
vision-side computation (image preprocessing, patch embeddings) is produced by weights that
are constant across every candidate. Reusing it is always correct, not merely "probably
fine" — there is nothing for a change in LM weights to make stale on that side.

## 3. No output/hidden-state reuse on our side

`run_randopt_image_aware.py` builds `selection_requests`/`test_requests` **once**, before
the candidate loop, and passes the same request objects to `engine.generate.remote(...)`
for every candidate — this is intentional (same prompts, different weights each time) and
is exactly why (1) matters: with prefix caching off, each `.generate()` call is a full,
independent forward pass under whatever weights are current at that moment. `enforce_eager=True`
(also unconditionally set in `engine_kwargs`) additionally means no persistent compiled
CUDA-graph state that could bake in stale weight references.

## Conclusion: cache safety is already guaranteed. No code change made.

Per the request: no unnecessary changes when safety is already established. The one thing
added is `diagnostics/gate2_gpu_preflight.py` (below) — an empirical, end-to-end check on
real GPU hardware that directly exercises this exact mechanism (perturb → generate → verify
changed → restore → generate → verify matches base) before N=20 runs. Its "output changes
after perturbation" check doubles as a live proof that no caching is masking weight
changes: if prefix caching (or any other cache) *were* incorrectly reusing stale state, that
check would be the first thing to fail.
