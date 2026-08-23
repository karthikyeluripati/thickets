"""The one place upstream/VLM-specific corrections live.

external/RandOpt/ (see external/EXTERNAL_COMMIT.txt) is never edited. Any divergence
needed to make it work correctly for Qwen2.5-VL-3B-Instruct specifically is documented and
applied here, in our own wrapper code, instead of as a patch to the external clone.

Known divergences (see REPRO_SPEC.md for full citations):

1. Model class for standalone single-expert reconstruction.
   external/RandOpt/utils/repro_seed.py loads the checkpoint via
   `transformers.AutoModelForCausalLM`, which is not the correct class for
   Qwen2.5-VL-3B-Instruct -- its own config.json declares
   `architectures: ["Qwen2_5_VLForConditionalGeneration"]` (confirmed by fetching the
   checkpoint's config.json/safetensors index metadata from the HF hub, no weights
   downloaded). CORRECTED_MODEL_CLASS_NAME below names the right class for any wrapper
   script we write that reconstructs a saved top-K expert for this checkpoint.

2. Multimodal engine flag for the GQA/VLM path -- RESOLVED, subsumed by divergence #4.
   The `multimodal=True` flag itself turned out not to be the story: the real gap is #4
   below (no image data is ever constructed at all, so the flag alone wouldn't have
   helped). See MULTIMODAL_FIX_NOTES.

3. Model revision not reaching vLLM (see resolve_model_snapshot below) -- RESOLVED.
   randopt.py has no --revision argument; passing the hub name let vLLM resolve
   revision=None. Fixed by resolving the pin to a local snapshot path via
   huggingface_hub.snapshot_download and passing that path as --model_name.

4. Images never reach the model at all -- CONFIRMED root cause of the Gate 1 hard fail
   (baseline 17.94%/15% vs published 56.6%). GATE1_DIAGNOSIS.md has the full writeup.
   randopt.py formats GQA prompts via tokenizer.apply_chat_template() into plain text
   (which inserts <|vision_start|><|image_pad|><|vision_end|> placeholder tokens) and
   calls vLLM generate() with those bare strings -- `multi_modal_data` appears nowhere in
   the released repo at any git revision, including the March 2026 paper-era commit.
   Confirmed on GPU (100-example paired audit, seed=42): text-only 15%/22%
   (head/march-era scoring) vs identical-prompt-with-image 59%/59%; 45 examples flipped
   wrong->correct when the image was added vs 1 the other way; 20/20 question<->image
   pairs independently verified against the source HF dataset.
   generate_with_images() below is the minimal fix, used by eval_base_image_aware.py for
   the Gate 1 baseline (population_size=0, no perturbation/selection/voting involved), and
   build_image_aware_requests() (the request-construction piece factored out of it) is
   reused by run_randopt_image_aware.py for Gate 2 -- same request shape, generated via
   Ray's engines[i].generate.remote(requests, ...) there instead of a local llm.generate(),
   since Gate 2 must go through the actual Ray-actor vLLM engines (core/engine.py:
   launch_engines, unmodified) to reach utils/worker_extn.py's real weight-perturbation
   RPCs. Neither path touches core/engine.py or utils/worker_extn.py's own code.

5. Ray must be bootstrapped before launch_engines() -- core/engine.py:launch_engines calls
   ray.cluster_resources() and assumes an active Ray session; it never calls ray.init()
   itself (confirmed: no ray.init anywhere in core/engine.py). Upstream's OWN entrypoint,
   randopt.py:main(), does this immediately before its own launch_engines() call:
       if os.environ.get("RAY_ADDRESS"):
           ray.init(address="auto", ignore_reinit_error=True)
       else:
           ray.init(address="local", ignore_reinit_error=True)
   (verified against the pinned source; described here, not copied -- it's two lines).
   bootstrap_ray() below mirrors this exactly. Both run_randopt_image_aware.py and
   diagnostics/gate2_gpu_preflight.py call core/engine.py's launch_engines() directly
   without going through randopt.py's own main(), so neither got this bootstrap step for
   free -- both must call bootstrap_ray() themselves before launch_engines().

6. Ray WORKER processes can't import `core`/`data_handlers` even though the driver can --
   ModuleNotFoundError: No module named 'core' when Ray deserializes the RandOptNcclLLM
   actor on a worker. We add external/RandOpt to sys.path in the driver process (a runtime,
   in-memory mutation of THAT process only); Ray spawns workers as separate processes that
   do not inherit it. Checked how upstream avoids this in its own typical usage: randopt.py
   itself does zero sys.path manipulation (verified -- grepped the whole file), and
   scripts/local_run.sh instead `cd`s to the repo root before running `python3 randopt.py`
   (`cd "$(dirname "$0")/.."`), relying on Python's default script-directory sys.path[0]
   insertion -- a driver-only mechanism whose effect on Ray's own worker processes is not
   something we can rely on either way, and irrelevant to us regardless since we
   sys.path.insert() rather than chdir(). The explicit, Ray-documented, guaranteed-correct
   fix is a job-level runtime_env PYTHONPATH env var, applied at ray.init() time -- this
   propagates to every worker Ray spawns for that job, whether the underlying cluster is a
   brand-new local one or a pre-existing one we're connecting to fresh (Ray's runtime_env is
   scoped to the JOB/driver call, not the cluster's original startup). bootstrap_ray() below
   does this only when it's the one actually calling ray.init() -- if Ray was already
   initialized by something else in our own process, we don't control its runtime_env, so
   we can't inject a PYTHONPATH after the fact; that path relies on
   verify_workers_can_import_external_root() below actually running and failing loudly
   rather than launch_engines() being trusted blindly.

7. vLLM's multimodal encoder-OUTPUT cache serves stale vision embeddings across
   scoped-RandOpt candidates that perturb the visual path -- confirmed real bug (GPU
   evidence: vision_encoder r=0.005/r=0.02 candidates all scored exactly the base score,
   0 experts / 100 ties / 0 regressions, for 100 independently-seeded perturbations).
   Verified directly against the pinned vllm==0.27.1 source (not assumed from another
   version -- see reset_vllm_encoder_cache() below for the exact file:line citations):
   this cache is distinct from mm_processor_cache_gb (image PREPROCESSING cache);
   disabling that does not touch it. vllm.entrypoints.llm.LLM (what our unmodified,
   upstream RandOptNcclLLM subclasses) has NO reset_encoder_cache() method itself --
   confirmed by grepping entrypoints/llm.py and all three of its mixin base classes,
   zero matches -- unlike reset_mm_cache/reset_prefix_cache, which ARE exposed there.
   LLMEngine.reset_encoder_cache() exists and clears BOTH the scheduler's own bookkeeping
   (Scheduler.reset_encoder_cache -> encoder_cache_manager.reset(), a driver-process
   structure) and the worker-side cached tensors, but LLM never forwards to it, and since
   our engine is a Ray actor wrapping the whole unmodified LLM object, Ray's actor RPC
   only exposes methods that exist directly on the wrapped class -- the driver-process
   scheduler half is therefore not reachable from our code. What IS reachable: the same
   collective_rpc(method, args) mechanism already used throughout this project, dispatched
   with the string "reset_encoder_cache" -- a real, built-in vllm.v1.worker.gpu_worker.
   Worker.reset_encoder_cache() method (not something WorkerExtension adds), which is
   exactly what vLLM's own Executor.reset_encoder_cache() convenience method does
   internally. reset_vllm_encoder_cache() below calls this directly. KNOWN LIMITATION,
   not glossed over: the scheduler's cache-hit decision is made purely from its own
   bookkeeping without consulting the worker, so a worker-only reset is not proven
   sufficient by source inspection alone -- the GPU A/B validation this fix ships
   alongside is the empirical check for whether the reachable half is enough in practice.

8. Divergence #7's worker-only reset was CONFIRMED insufficient on GPU -- exactly the
   predicted limitation materialized: `RuntimeError: Encoder cache miss for <mm_hash>`,
   scheduler output immediately before the crash showing `scheduled_encoder_inputs={}`
   (the scheduler believed the embedding was cached and never scheduled recomputation; the
   worker's copy had already been cleared -> lookup miss -> crash). The full engine-level
   reset (`LLMEngine.reset_encoder_cache()`, clearing both scheduler and worker halves) is
   not reachable via any method vllm.entrypoints.llm.LLM exposes, and Ray's actor RPC only
   ever calls methods that exist on the wrapped class. Fix: external/RandOpt/core/engine.py:
   launch_engines applies `ray.remote(...)` to RandOptNcclLLM FRESH, inside the function
   body, on every call (verified directly against the pinned source -- not a class-level
   `@ray.remote` decorator) -- so adding a method to the already-imported RandOptNcclLLM
   class object BEFORE calling launch_engines() makes Ray's actor-method registry include
   it. ensure_full_encoder_cache_reset_exposed() does exactly this (a runtime attribute
   addition to an in-memory class object, in our own package's code -- external/RandOpt is
   never edited on disk, no subclass, launch_engines() itself unmodified), and
   reset_vllm_encoder_cache_full() calls the resulting real actor method
   (`engine.reset_encoder_cache_full.remote()`, reaching `self.llm_engine.
   reset_encoder_cache()`), superseding divergence #7's collective_rpc-based approach for
   the actual scoped-RandOpt candidate loop.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# transformers class to use instead of AutoModelForCausalLM when reconstructing a saved
# Qwen2.5-VL-3B-Instruct expert from a (seed, sigma) pair.
CORRECTED_MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"

MULTIMODAL_FIX_NOTES = (
    "Confirmed on GPU (see GATE1_DIAGNOSIS.md): the missing multimodal=True flag was not "
    "the actual problem. The real gap is that no code path in the released repo ever "
    "constructs multi_modal_data for vLLM's generate() call -- images are never attached "
    "to a request regardless of engine flags. generate_with_images() below fixes this "
    "directly for the Gate 1 baseline path."
)

EXTERNAL_ROOT = Path(__file__).resolve().parents[2] / "external" / "RandOpt"


def bootstrap_ray(external_root: "str | Path" = EXTERNAL_ROOT) -> bool:
    """Mirrors upstream randopt.py:main()'s own Ray init sequence (divergence #5 above):
    RAY_ADDRESS env var -> address="auto", else address="local", both with
    ignore_reinit_error=True. Required because core/engine.py:launch_engines() assumes an
    active Ray session and never starts one itself.

    Also fixes divergence #6 (ModuleNotFoundError: No module named 'core' on Ray workers):
    when THIS call is the one actually invoking ray.init() (Ray wasn't already initialized
    in our process), it passes a job-level runtime_env injecting external_root onto
    PYTHONPATH -- this propagates to every worker Ray spawns for this job, whether the
    underlying cluster is brand-new or one we're connecting to fresh, and preserves upstream
    module identity (workers resolve `core`/`core.engine` to the same pinned external file
    the driver uses, not a copy). Any existing PYTHONPATH is preserved, external_root
    prepended.

    Returns True if THIS call actually initialized Ray (i.e. the caller now owns the
    session, is responsible for shutting it down on exit/failure, AND had its runtime_env
    applied), False if Ray was already running (e.g. an existing cluster someone else
    started, possibly in our own process) -- in which case we neither shut it down nor can
    guarantee its workers see external_root; the caller MUST run
    verify_workers_can_import_external_root() and fail clearly rather than assume.
    """
    import os

    import ray

    already_running = ray.is_initialized()
    if already_running:
        return False

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = os.pathsep.join(p for p in (str(external_root), existing_pythonpath) if p)
    runtime_env = {"env_vars": {"PYTHONPATH": new_pythonpath}}

    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
    else:
        ray.init(address="local", ignore_reinit_error=True, runtime_env=runtime_env)
    return True


def verify_workers_can_import_external_root(external_root: "str | Path" = EXTERNAL_ROOT) -> None:
    """Tiny worker-side sanity check, run BEFORE launch_engines() -- catches "ModuleNotFoundError:
    No module named 'core'" in seconds via a trivial remote call, instead of after the
    slower/more opaque actor-startup path (placement groups + vLLM engine init inside
    launch_engines()) fails deep in Ray actor deserialization.

    Submits a @ray.remote task (executes on an actual Ray WORKER process, not the driver)
    that imports `core`/`core.engine` -- the exact modules launch_engines()'s RandOptNcclLLM
    actor needs -- and returns their resolved file paths. Also verifies the worker resolved
    the SAME pinned external_root, not some other same-named `core` module that happens to
    be importable on the worker (e.g. a stray package or stale checkout elsewhere on
    PYTHONPATH) -- silently running the wrong code would be worse than failing loudly.

    Raises RuntimeError with an actionable message on any failure. Call this unconditionally
    (regardless of whether bootstrap_ray() owned the session) -- it's the only reliable way
    to know workers can actually see external_root, especially when connected to an
    already-initialized session whose runtime_env we didn't control.
    """
    import ray

    external_root = str(Path(external_root).resolve())

    @ray.remote
    def _check_core_importable():
        import core  # type: ignore
        import core.engine  # type: ignore

        return core.__file__, core.engine.__file__

    try:
        core_file, engine_file = ray.get(_check_core_importable.remote())
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Ray WORKER processes cannot import 'core' from {external_root} ({exc}). The "
            f"driver can import it (sys.path.insert in this process), but Ray workers are "
            f"separate processes that don't inherit that -- they need it via PYTHONPATH/"
            f"runtime_env instead. If bootstrap_ray() owned this Ray session, its runtime_env "
            f"PYTHONPATH injection should have handled this; something else is wrong. If "
            f"instead we connected to an ALREADY-RUNNING external Ray session (bootstrap_ray "
            f"returned owned=False), we do not control its runtime_env -- that cluster's "
            f"worker nodes need PYTHONPATH configured to include this path themselves before "
            f"we can proceed."
        ) from exc

    if not core_file.startswith(external_root):
        raise RuntimeError(
            f"Ray workers imported a DIFFERENT 'core' module than the pinned external "
            f"RandOpt source: worker resolved core.__file__={core_file!r}, expected it under "
            f"{external_root!r}. Refusing to proceed rather than silently run unexpected code."
        )

    print(f"Worker-side import check passed: core={core_file}, core.engine={engine_file}")


def resolve_model_snapshot(model_name: str, revision: str) -> str:
    """Downloads (or reuses from cache) the checkpoint at the EXACT pinned revision and
    returns the local snapshot path.

    Divergence #3: upstream randopt.py has no --revision argument and passes --model_name
    straight to vLLM, which then resolves whatever the hub's current main is
    (runtime logs show revision=None). Passing a local snapshot path pinned via
    huggingface_hub.snapshot_download is the zero-upstream-modification way to guarantee
    the configured revision is what actually runs. Note: the snapshot path contains the
    model name, so upstream's is_instruct_model substring check ('instruct' in name)
    still matches and the chat-template path is still taken.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_name, revision=revision)


def load_gqa_handler(external_root: "str | Path" = EXTERNAL_ROOT):
    """Imports GQAHandler from the external clone -- same class, same code, only reachable
    from our process because we add the external repo to sys.path (never copied/edited).
    """
    root = str(external_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from data_handlers.gqa import GQAHandler  # type: ignore

    return GQAHandler()


def format_chat_prompt(tokenizer, messages: List[dict]) -> str:
    """Byte-for-byte what upstream randopt.py's format_prompt() does for an instruct model
    (Qwen2.5-VL-3B-**Instruct** always matches that branch). Only the text side of the
    prompt -- the image is attached separately via generate_with_images()'s
    multi_modal_data, which is exactly the one piece upstream never does.
    """
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_image_aware_requests(task_datas: List[Dict], tokenizer) -> List[dict]:
    """The minimal fix for divergence #4, factored out so both the direct-LLM path
    (generate_with_images, Gate 1) and the Ray-actor path (run_randopt_image_aware.py,
    Gate 2) build byte-identical requests: identical prompt text/message shape to upstream
    (data_handlers/gqa.py's own {"type": "image", ...} + {"type": "text", ...} content
    list), but each request carries the actual image via vLLM's multi_modal_data instead of
    silently dropping it. task_datas entries are GQAHandler.load_data() records (must have
    "image_path" -- i.e. images_available was True when the split was prepared).
    """
    from PIL import Image

    requests = []
    for d in task_datas:
        text = format_chat_prompt(tokenizer, d["messages"])
        image = Image.open(d["image_path"]).convert("RGB")
        requests.append({"prompt": text, "multi_modal_data": {"image": image}})
    return requests


def build_multimodal_requests(
    prompts_and_images: List[Tuple[List[dict], Optional["Image.Image"]]], tokenizer,
) -> List[dict]:
    """Generalizes build_image_aware_requests() above for the Capability Benchmark Gate
    (benchmarks/runner.py) -- arbitrary (already-built chat "messages" list, optional PIL
    image) pairs, no GQAHandler-shaped dict keys assumed (build_image_aware_requests itself
    is untouched, still used byte-for-byte by Gate 1/2). Supports image=None (the
    image-dependence sanity check's text-only condition) by omitting multi_modal_data
    entirely from that request, matching vLLM's own convention for a text-only prompt,
    rather than passing a null/placeholder image.
    """
    requests: List[dict] = []
    for messages, image in prompts_and_images:
        text = format_chat_prompt(tokenizer, messages)
        request: Dict[str, object] = {"prompt": text}
        if image is not None:
            request["multi_modal_data"] = {"image": image}
        requests.append(request)
    return requests


def generate_with_images(llm, sampling_params, task_datas: List[Dict], tokenizer) -> List[str]:
    """Gate 1 path: a local (non-Ray) vLLM LLM instance, used by eval_base_image_aware.py."""
    requests = build_image_aware_requests(task_datas, tokenizer)
    outputs = llm.generate(requests, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


def reset_vllm_encoder_cache(engine) -> None:
    """WORKER-ONLY encoder-cache reset -- SUPERSEDED as the scientific visual-scope reset by
    reset_vllm_encoder_cache_full() below. Kept for reference/forensic value (this is what
    the vision_encoder r=.005/.02 GPU runs used) but confirmed INSUFFICIENT on GPU:
    RuntimeError: Encoder cache miss for <mm_hash>, with scheduler output immediately before
    the crash showing scheduled_encoder_inputs={} -- proof the driver-process scheduler's own
    encoder_cache_manager bookkeeping still believed the (now worker-cleared) embedding was
    valid and never scheduled recomputation. See reset_vllm_encoder_cache_full()'s docstring
    for the full engine-level reset that fixes this. Do not wire this function into the
    scoped-RandOpt candidate loop.

    Invalidates vLLM's per-worker multimodal encoder-OUTPUT cache -- see divergence #7 above
    for the full investigation (confirmed against the pinned vllm==0.27.1 source, not assumed
    from another version). Dispatches the string "reset_encoder_cache" through the same
    collective_rpc(method, args) mechanism already used throughout this project (`engine` is
    the Ray actor handle from core/engine.py:launch_engines) -- this resolves to a REAL,
    built-in vllm.v1.worker.gpu_worker.Worker.reset_encoder_cache() method (not something
    WorkerExtension adds), the identical call vLLM's own Executor.reset_encoder_cache()
    convenience method makes internally (`self.collective_rpc("reset_encoder_cache")`).

    Hard-fails (raises, does not silently continue) if the collective_rpc call itself errors,
    or if it doesn't return the expected TP=1 single-worker-result shape -- refusing to
    assume the reset succeeded rather than silently proceeding with a visual scope's
    perturbation while the cache might still be stale.
    """
    import ray

    try:
        results = ray.get(engine.collective_rpc.remote("reset_encoder_cache", args=()))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"reset_vllm_encoder_cache: collective_rpc('reset_encoder_cache') failed "
            f"({type(exc).__name__}: {exc}). The installed vLLM engine does not expose a "
            f"working encoder-cache reset path -- refusing to silently proceed with a "
            f"visual scope's perturbation, since stale cached vision embeddings would make "
            f"every candidate's score collapse to the base score (exactly the bug this "
            f"exists to prevent)."
        ) from exc

    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(
            f"reset_vllm_encoder_cache: collective_rpc('reset_encoder_cache') returned an "
            f"unexpected shape ({results!r}) -- expected vLLM's own list-of-per-worker-"
            f"results contract with exactly one worker (TP=1). Refusing to assume the reset "
            f"succeeded."
        )


_ENCODER_CACHE_FULL_RESET_METHOD_NAME = "reset_encoder_cache_full"


def ensure_full_encoder_cache_reset_exposed(external_root: "str | Path" = EXTERNAL_ROOT) -> None:
    """Divergence #8 (supersedes #7's worker-only approach, confirmed insufficient on GPU:
    RuntimeError: Encoder cache miss -- the driver-process scheduler's own bookkeeping never
    learned the worker-side cache was cleared, so it kept scheduling requests as if the
    embedding were still valid, producing a hard crash rather than silent staleness).

    We need vllm.v1.engine.llm_engine.LLMEngine.reset_encoder_cache() (confirmed, verified
    directly against the pinned vllm==0.27.1 source: vllm/v1/engine/llm_engine.py) -- it
    calls self.engine_core.reset_encoder_cache(), which clears BOTH
    Scheduler.reset_encoder_cache() (driver-process bookkeeping,
    vllm/v1/core/sched/scheduler.py) AND the worker-side cache, the full engine-level reset.
    vllm.entrypoints.llm.LLM (what external/RandOpt/core/engine.py's RandOptNcclLLM
    subclasses, unmodified) never forwards to this method, and since our engine is a Ray
    actor, Ray's actor RPC only exposes methods that exist directly on the wrapped class at
    the moment `ray.remote(...)` decorates it -- LLMEngine.reset_encoder_cache() is normally
    unreachable from outside the actor process for exactly that reason.

    The fix: external/RandOpt/core/engine.py:launch_engines applies `ray.remote(...)` to
    RandOptNcclLLM FRESH, INSIDE the function body, every time it's called -- confirmed by
    reading the pinned source directly (`ray.remote(num_cpus=0, num_gpus=0,
    scheduling_strategy=strategy)(RandOptNcclLLM).remote(**engine_kwargs)`), not a
    class-level `@ray.remote` decorator applied once at class-definition/import time. Since
    Python classes are ordinary mutable objects, and `RandOptNcclLLM` is looked up from
    `core.engine`'s module namespace at THAT call time, adding a method to the (already
    imported, in-memory) class object BEFORE launch_engines() is called makes Ray pick it up
    when it inspects the class to build the actor's method registry -- giving us a REAL Ray
    actor method (`engine.reset_encoder_cache_full.remote()`, called like
    `engine.generate.remote(...)`, NOT collective_rpc, which only ever reaches workers) that
    reaches `self.llm_engine.reset_encoder_cache()` on the actual running instance.

    This is a runtime attribute addition to an in-memory Python object, in our own package's
    code -- external/RandOpt's core/engine.py is never edited on disk, RandOptNcclLLM's
    class body is never modified in any file, no subclass is created, and launch_engines()
    itself is called completely unmodified. Deterministic and idempotent: setattr only if the
    method name isn't already present, so calling this repeatedly (e.g. once per script
    invocation, safely re-callable across multiple launch_engines() calls within one process)
    is a no-op after the first call. MUST be called before launch_engines(), or the resulting
    actor won't have this method and reset_vllm_encoder_cache_full() below will raise.
    """
    root = str(external_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from core.engine import RandOptNcclLLM  # type: ignore

    if not hasattr(RandOptNcclLLM, _ENCODER_CACHE_FULL_RESET_METHOD_NAME):
        def _reset_encoder_cache_full(self) -> None:
            self.llm_engine.reset_encoder_cache()

        setattr(RandOptNcclLLM, _ENCODER_CACHE_FULL_RESET_METHOD_NAME, _reset_encoder_cache_full)


def reset_vllm_encoder_cache_full(engine) -> None:
    """Calls the FULL engine-level encoder-cache reset (scheduler bookkeeping + worker-side
    cache together) via the reset_encoder_cache_full method
    ensure_full_encoder_cache_reset_exposed() adds to RandOptNcclLLM before launch_engines()
    is called. This is a genuine Ray actor method call (like engine.generate.remote(...)),
    NOT collective_rpc -- collective_rpc only ever reaches worker processes and was confirmed
    on GPU to be insufficient for this specific cache (RuntimeError: Encoder cache miss,
    because the driver-process scheduler never learned the worker-side half was cleared).

    Hard-fails (raises, does not silently continue) if the actor doesn't expose
    reset_encoder_cache_full at all (ensure_full_encoder_cache_reset_exposed() wasn't called
    before launch_engines(), so the monkey-patched method wasn't present when Ray wrapped
    RandOptNcclLLM as an actor -- surfaces as AttributeError on the actor handle) or if the
    call itself errors for any other reason. Never falls back to the worker-only
    reset_vllm_encoder_cache() -- that path is confirmed insufficient for the scientific
    visual-scope path.
    """
    import ray

    try:
        method = getattr(engine, _ENCODER_CACHE_FULL_RESET_METHOD_NAME)
    except AttributeError as exc:
        raise RuntimeError(
            f"reset_vllm_encoder_cache_full: the engine actor does not expose "
            f"'{_ENCODER_CACHE_FULL_RESET_METHOD_NAME}' -- "
            f"ensure_full_encoder_cache_reset_exposed() must be called BEFORE launch_engines() "
            f"so the monkey-patched method is present when Ray wraps RandOptNcclLLM as an "
            f"actor. Refusing to silently fall back to the worker-only reset, which was "
            f"confirmed insufficient on GPU (RuntimeError: Encoder cache miss)."
        ) from exc

    try:
        ray.get(method.remote())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"reset_vllm_encoder_cache_full: engine.{_ENCODER_CACHE_FULL_RESET_METHOD_NAME}."
            f"remote() failed ({type(exc).__name__}: {exc}). Refusing to silently proceed "
            f"with a visual scope's perturbation without a confirmed successful full "
            f"(scheduler + worker) cache reset."
        ) from exc
