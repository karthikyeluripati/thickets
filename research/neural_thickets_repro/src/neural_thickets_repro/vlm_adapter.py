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
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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


def generate_with_images(llm, sampling_params, task_datas: List[Dict], tokenizer) -> List[str]:
    """Gate 1 path: a local (non-Ray) vLLM LLM instance, used by eval_base_image_aware.py."""
    requests = build_image_aware_requests(task_datas, tokenizer)
    outputs = llm.generate(requests, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]
