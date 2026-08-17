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

2. Multimodal engine flag for the GQA/VLM path.
   external/RandOpt/randopt.py's main() does not appear to pass `multimodal=True` to
   core/engine.py's launch_engines() even though GQA prompts contain image content;
   `multimodal=True` is what sets vLLM's `limit_mm_per_prompt={"image": 1}`. Whether this
   matters (the pinned vLLM version may auto-detect multimodal models from config) is
   UNRESOLVED from reading alone -- see REPRO_SPEC.md. It can only be settled by testing
   on GPU hardware during Gate 1. If it proves to be a real gap, external/multimodal_
   patch_launcher.py is a ready-to-use fix: it monkeypatches launch_engines() at import
   time to force multimodal=True, without ever editing the external clone's files on disk.
   Try eval_base.py/run_randopt.py unmodified first; fall back to the launcher only if that
   run fails or clearly mishandles image inputs. MULTIMODAL_FIX_NOTES below is where the
   finding gets recorded either way.
"""
from __future__ import annotations

# transformers class to use instead of AutoModelForCausalLM when reconstructing a saved
# Qwen2.5-VL-3B-Instruct expert from a (seed, sigma) pair.
CORRECTED_MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"

# Filled in once Gate 1 testing on real GPU hardware resolves divergence #2 above.
# None = not yet tested. When resolved, set to a short factual note (what was observed,
# whether a fix was needed and what it was), not a guess.
MULTIMODAL_FIX_NOTES: "str | None" = None


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
