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
   the Gate 1 baseline (population_size=0, no perturbation/selection/voting involved).
   It does NOT touch core/engine.py, utils/worker_extn.py, or any perturbation/selection/
   voting code -- extending image-awareness to the RandOpt candidate-sampling and ensemble
   loops (needed for Gate 2) is explicitly out of scope until Gate 2 is authorized.
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


def generate_with_images(llm, sampling_params, task_datas: List[Dict], tokenizer) -> List[str]:
    """The minimal fix for divergence #4: identical prompt text/decoding to upstream, but
    each request carries the actual image via vLLM's multi_modal_data instead of silently
    dropping it. task_datas entries are GQAHandler.load_data() records (must have
    "image_path" -- i.e. images_available was True when the split was prepared).
    """
    from PIL import Image

    requests = []
    for d in task_datas:
        text = format_chat_prompt(tokenizer, d["messages"])
        image = Image.open(d["image_path"]).convert("RGB")
        requests.append({"prompt": text, "multi_modal_data": {"image": image}})
    outputs = llm.generate(requests, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]
