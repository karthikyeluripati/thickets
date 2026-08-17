"""Generation-only script, run identically in BOTH vLLM environments (0.27.1's existing
venv, and the isolated 0.11.0 venv/container) -- same code, same args, only the installed
vLLM version differs. Deliberately standalone (only needs transformers, vllm, Pillow,
huggingface_hub -- NOT the full neural_thickets_repro package or the external RandOpt
clone) so it drops cleanly into a minimal isolated environment without touching whatever
else is installed there.

Reads the fixed 200-example sample (select_fixed_sample.py's output) and generates with:
same checkpoint+revision, same prompt text (verbatim from the fixed-sample file, not
re-derived), same image, same bf16 precision, same greedy decoding/max_tokens. The label
you pass just names the output file -- it changes nothing about generation itself.

Usage (run once per environment):
    python generate_predictions.py \
        --fixed-sample results/gate1_diagnosis/vllm_version_control/fixed_200.json \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \
        --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
        --max-tokens 256 --seed 42 --label vllm0271 \
        --out-dir results/gate1_diagnosis/vllm_version_control
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-sample", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", required=True, help="pinned model revision -- must be identical across both runs")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True, help="e.g. vllm0271 or vllm0110 -- names the output file only")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    fixed = json.loads(Path(args.fixed_sample).read_text())
    examples = fixed["examples"]
    print(f"Loaded {len(examples)} fixed examples (pinned seed={fixed['seed']}).")

    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=args.model_name, revision=args.revision)
    print(f"Resolved {args.model_name}@{args.revision} -> {model_path}")

    import vllm
    from PIL import Image
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"vLLM version in this environment: {vllm.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype=args.precision,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(temperature=0.0, seed=args.seed, max_tokens=args.max_tokens)

    requests = []
    for ex in examples:
        # Message shape MUST mirror external/RandOpt/data_handlers/gqa.py's own construction
        # (also what vlm_adapter.generate_with_images/format_chat_prompt renders) exactly:
        # a content LIST with an explicit {"type": "image", ...} block, not a bare string.
        # Qwen2.5-VL's chat template only inserts the <|vision_start|><|image_pad|>
        # <|vision_end|> placeholder tokens when it sees that marker in the input messages --
        # a bare-string content produces a prompt with ZERO image placeholders, so vLLM's
        # multi_modal_data has nothing to bind to and raises "Failed to apply prompt
        # replacement for mm_items['image'][0]" on the first multimodal request. This is
        # NOT a second, differing multimodal formatting implementation -- it's the same
        # shape, deliberately duplicated (not imported) only so this script stays
        # standalone enough to run inside the isolated vLLM 0.11.0 Docker container.
        messages = [{"role": "user", "content": [
            {"type": "image", "image": ex["image_path"]},
            {"type": "text", "text": ex["formatted_prompt_text"]},
        ]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image = Image.open(ex["image_path"]).convert("RGB")
        requests.append({"prompt": text, "multi_modal_data": {"image": image}})

    start = time.time()
    outputs = llm.generate(requests, sampling_params, use_tqdm=True)
    elapsed = time.time() - start
    print(f"Generated {len(outputs)} responses in {elapsed:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"predictions_{args.label}.jsonl"
    with predictions_path.open("w") as f:
        for ex, out in zip(examples, outputs):
            f.write(json.dumps({
                "question_id": ex["question_id"],
                "image_id": ex["image_id"],
                "reference_answer": ex["reference_answer"],
                "raw_prediction": out.outputs[0].text,
            }) + "\n")

    metadata_path = out_dir / f"metadata_{args.label}.json"
    metadata_path.write_text(json.dumps({
        "label": args.label,
        "vllm_version": vllm.__version__,
        "python_version": platform.python_version(),
        "model_name": args.model_name,
        "model_revision": args.revision,
        "model_snapshot_path": model_path,
        "precision": args.precision,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "fixed_sample_path": str(args.fixed_sample),
        "fixed_sample_seed": fixed["seed"],
        "n_examples": len(examples),
        "generation_elapsed_seconds": elapsed,
    }, indent=2))

    print(f"Wrote {predictions_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
