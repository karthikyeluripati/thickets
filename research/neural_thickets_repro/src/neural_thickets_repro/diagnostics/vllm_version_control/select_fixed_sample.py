"""Pins a fixed 200-example sample from testdev.parquet ONCE, to a file, so the vLLM
0.27.1 and vLLM 0.11.0 runs (potentially different Python environments/venvs, possibly
different machines) evaluate byte-identically the same question IDs -- not two independent
same-seed samples that happen to usually agree.

Usage:
    python -m neural_thickets_repro.diagnostics.vllm_version_control.select_fixed_sample \
        --data-dir external/RandOpt/data/gqa --sample-size 200 --seed 42 \
        --out results/gate1_diagnosis/vllm_version_control/fixed_200.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(EXTERNAL_ROOT / "data" / "gqa"))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "gate1_diagnosis" / "vllm_version_control" / "fixed_200.json"))
    args = parser.parse_args(argv)

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from data_handlers.gqa import GQAHandler  # type: ignore

    handler = GQAHandler()
    task_datas = handler.load_data(str(Path(args.data_dir) / "testdev.parquet"), split="test", max_samples=None)
    with_images = [d for d in task_datas if "image_path" in d]
    if len(with_images) != len(task_datas):
        print(f"WARNING: {len(task_datas) - len(with_images)} examples have no image_path", file=sys.stderr)

    rng = random.Random(args.seed)
    sample = rng.sample(with_images, min(args.sample_size, len(with_images)))

    fixed = [
        {
            "question_id": str(d["question_id"]),
            "image_id": Path(d["image_path"]).stem,
            "image_path": d["image_path"],
            # The FULLY FORMATTED prompt text upstream builds (the "Look at the image and
            # answer the question... \boxed{}" wrapper around the bare GQA question) --
            # generate_predictions.py uses this verbatim, so both vLLM versions see
            # byte-identical prompt text, not a re-derivation that could subtly drift.
            "formatted_prompt_text": d["messages"][0]["content"][1]["text"],
            "reference_answer": d["ground_truth"]["answer"],
        }
        for d in sample
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seed": args.seed, "n": len(fixed), "examples": fixed}, indent=2))
    print(f"Pinned {len(fixed)} examples (seed={args.seed}) -> {out_path}")
    print("Use this exact file (not a re-sample) for BOTH vLLM version generation runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
