"""Gate 1 HARD-FAIL diagnostic (baseline 17.94% vs published 56.6%). GPU required.

Diagnosis only -- changes nothing about the reproduction pipeline, tunes nothing.

Hypothesis under test (from static analysis of upstream at every git revision including the
March 2026 paper-era commit): the released RandOpt code formats GQA prompts into plain text
strings via tokenizer.apply_chat_template -- which inserts <|vision_start|><|image_pad|>
<|vision_end|> placeholders -- but calls vLLM generate() without ever constructing
multi_modal_data, so THE IMAGES NEVER REACH THE MODEL and it answers blind. A blind model
on GQA is consistent with ~18%.

What this script does, on a seeded sample of the prepared testdev data:
  Path A ("upstream-replica"): text-only generation exactly as randopt.py does it.
    Expected to reproduce the ~18% regime if the hypothesis is right.
  Path B ("multimodal-control"): identical prompts but with the actual image passed as
    vLLM multi_modal_data. If accuracy jumps toward ~56%, the missing-image path is
    confirmed as the root cause.
  Both paths are scored two ways:
    - "head" scoring: extract_answer_for_voting -> is_voted_answer_correct (what our
      pinned HEAD's base eval does, post-May-2026 PR #4)
    - "march" scoring: compute_reward > 0 (what the March 2026 paper-era base eval did --
      more lenient: it also whole-word-scans the first raw response line)
  Per-example records (question id, image id/path, question, GT, raw response, extracted
  answer, correctness) go to a JSONL; failures are classified as model-answer failures
  (GT nowhere in the response) vs extraction/scoring failures (GT present by whole-word
  scan of the full response but scored wrong).
  Also verifies N question<->image pairs against a freshly-loaded copy of the HF source
  dataset (question/answer/imageId match + image dimensions match the original).

Usage (on the pod):
    python -m neural_thickets_repro.diagnostics.gate1_failure_audit \
        --config configs/gqa_repro.yaml --sample-size 100 --verify-pairs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

from ..config import load_config
from ..vlm_adapter import resolve_model_snapshot

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

DATASET_NAME = "lmms-lab-encoder/GQA"


def _load_handler():
    sys.path.insert(0, str(EXTERNAL_ROOT))
    from data_handlers.gqa import GQAHandler  # type: ignore

    return GQAHandler()


def _format_prompt(tokenizer, messages) -> str:
    # Byte-for-byte what upstream randopt.py's format_prompt does for an instruct model.
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _score_both_ways(handler, response_text: str, ground_truth) -> Dict:
    extracted = handler.extract_answer_for_voting(response_text) or ""
    head_correct = bool(extracted) and handler.is_voted_answer_correct(extracted, ground_truth)
    march_correct = handler.compute_reward(response_text, ground_truth) > 0
    return {
        "extracted_answer": extracted,
        "correct_head_scoring": bool(head_correct),
        "correct_march_scoring": bool(march_correct),
    }


def _classify_failure(handler, response_text: str, ground_truth) -> str:
    """For an example scored incorrect under HEAD scoring: if the GT answer appears
    (whole-word, incl. singular/synonym forms) anywhere in the raw response, the model
    plausibly answered right and extraction/scoring lost it; otherwise the model itself
    answered wrong.
    """
    gt_answer = ground_truth.get("answer", "") if isinstance(ground_truth, dict) else str(ground_truth)
    gt_norm = handler._normalize_answer(gt_answer)
    if handler._whole_word_search(response_text.lower(), gt_norm):
        return "extraction_or_scoring_failure"
    return "model_answer_failure"


def run_generation(llm, sampling_params, task_datas, tokenizer, with_images: bool) -> List[str]:
    from PIL import Image

    inputs = []
    for d in task_datas:
        text = _format_prompt(tokenizer, d["messages"])
        if with_images:
            image = Image.open(d["image_path"]).convert("RGB")
            inputs.append({"prompt": text, "multi_modal_data": {"image": image}})
        else:
            inputs.append(text)
    outputs = llm.generate(inputs, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


def verify_pairs(task_datas, sample_indices, images_dir: Path, n_pairs: int, seed: int) -> Dict:
    """Checks question/answer/imageId against a fresh HF load, and image dimensions
    against the original HF image bytes.
    """
    from datasets import load_dataset
    from PIL import Image

    instructions = load_dataset(DATASET_NAME, "testdev_balanced_instructions", split="testdev")
    by_qid = {row["id"]: row for row in instructions}

    rng = random.Random(seed)
    chosen = rng.sample(list(sample_indices), min(n_pairs, len(sample_indices)))
    needed_image_ids = {str(task_datas[i]["question_id"]): str(Path(task_datas[i]["image_path"]).stem) for i in chosen}

    hf_image_sizes = {}
    images = load_dataset(DATASET_NAME, "testdev_balanced_images", split="testdev")
    wanted = set(needed_image_ids.values())
    for row in images:
        if row["id"] in wanted:
            hf_image_sizes[row["id"]] = tuple(row["image"].size)
            if len(hf_image_sizes) == len(wanted):
                break

    results = []
    for i in chosen:
        d = task_datas[i]
        qid = str(d["question_id"])
        local_image_id = str(Path(d["image_path"]).stem)
        hf_row = by_qid.get(qid)
        entry = {
            "question_id": qid,
            "found_in_source": hf_row is not None,
            "question_matches": None,
            "answer_matches": None,
            "image_id_matches": None,
            "image_dims_match": None,
        }
        if hf_row is not None:
            entry["question_matches"] = (hf_row["question"] == d["messages"][0]["content"][1]["text"].split("Question: ")[1].split("\n\n")[0])
            entry["answer_matches"] = (str(hf_row["answer"]).strip().lower() == d["ground_truth"]["answer"])
            entry["image_id_matches"] = (str(hf_row["imageId"]) == local_image_id)
            local_size = Image.open(d["image_path"]).size
            hf_size = hf_image_sizes.get(local_image_id)
            entry["image_dims_match"] = (hf_size is not None and tuple(local_size) == hf_size)
        results.append(entry)

    all_ok = all(
        e["found_in_source"] and e["question_matches"] and e["answer_matches"]
        and e["image_id_matches"] and e["image_dims_match"]
        for e in results
    )
    return {"pairs_checked": len(results), "all_ok": all_ok, "details": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--data-dir", default=str(EXTERNAL_ROOT / "data" / "gqa"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--verify-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-path-a", action="store_true", help="skip the upstream-replica text-only generation")
    parser.add_argument("--skip-path-b", action="store_true", help="skip the multimodal-control generation")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "gate1_diagnosis"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handler = _load_handler()
    testdev_path = str(Path(args.data_dir) / "testdev.parquet")
    task_datas = handler.load_data(testdev_path, split="test", max_samples=None)

    with_images = [i for i, d in enumerate(task_datas) if "image_path" in d]
    if len(with_images) != len(task_datas):
        print(f"WARNING: only {len(with_images)}/{len(task_datas)} examples have image_path -- "
              f"handler fell back to text-only for the rest; that itself is diagnostic signal.")

    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(with_images, min(args.sample_size, len(with_images))))
    sample = [task_datas[i] for i in sample_indices]
    print(f"Sampled {len(sample)} examples (seed={args.seed}) from {len(task_datas)} testdev examples.")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Model snapshot: {model_path}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype=cfg.model.precision,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.evaluation.max_tokens
    )

    report: Dict = {
        "hypothesis": "released code passes text-only prompts to vLLM; images never reach the model",
        "sample_size": len(sample),
        "seed": args.seed,
        "model_snapshot": model_path,
        "paths": {},
    }

    records_by_path: Dict[str, List[Dict]] = {}
    for path_name, use_images, skip in (
        ("A_upstream_replica_text_only", False, args.skip_path_a),
        ("B_multimodal_control", True, args.skip_path_b),
    ):
        if skip:
            continue
        print(f"\n=== Generating: {path_name} ===")
        responses = run_generation(llm, sampling_params, sample, tokenizer, with_images=use_images)

        records = []
        head_correct = march_correct = 0
        failure_counts = {"model_answer_failure": 0, "extraction_or_scoring_failure": 0}
        for d, resp in zip(sample, responses):
            scored = _score_both_ways(handler, resp, d["ground_truth"])
            rec = {
                "question_id": str(d["question_id"]),
                "image_id": Path(d["image_path"]).stem,
                "image_path": d["image_path"],
                "question": d["messages"][0]["content"][1]["text"],
                "ground_truth_answer": d["ground_truth"]["answer"],
                "raw_response": resp,
                **scored,
            }
            if scored["correct_head_scoring"]:
                head_correct += 1
            else:
                rec["failure_class"] = _classify_failure(handler, resp, d["ground_truth"])
                failure_counts[rec["failure_class"]] += 1
            if scored["correct_march_scoring"]:
                march_correct += 1
            records.append(rec)

        records_by_path[path_name] = records
        n = len(records)
        report["paths"][path_name] = {
            "accuracy_head_scoring": head_correct / n,
            "accuracy_march_scoring": march_correct / n,
            "n": n,
            "failure_classification_of_head_incorrect": failure_counts,
        }
        print(f"{path_name}: head-scoring {head_correct}/{n} = {head_correct/n:.1%} | "
              f"march-scoring {march_correct}/{n} = {march_correct/n:.1%} | failures: {failure_counts}")

        jsonl_path = out_dir / f"sample_{path_name}.jsonl"
        with jsonl_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {jsonl_path}")

    print(f"\n=== Verifying {args.verify_pairs} question<->image pairs against {DATASET_NAME} ===")
    report["pair_verification"] = verify_pairs(
        task_datas, sample_indices, Path(args.data_dir) / "images", args.verify_pairs, args.seed
    )
    print(f"pair verification all_ok: {report['pair_verification']['all_ok']}")

    report_path = out_dir / "gate1_failure_audit.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    print("\nSummary:")
    print(json.dumps({k: v for k, v in report.items() if k != "pair_verification"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
