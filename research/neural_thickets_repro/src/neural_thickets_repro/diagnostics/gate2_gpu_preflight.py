"""Gate 2 GPU preflight -- run once before committing to a real N=20 candidate loop. GPU
required. Does not redesign RandOpt; a pure sanity check on the REAL mechanism.

Complements (does not replace) Gate 0's CPU-only perturb/restore tests
(tests/test_perturb_cpu.py), which validated the pure-Python math against a synthetic
model. This validates the REAL end-to-end mechanism on actual GPU hardware, at real bf16
precision, against the real checkpoint: core/engine.py:launch_engines (unmodified),
utils/worker_extn.py:WorkerExtension.perturb_self_weights/restore_self_weights (unmodified,
invoked via Ray collective_rpc), and the image-aware generation path
(vlm_adapter.build_image_aware_requests) -- together, not each piece in isolation.

Also serves as the empirical half of the cache-safety review (see
GATE2_CACHE_SAFETY_REVIEW.md for the static half): step 3 below ("verify at least one
output changed after perturbation") is a live proof that no cache is masking the weight
change -- if vLLM's prefix cache (or anything else) were incorrectly reusing state computed
under the base weights, that is exactly the check that would catch it.

Steps, on one small fixed batch (first N_PREFLIGHT_EXAMPLES of the selection set):
  1. generate under base (unperturbed) weights
  2. perturb ONE fixed test candidate (not a real Gate-2 candidate -- a separate, clearly
     labeled seed/sigma used only for this check)
  3. regenerate the SAME batch -- FAIL if nothing changed (perturbation not taking effect,
     or something caching around it)
  4. restore weights (same seed/sigma)
  5. regenerate the SAME batch again -- expect an EXACT match to the base-weights output
     under greedy decoding + enable_prefix_caching=False; if not exact, report the mismatch
     explicitly rather than silently passing or hard-failing -- Gate 0 already established
     that a single perturb/restore cycle is only exact to ~1 ULP at the weight level
     (tests/test_perturb_cpu.py), and text-level determinism additionally depends on
     whether any logit came out close enough to a tie for that to flip the argmax token.

Usage:
    python -m neural_thickets_repro.diagnostics.gate2_gpu_preflight --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # same runtime fix as Gate 1/2 -- see eval_base_image_aware.py

from .config import load_config
from .env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module
from .vlm_adapter import build_image_aware_requests, resolve_model_snapshot

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "external" / "RandOpt"

N_PREFLIGHT_EXAMPLES = 5
TEST_SEED = 999_999_999  # fixed, clearly not drawn from any real candidate's RNG stream
TEST_SIGMA = 0.01  # deliberately large-ish so a real behavior change is easy to detect


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "gate2_diagnosis" / "gpu_preflight_report.json"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Gate 2 GPU preflight",
            [check_cuda(), check_module("vllm"), check_module("ray"), check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        return 1

    cfg.require_resolved("model.revision", "dataset.selection_split", "dataset.test_split")

    model_path = resolve_model_snapshot(cfg.model.name, cfg.model.revision)
    print(f"Resolved {cfg.model.name}@{cfg.model.revision} -> {model_path}")

    import ray
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    sys.path.insert(0, str(EXTERNAL_ROOT))
    from core.engine import cleanup_engines, launch_engines  # type: ignore
    from data_handlers.gqa import GQAHandler  # type: ignore

    handler = GQAHandler()
    task_datas = handler.load_data(
        str(EXTERNAL_ROOT / "data" / "gqa" / "train.parquet"), split="train", max_samples=None,
    )
    task_datas = [d for d in task_datas if "image_path" in d][:N_PREFLIGHT_EXAMPLES]
    if len(task_datas) < N_PREFLIGHT_EXAMPLES:
        print(f"WARNING: only {len(task_datas)} examples with images available for the preflight batch (wanted {N_PREFLIGHT_EXAMPLES})", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    requests = build_image_aware_requests(task_datas, tokenizer)
    sampling_params = SamplingParams(temperature=0.0, seed=cfg.reproducibility.global_seed, max_tokens=cfg.evaluation.max_tokens)

    # enable_prefix_caching intentionally NOT overridden here -- launch_engines' own default
    # is False; see GATE2_CACHE_SAFETY_REVIEW.md for why that's what makes this check valid.
    engines, pgs = launch_engines(1, model_path, precision=cfg.model.precision, tensor_parallel_size=1, multimodal=True)
    engine = engines[0]

    def generate() -> List[str]:
        outputs = ray.get(engine.generate.remote(requests, sampling_params, use_tqdm=False))
        return [o.outputs[0].text for o in outputs]

    try:
        print(f"1. Generating under base weights (n={len(task_datas)})...")
        base_outputs = generate()

        print(f"2. Perturbing test candidate (seed={TEST_SEED}, sigma={TEST_SIGMA})...")
        ray.get(engine.collective_rpc.remote("perturb_self_weights", args=(TEST_SEED, TEST_SIGMA, False)))

        print("3. Regenerating under perturbed weights...")
        perturbed_outputs = generate()

        n_changed = sum(1 for a, b in zip(base_outputs, perturbed_outputs) if a.strip() != b.strip())
        step3_pass = n_changed > 0
        print(f"   {n_changed}/{len(task_datas)} outputs changed after perturbation -- {'PASS' if step3_pass else 'FAIL'}")

        print(f"4. Restoring weights (seed={TEST_SEED}, sigma={TEST_SIGMA})...")
        ray.get(engine.collective_rpc.remote("restore_self_weights", args=(TEST_SEED, TEST_SIGMA, False)))

        print("5. Regenerating under restored weights...")
        restored_outputs = generate()

        n_mismatched_after_restore = sum(1 for a, b in zip(base_outputs, restored_outputs) if a.strip() != b.strip())
        step5_exact_pass = n_mismatched_after_restore == 0
        print(f"   {len(task_datas) - n_mismatched_after_restore}/{len(task_datas)} outputs exactly match base after restore "
              f"-- {'PASS (exact)' if step5_exact_pass else 'MISMATCH -- see report for details, not necessarily fatal (see docstring)'}")
    finally:
        cleanup_engines(engines, pgs)

    overall_pass = step3_pass and step5_exact_pass
    report = {
        "n_examples": len(task_datas),
        "test_seed": TEST_SEED,
        "test_sigma": TEST_SIGMA,
        "step3_output_changed_after_perturbation": {
            "pass": step3_pass, "n_changed": n_changed, "n_total": len(task_datas),
        },
        "step5_output_matches_base_after_restore": {
            "exact_pass": step5_exact_pass, "n_mismatched": n_mismatched_after_restore, "n_total": len(task_datas),
        },
        "overall": "PASS" if overall_pass else "FAIL",
        "detail": [
            {
                "example_id": str(d["question_id"]),
                "base_output": base_outputs[i],
                "perturbed_output": perturbed_outputs[i],
                "restored_output": restored_outputs[i],
                "changed_after_perturbation": base_outputs[i].strip() != perturbed_outputs[i].strip(),
                "matches_base_after_restore": base_outputs[i].strip() == restored_outputs[i].strip(),
            }
            for i, d in enumerate(task_datas)
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nOVERALL: {report['overall']}")
    if not overall_pass:
        print(
            "Do not proceed to N=20 until this passes. If step 3 failed (no change detected "
            "after perturbation), that is exactly the signature of a cache-masking bug -- "
            "re-check GATE2_CACHE_SAFETY_REVIEW.md's assumptions on this specific environment "
            "before assuming it's something else. If only step 5 failed (restore doesn't "
            "exactly match base), inspect the mismatched examples in the report -- a small "
            "number of near-tie token flips is a different, lower-severity finding than "
            "systematic divergence across most/all examples.",
            file=sys.stderr,
        )
    print(f"Wrote {out_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
