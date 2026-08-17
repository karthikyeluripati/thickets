"""Gate 1 entrypoint: python -m neural_thickets_repro.eval_base --config configs/gqa_repro.yaml

Invokes the external RandOpt repo's randopt.py with --population_size 0, which is a
zero-modification way to get base-model-only accuracy from the official script: with
population_size=0, run_sampling's candidate loop and run_ensemble_evaluation's batch loop
both become no-ops (verified by reading randopt.py's control flow -- see REPRO_SPEC.md), so
the only real work performed is evaluate_base_model(). Requires a CUDA GPU + vllm + ray;
refuses to start otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .env_check import GateBlockedError, assert_feasible, check_cuda, check_disk, check_module

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_RANDOPT = REPO_ROOT / "external" / "RandOpt" / "randopt.py"
EXTERNAL_RANDOPT_ROOT = EXTERNAL_RANDOPT.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        assert_feasible(
            "Gate 1 (eval_base)",
            [
                check_cuda(),
                check_module("vllm"),
                check_module("ray"),
                check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb),
            ],
        )
    except GateBlockedError as e:
        print(str(e), file=sys.stderr)
        print("Run `python -m neural_thickets_repro.validate_env` for the full picture.", file=sys.stderr)
        return 1

    if not EXTERNAL_RANDOPT.exists():
        print(
            f"{EXTERNAL_RANDOPT} not found. Run `python external/setup_external_repo.py` first.",
            file=sys.stderr,
        )
        return 1

    # sigmas is intentionally NOT required here: population_size=0 never samples a
    # perturbation, so the unresolved sigma question does not block a baseline-only run.
    cfg.require_resolved("model.revision", "dataset.selection_split", "dataset.test_split")

    results_dir = REPO_ROOT / "results" / "base"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EXTERNAL_RANDOPT),
        "--dataset", "gqa",
        "--model_name", cfg.model.name,
        "--precision", cfg.model.precision,
        "--train_samples", str(cfg.dataset.selection_set_size),
        "--population_size", "0",
        "--max_tokens", str(cfg.evaluation.max_tokens),
        "--global_seed", str(cfg.reproducibility.global_seed),
        "--experiment_dir", str(results_dir),
    ]
    print("Running:", " ".join(cmd))
    # cwd MUST be the external RandOpt repo root: GQAHandler resolves data/gqa/train.parquet
    # etc. relative to the process cwd, not relative to randopt.py's own location, so
    # running with our repo as cwd (the default) fails to find data prepared under
    # external/RandOpt/data/gqa/. We never copy/move/symlink the data to work around this --
    # cwd is the correct fix and preserves the official repo's expected layout.
    subprocess.run(cmd, check=True, cwd=EXTERNAL_RANDOPT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
