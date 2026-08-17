"""Gate 2/3 entrypoint:
    python -m neural_thickets_repro.run_randopt --config configs/gqa_repro.yaml \
        --sigma-candidate sigma_default [--N 20 --K 5]

Invokes the external RandOpt repo's randopt.py directly via subprocess. Enforces the gate
sequence from REPRO_SPEC.md / the project plan:
  - refuses to start unless Gate 1's results/base/metrics.json exists (require_gate1_before_gate2)
  - refuses to start a full N/K run unless a Gate 2 small-scale result exists (require_gate2_before_gate3)
Sigma is never silently defaulted: --sigma-candidate must name one of
config.randopt.sigma_candidates (see REPRO_SPEC.md "Sigma - resolution plan"), and the run
is labeled with that name in its output directory, so results are never presented as "the"
reproduced sigma without saying which candidate assumption produced them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .env_check import (
    GateBlockedError,
    assert_feasible,
    check_cuda,
    check_disk,
    check_gate_artifact,
    check_module,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_RANDOPT = REPO_ROOT / "external" / "RandOpt" / "randopt.py"
GATE1_ARTIFACT = REPO_ROOT / "results" / "base" / "metrics.json"
GATE2_ARTIFACT = REPO_ROOT / "results" / "randopt_smoke" / "results.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--N", type=int, default=None, help="override population size (Gate 2 smoke test)")
    parser.add_argument("--K", type=int, default=None, help="override top-K (Gate 2 smoke test)")
    parser.add_argument(
        "--sigma-candidate",
        required=True,
        help="name of a config.randopt.sigma_candidates entry -- required, never defaulted",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.sigma_candidate not in cfg.randopt.sigma_candidates:
        print(
            f"--sigma-candidate must be one of {sorted(cfg.randopt.sigma_candidates)}, "
            f"got {args.sigma_candidate!r}. Sigma is a first-class unresolved reproduction "
            f"variable (see REPRO_SPEC.md) -- it is never silently defaulted.",
            file=sys.stderr,
        )
        return 1
    sigma_values = cfg.randopt.sigma_candidates[args.sigma_candidate]

    N = args.N if args.N is not None else cfg.randopt.N
    K = args.K if args.K is not None else cfg.randopt.K
    is_full_run = (N == cfg.randopt.N and K == cfg.randopt.K)

    checks = [
        check_cuda(),
        check_module("vllm"),
        check_module("ray"),
        check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb),
    ]
    if cfg.gates.require_gate1_before_gate2:
        checks.append(check_gate_artifact(GATE1_ARTIFACT))
    if is_full_run and cfg.gates.require_gate2_before_gate3:
        checks.append(check_gate_artifact(GATE2_ARTIFACT))

    try:
        assert_feasible("run_randopt", checks)
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

    cfg.require_resolved("model.revision", "dataset.selection_split", "dataset.test_split")

    label = f"N{N}_K{K}_{args.sigma_candidate}"
    results_dir = REPO_ROOT / "results" / ("randopt_smoke" if not is_full_run else f"randopt_{label}")
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EXTERNAL_RANDOPT),
        "--dataset", "gqa",
        "--model_name", cfg.model.name,
        "--precision", cfg.model.precision,
        "--train_samples", str(cfg.dataset.selection_set_size),
        "--population_size", str(N),
        "--top_k_ratios", str(K / N),
        "--sigma_values", ",".join(str(s) for s in sigma_values),
        "--max_tokens", str(cfg.evaluation.max_tokens),
        "--global_seed", str(cfg.reproducibility.global_seed),
        "--experiment_dir", str(results_dir),
    ]
    print(f"Sigma candidate: {args.sigma_candidate} = {sigma_values}  (UNRESOLVED assumption, see REPRO_SPEC.md)")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
