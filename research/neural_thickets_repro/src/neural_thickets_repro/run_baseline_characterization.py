"""Baseline-characterization orchestrator: runs the Capability Benchmark Gate's fixed N=200
base-model evaluation (+ optional repeat/image-sanity) across the 7 currently-accessible
capabilities in ONE process, reusing a single vLLM engine across all of them instead of
paying its ~20s+ engine-init cost (weight load + CUDA graph capture) once per capability.

This establishes S_t(theta_0) -- the base-model, zero-perturbation measurement for every
capability -- BEFORE any RandOpt/perturbation sweep. It does NOT run RandOpt, does NOT
perturb weights, and does NOT change any capability/benchmark DEFINITION: it is a thin driver
around run_capability_benchmark_gate.run_one_capability(), which already implements the
per-capability load/subset/integrity/generate/score/card pipeline -- nothing here duplicates
or second-guesses that logic.

DEFAULT_CAPABILITY_CONFIGS excludes:
  - object_recognition.yaml (ImageNet-1K) -- blocked by gated HF access, not a code issue,
    never substituted (see CAPABILITY_BENCHMARK_GATE.md).
  - ocr_text_recognition.yaml (the FULL, non-OCR-grounded TextVQA capability) -- the paper's
    OCR/text-recognition claim uses ocr_text_recognition_grounded instead (see
    CAPABILITY_BENCHMARK_GATE.md's OCR-grounded-subset section); the full-TextVQA capability
    still exists and is runnable on its own via run_capability_benchmark_gate.py directly,
    just not part of this baseline-characterization default set.

Usage:
    # cheap: validate every config's data loading + integrity, no GPU/model call at all
    python -m neural_thickets_repro.run_baseline_characterization --dry-run

    # the real baseline-characterization run (N=200, repeat, image-sanity, one shared engine)
    python -m neural_thickets_repro.run_baseline_characterization --repeat --image-sanity

    # a cheap N=5 smoke test across all 7 capabilities before spending real GPU time on N=200
    python -m neural_thickets_repro.run_baseline_characterization --subset-size 5 --repeat --image-sanity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .benchmarks.summary import write_summary
from .config import CapabilityBenchmarkConfig, load_capability_benchmark_config
from .run_capability_benchmark_gate import build_output_dir, run_one_capability, runtime_versions

REPO_ROOT = Path(__file__).resolve().parents[2]

# Distinct from rc=1 (an ORDINARY, already-clearly-reported failure inside run_one_capability
# -- integrity check failed, or env_check.assert_feasible blocked the run) -- rc=2 means the
# shared engine or a generation call itself raised an uncaught exception (e.g. a fatal vLLM
# EngineCore error) partway through a capability. No auto-restart is attempted: the process
# stops here, and every capability directory already written before the crash is left
# untouched on disk (each capability's own outputs are fully written before the loop moves
# on to the next one, so a crash mid-capability never partially corrupts a PRIOR capability's
# already-completed results).
EXIT_CODE_CAPABILITY_CRASHED = 2

# The 7 capabilities currently accessible for baseline characterization (see module
# docstring for why object_recognition/ocr_text_recognition are excluded from this default
# set -- both remain independently runnable via run_capability_benchmark_gate.py).
DEFAULT_CAPABILITY_CONFIGS: List[str] = [
    "visual_grounding.yaml",
    "counting.yaml",
    "spatial_reasoning.yaml",
    "ocr_text_recognition_grounded.yaml",
    "attribute_recognition.yaml",
    "relational_reasoning.yaml",
    "fine_grained_recognition.yaml",
]


class ModelMismatchError(RuntimeError):
    """Two capability configs in this run point at different models/revisions/precisions --
    refuses to silently reuse one engine across genuinely different models.
    """


def resolve_config_paths(config_names: Optional[List[str]] = None) -> List[Path]:
    """`config_names` may be bare filenames (resolved under configs/benchmarks/, the default
    set's own convention) or full paths -- kept flexible for a caller that wants to point at
    a config living elsewhere, without requiring it.
    """
    names = config_names if config_names is not None else DEFAULT_CAPABILITY_CONFIGS
    paths = []
    for name in names:
        candidate = Path(name)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = REPO_ROOT / "configs" / "benchmarks" / name
        paths.append(candidate)
    return paths


def _assert_same_model(configs: List[CapabilityBenchmarkConfig]) -> None:
    """All configs in a single shared-engine run must pin the identical model -- this is
    what makes reusing one vLLM engine across capabilities valid in the first place, not an
    assumption to silently skip checking.
    """
    if not configs:
        return
    first = configs[0].model
    for cfg in configs[1:]:
        if (cfg.model.name, cfg.model.revision, cfg.model.precision) != (first.name, first.revision, first.precision):
            raise ModelMismatchError(
                f"Capability configs pin different models -- cannot share one vLLM engine: "
                f"{first.name}@{first.revision} ({first.precision}) vs "
                f"{cfg.model.name}@{cfg.model.revision} ({cfg.model.precision}). "
                f"Run these separately via run_capability_benchmark_gate.py instead."
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", default=None,
        help="comma-separated capability config names/paths to run; default is the 7 accessible capabilities (see module docstring)",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "baseline_characterization"))
    parser.add_argument("--subset-size", type=int, default=None, help="override every config's dataset.subset_size (e.g. for a cheap N=5 smoke test across all capabilities)")
    parser.add_argument("--repeat", action="store_true", help="run a second identical pass per capability to check repeatability")
    parser.add_argument("--image-sanity", action="store_true", help="run the correct/shuffled/text-only image-dependence sanity check per capability")
    parser.add_argument("--dry-run", action="store_true", help="load examples / build subsets / run integrity checks for every capability -- no model call, no GPU needed")
    args = parser.parse_args(argv)

    config_names = args.configs.split(",") if args.configs else None
    config_paths = resolve_config_paths(config_names)

    print(f"=== Baseline characterization: {len(config_paths)} capabilities ===")
    for p in config_paths:
        print(f"  - {p}")

    configs = [load_capability_benchmark_config(p) for p in config_paths]
    for cfg in configs:
        cfg.require_resolved("model.revision", "dataset.split")

    if not args.dry_run:
        _assert_same_model(configs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime_versions.json").write_text(json.dumps(runtime_versions(), indent=2))

    llm = None
    tokenizer = None
    results: Dict[str, int] = {}
    engine_reused_count = 0

    for cfg, cfg_path in zip(configs, config_paths):
        capability_out_dir = build_output_dir(out_dir, cfg.dataset.capability)
        print(f"\n{'=' * 80}\n{cfg.dataset.capability} ({cfg_path.name})\n{'=' * 80}")

        engine_was_already_loaded = llm is not None
        try:
            rc, llm, tokenizer = run_one_capability(
                cfg, subset_size=args.subset_size, out_dir=capability_out_dir,
                repeat=args.repeat, image_sanity=args.image_sanity, dry_run=args.dry_run,
                llm=llm, tokenizer=tokenizer,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any uncaught exception
            # (a fatal vLLM EngineCore error, or anything else) must fail CLEARLY here rather
            # than propagate as a bare traceback with no indication of which capability was
            # running or that earlier capabilities' results are still intact. No auto-restart
            # of the (now possibly dead) shared engine is attempted -- see EXIT_CODE_CAPABILITY_CRASHED.
            completed = sorted(c for c, code in results.items() if code == 0)
            print(
                f"\nFATAL: capability {cfg.dataset.capability!r} ({cfg_path.name}) crashed during "
                f"generation: {type(exc).__name__}: {exc}\n"
                f"Previously completed capabilities ({len(completed)}: {completed}) remain intact "
                f"under {out_dir} -- stopping here, not attempting to continue or auto-restart "
                f"the shared engine.",
                file=sys.stderr,
            )
            return EXIT_CODE_CAPABILITY_CRASHED
        if engine_was_already_loaded and llm is not None:
            engine_reused_count += 1
        results[cfg.dataset.capability] = rc
        if rc != 0:
            print(f"STOP: {cfg.dataset.capability} failed (rc={rc}) -- not proceeding to remaining capabilities.", file=sys.stderr)
            return rc

    if not args.dry_run:
        print(f"\nShared vLLM engine reused across {engine_reused_count}/{max(len(configs) - 1, 0)} subsequent capabilities (loaded once, not once per capability).")

    if args.dry_run:
        print("\n--dry-run: skipping summary generation (no cards were written).")
        return 0

    # expected_capabilities = every capability THIS run actually attempted (not just the
    # ones with rc==0) -- guarantees every one of them gets a summary.md row, either with
    # real data or an explicit "MISSING" marker, rather than an attempted capability's row
    # silently vanishing if its card.json somehow wasn't written (see summary.py's own
    # MISSING-CAPABILITY ROBUSTNESS docstring note for the real incident this fixes).
    expected_capabilities = [cfg.dataset.capability for cfg in configs]
    summary = write_summary(out_dir, expected_capabilities=expected_capabilities)
    print(f"\n=== Baseline characterization summary: {summary['n_capabilities']}/{summary['n_expected_capabilities']} capabilities, {summary['status_counts']} ===")
    if summary["missing_capabilities"]:
        print(f"WARNING: missing capabilities (attempted but no card.json found): {summary['missing_capabilities']}", file=sys.stderr)
    print(f"Wrote {out_dir}/summary.md and {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
