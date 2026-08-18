# Neural Thickets §5.2 Reproduction

Reproducing (only) the RandOpt experiment from §5.2 of *Neural Thickets: Diverse Task
Experts Are Dense Around Pretrained Weights* (Yulu Gan & Phillip Isola, arXiv:2603.12228,
ICML 2026 Spotlight): Qwen2.5-VL-3B-Instruct on GQA, base 56.6% → RandOpt (N=5000, K=50)
69.0%. No WACV-extension work (layer localization, CoRP, distillation, routing, iterative
RandOpt, etc.) belongs here.

See `REPRO_SPEC.md` for the full Confirmed/UNRESOLVED specification table.

## Status: Gate 0 + Gate 1 complete, Gate 2 prepped for GPU execution (not yet run)

This local machine has no CUDA GPU — nothing that touches the real model/dataset can run
here; everything below runs on a RunPod pod (1x L40S 48GB) instead (see `RUNPOD_SETUP.md`).

**Branch structure**: `neural-thickets-repro-gate1-prep` is preserved as the complete Gate 1
reproduction/forensics record (image-blindness diagnosis, the residual-gap investigation,
the vLLM-version-control experiment). **This branch (`neural-thickets-repro-gate2-prep`)**
starts from commit `c779ede` on that branch — the canonical **54.19% march-era-scoring
image-aware baseline** (`results/base_image_aware/`) — and does not carry the later forensic
commits forward. Gate 1 is accepted at that number: a paper-faithful reconstruction of the
released method (image-blindness fixed, the one confirmed root cause) with a documented,
unrecoverable runtime-version discrepancy against the published 56.6% (see the other
branch's `GATE1_DIAGNOSIS.md` for the full residual-gap forensics).

Run `python -m neural_thickets_repro.validate_env` to see exactly what's blocked and why on
whatever machine you run it on.

## Pipeline gates

1. **Gate 0 — Specification & scaffold** (CPU-only): `REPRO_SPEC.md`, config, environment
   gate, and pure logic (perturbation math, top-K, voting, resumable ledger) tested against
   synthetic data.
2. **Gate 1 — Baseline, image-aware** (GPU, **done**): `eval_base_image_aware.py` —
   confirmed root cause was that the released code never attaches images to the vLLM
   request (`GATE1_DIAGNOSIS.md`); this adapter fixes only that, nothing else. Result:
   54.19% march-era scoring on the full 12,578-example testdev-balanced set vs published
   56.6% (−2.41pp, accepted with a documented runtime-version discrepancy). The original
   `eval_base.py` (subprocess-wraps unmodified `randopt.py`) is kept only as a reference for
   what the literal released code does if run as-is (17.94%, image-blind).
3. **Gate 2 — Small-scale RandOpt, image-aware** (GPU, **prepared, not executed**):
   `run_randopt_image_aware.py --N 20 --K 5 --sigma-candidate <name> --restoration-mode <mode>`
   — real end-to-end mechanics check (perturb → select → ensemble → vote, all via the
   unmodified external `WorkerExtension`/`launch_engines`/`GQAHandler`, images attached the
   same way Gate 1's adapter does) against each candidate sigma set as a labeled sensitivity
   config, not a search for "the" right sigma. `--restoration-mode` is required, never
   defaulted: `diagnostics/gate2_restoration_ab.py`'s A/B evidence found `released_compat`
   (`perturb_self_weights`/`restore_self_weights`) leaves real residual drift across repeated
   cycles while `fixed_base` (`apply_perturbation`/`reset_to_base_weights`) restores exactly
   — both are real unmodified `WorkerExtension` mechanisms and neither replaces the other:
   `released_compat` for paper-code reproduction, `fixed_base` for WACV-extension experiments
   (see REPRO_SPEC.md "Gate 2 restoration semantics"). `run_randopt.py`
   (unmodified-`randopt.py` subprocess wrap) is superseded for real use — it would inherit
   the identical image-blindness.
4. **Gate 3 — Full N=5000, K=50** (GPU, not started, requires Gate 2 reviewed first):
   `run_randopt_image_aware.py --sigma-candidate <name> --restoration-mode <mode>` (defaults
   to the paper's N/K; restoration mode still required explicitly).

## Running Gate 0 (works now)

```bash
pip install -r requirements/requirements-cpu.txt
cd research/neural_thickets_repro
pytest tests/                                    # picks up src/ automatically via pyproject.toml
PYTHONPATH=src python -m neural_thickets_repro.validate_env   # or: pip install -e . first, then drop PYTHONPATH
```

## Running Gates 1-3 (needs a CUDA GPU + the official repo cloned)

```bash
pip install -e .   # or: export PYTHONPATH=src (Windows: set PYTHONPATH=src)
pip install -r requirements/requirements-gpu.txt
python external/setup_external_repo.py   # clones sunrainyg/RandOpt at a pinned commit; never committed
python -m neural_thickets_repro.prepare_gqa_data --config configs/gqa_repro.yaml
python -m neural_thickets_repro.eval_base_image_aware --config configs/gqa_repro.yaml   # Gate 1, already run -> 54.19%
python -m neural_thickets_repro.run_randopt_image_aware --config configs/gqa_repro.yaml --N 20 --K 5 --sigma-candidate sigma_default --restoration-mode released_compat   # Gate 2 smoke test, paper-code reproduction
python -m neural_thickets_repro.run_randopt_image_aware --config configs/gqa_repro.yaml --N 20 --K 5 --sigma-candidate sigma_default --restoration-mode fixed_base        # Gate 2 smoke test, WACV-ready mechanics
python -m neural_thickets_repro.run_randopt_image_aware --config configs/gqa_repro.yaml --sigma-candidate sigma_default --restoration-mode <mode>   # Gate 3, full N=5000/K=50, only after Gate 2 is reviewed
python -m neural_thickets_repro.report --config configs/gqa_repro.yaml
```

`REPRODUCTION_REPORT.md` is a later-phase deliverable, produced only once Gate 3 has real
results — not part of this Gate-0 scaffold.

## Reuse / licensing note

The official RandOpt repo (`github.com/sunrainyg/RandOpt`) has no declared license. Its
code is read, described, and invoked as an external subprocess (`external/`, gitignored,
pinned to a commit SHA — see `external/EXTERNAL_COMMIT.txt`), but never copied or
transcribed into this repository, including in `REPRO_SPEC.md`.
