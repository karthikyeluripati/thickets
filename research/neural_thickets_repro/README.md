# Neural Thickets §5.2 Reproduction

Reproducing (only) the RandOpt experiment from §5.2 of *Neural Thickets: Diverse Task
Experts Are Dense Around Pretrained Weights* (Yulu Gan & Phillip Isola, arXiv:2603.12228,
ICML 2026 Spotlight): Qwen2.5-VL-3B-Instruct on GQA, base 56.6% → RandOpt (N=5000, K=50)
69.0%. No WACV-extension work (layer localization, CoRP, distillation, routing, iterative
RandOpt, etc.) belongs here.

See `REPRO_SPEC.md` for the full Confirmed/UNRESOLVED specification table.

## Status: Gate 0 complete, Gate 1 prepped for GPU execution, Gates 2-3 not started

This local machine has no CUDA GPU and only ~15GB free disk — nothing that touches the real
model/dataset can run here. Gate 1 (baseline) is prepared to run on a RunPod pod (1x L40S
48GB) instead: see **`RUNPOD_SETUP.md`** for the exact command sequence. As of this pass,
the GQA dataset provenance question is resolved (by documented reproduction assumption, see
`REPRO_SPEC.md`) and validated end-to-end against the real HF dataset, and a perturb/restore
drift diagnostic (`src/neural_thickets_repro/diagnostics/perturb_restore_drift.py`, kept
separate from the reproduction algorithm) is ready to run at real bf16 precision.

Run `python -m neural_thickets_repro.validate_env` to see exactly what's blocked and why on
whatever machine you run it on.

## Pipeline gates

1. **Gate 0 — Specification & scaffold** (this phase, CPU-only): `REPRO_SPEC.md`, config,
   environment gate, and pure logic (perturbation math, top-K, voting, resumable ledger)
   tested against synthetic data. This is **scaffold/unit validation** — it shows the
   scaffold's logic is consistent with our reconstructed specification. It is not proof of
   equivalence to the upstream VLM implementation, which is entirely unexercised until
   Gate 1.
2. **Gate 1 — Baseline** (requires GPU): `eval_base.py` runs the unperturbed model on real
   GQA. Compared to published 56.6% with explicit thresholds: ≤1pp → proceed and document;
   1-3pp → stop and investigate; >3pp → hard stop.
3. **Gate 2 — Small-scale RandOpt on GPU** (requires GPU, requires Gate 1 passed):
   `run_randopt.py --N 20 --K 5 --sigma-candidate <name>` — real end-to-end mechanics check
   against each candidate sigma set as a labeled sensitivity config, not a search for "the"
   right sigma.
4. **Gate 3 — Full N=5000, K=50** (requires GPU, requires Gate 2 present):
   `run_randopt.py --sigma-candidate <name>`.

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
python -m neural_thickets_repro.eval_base --config configs/gqa_repro.yaml
python -m neural_thickets_repro.run_randopt --config configs/gqa_repro.yaml --N 20 --K 5 --sigma-candidate sigma_default
python -m neural_thickets_repro.run_randopt --config configs/gqa_repro.yaml --sigma-candidate sigma_default
python -m neural_thickets_repro.report --config configs/gqa_repro.yaml
```

`REPRODUCTION_REPORT.md` is a later-phase deliverable, produced only once Gate 3 has real
results — not part of this Gate-0 scaffold.

## Reuse / licensing note

The official RandOpt repo (`github.com/sunrainyg/RandOpt`) has no declared license. Its
code is read, described, and invoked as an external subprocess (`external/`, gitignored,
pinned to a commit SHA — see `external/EXTERNAL_COMMIT.txt`), but never copied or
transcribed into this repository, including in `REPRO_SPEC.md`.
