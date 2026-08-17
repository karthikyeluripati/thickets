# results/

Empty until Gate 1 runs on real GPU hardware. Populated by:

- `results/base/` — Gate 1 (`eval_base.py`): `metrics.json`, `predictions.jsonl`, `run_metadata.json`
- `results/randopt_smoke/` — Gate 2 small-scale run
- `results/randopt_N5000_K50_<sigma_candidate>/` — Gate 3 full run, labeled by which sigma candidate (see `REPRO_SPEC.md`) was used, since sigma is an unresolved reproduction assumption and results must never be presented without saying which candidate produced them
- `results/parameter_manifest.json` — real-checkpoint parameter manifest, generated once the model is loadable on GPU hardware

`*.json`/`*.jsonl` files under this directory are gitignored (see root `.gitignore`); only this README is committed.
