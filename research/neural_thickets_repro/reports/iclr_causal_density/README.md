# reports/iclr_causal_density/

Deliverables for the isolated 7B causal-density pilot (branch `iclr-causal-density-pilot`,
based on commit `9305cc8`). See `decision.md` for the registered outcome.

## Present in this directory

- `artifact_audit.json` / `artifact_audit.md` — Phase 0 audit: no existing repository artifact
  satisfies this pilot's reuse bar; catalogs what code IS reusable.
- `preregistration.md` — Phase 1: the full frozen design, generated from
  `src/neural_thickets_repro/iclr_causal_density/design.py`.
- `decision.json` / `decision.md` — Phase 10: the registered decision
  (`INCONCLUSIVE — EXECUTION BLOCKED`) and the full integrity/headline-evidence record.

## Not present — and why

No GPU execution occurred in this environment (see `decision.md`), so the following
Deliverables the task also lists do not exist yet, honestly:

- **Frozen subset manifest** / **frozen shuffle manifest** / **pilot run manifest** — these are
  build-time outputs that require a live dataset load (Phase 2) and/or a live GPU run (Phase
  6). The CODE that produces them, already implemented and tested, lives at
  `src/neural_thickets_repro/iclr_causal_density/subsets.py::write_subset_manifest`,
  `shuffle_manifest.py::write_shuffle_manifest`, and `candidates.py::write_candidate_manifest`.
- **Schema-valid long-form results**, **per-cell metrics**, **bootstrap outputs**,
  **search-budget analysis**, **grounded-selection analysis**, **integrity/restoration
  report** — all downstream of real result rows that do not exist. The code that would produce
  each, already implemented and unit-tested against synthetic data, lives at
  `metrics.py`, `search_budget.py`, `grounded_selection.py`, and `driver.py`
  (`summarize_population_run` for the integrity/completion report).
- **Figures** — would require plotting fabricated numbers; not generated, per the task's
  explicit "never fabricate results" instruction.

Generating these for real is the concrete next step once a GPU pod is available — see
`decision.md`'s "Exact next action".
