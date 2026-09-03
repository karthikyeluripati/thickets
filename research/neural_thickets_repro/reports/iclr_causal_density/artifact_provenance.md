# Artifact Provenance and Checksums — Isolated 7B Causal-Density Pilot

Raw per-example result artifacts (`results.jsonl`, `results_selection.jsonl`) are large
(~2.0 GB each uncompressed) and are **not** committed to git — consistent with this repository's
existing `.gitignore` convention (`results/**/*.jsonl`, `results/**/*.json`). This document
records exactly what exists, where it lives, and how to verify it against corruption or drift.

## What exists

| File | Rows | Description |
|---|---|---|
| `results.jsonl` | 1,680,000 | Decisive pilot (Phase 6), audit-set pass — 600 candidates × 2800 rows each. `subset_role="audit"`. |
| `results_selection.jsonl` | 1,680,000 | Decisive pilot (Phase 6), selection-set pass — same 600 candidates × 2800 rows each. `subset_role="selection"`. Required for Phase 9 (grounded selection). |
| `base_control_report.json` | — | Phase 5 base-control gate: unperturbed model, all 5 capabilities, both subsets (selection + audit), all 3 conditions, per-example scores. |
| `base_control_gate.json` | — | Phase 5 gate pass/fail summary (`{"pass": true, "failures": []}`). |
| `candidate_manifest.json` | — | The frozen 600-candidate population (scope, radius, seed, candidate_id) — deterministically reproducible from `iclr_causal_density.candidates.build_candidate_population()`. |
| `model_revision_resolution.json` | — | Live-resolved model revision (`Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`). |
| `run_summary.json` / `run_summary_selection.json` | — | Driver-level completion summaries for each pass (`run_complete: true`, `failed: 0` for both). |

## Where they live

1. **Original**: RunPod persistent network volume, mounted at `/workspace` on the pod used for
   this pilot (pod stopped 2026-09-03 after both passes completed and were verified — see
   completeness-validation results below). The volume itself may still exist depending on
   subsequent RunPod account actions; its exact current status is not tracked here.
2. **Backup (authoritative for reproduction)**: downloaded to the operator's local machine at
   `thickets_iclr_causal_density_backup/results_2026-09-03/` (sibling directory to the
   `iclr-causal-density-pilot` worktree), gzip-compressed (`results.jsonl.gz`,
   `results_selection.jsonl.gz`, ~93 MB and ~92 MB respectively) plus the small JSON files
   uncompressed.

Every checksum below was computed **on the pod**, immediately after both passes' completeness
validation passed, and independently re-verified against the local backup (both at the
compressed-transfer level and, for the two large files, at the raw decompressed-content level)
before the pod was stopped. Full checksum file: `checksums.sha256` (co-located with the backup,
not duplicated here to avoid a second source of truth going stale).

```
af66954300320d0b79c32904c1ea00d82a64ff54e26c8ca4dcff11bd1ba297bc  results.jsonl
b4a2131791eaa6a8e7a53ded63ee8ed4dc6aef90dcba2d15c2d61c2fe54dc7e6  results_selection.jsonl
2ecc86e303258e41e26bc4d209724e0a2b2fd2da0c126ef05e85e3a21d63fc38  base_control_report.json
a506865c97ab529505406bee2f5a594ec2d7f7606d8b65524dc340666099eb95  base_control_gate.json
58a095228e95819cb21920dbd4e25b5d41e0f8bcdb08bfaf5795aaebd35093c8  candidate_manifest.json
17b328f2be4eea49a1ca3009b80d6e40d218a7875f46d083b427c9ebaab369a8  model_revision_resolution.json
5183e94e0eac5c2351630e25c4902832ae4739a272ac0667a5674e95aec2109e  run_summary.json
6417865e30a303dc8f66970a8b0e71bbb861a8958a32b06edbd6aac1134c0c83  run_summary_selection.json
3a57a36ee34019a91d3ffc02cdbc6c4eeeb459e7112d20fd52872ddeb5480b1a  results.jsonl.gz
fd83f6ac97f6f324bab783e8781e5171e9c10bfe1d7dab5004395fb99982c8fa  results_selection.jsonl.gz
```

To re-verify the local backup at any time:

```
cd thickets_iclr_causal_density_backup/results_2026-09-03
sha256sum -c checksums.sha256 --ignore-missing   # verifies the .gz + small JSON files present
gzip -t results.jsonl.gz results_selection.jsonl.gz   # gzip stream integrity
zcat results.jsonl.gz | sha256sum             # must equal the results.jsonl line above
zcat results_selection.jsonl.gz | sha256sum   # must equal the results_selection.jsonl line above
```

## Completeness validation (independently re-run, not merely asserted)

Both `results.jsonl` and `results_selection.jsonl` were validated, on the pod, immediately after
each pass completed, against every item the task specification named: exact 6-cell
(`scope × radius`) × 100-seed coverage, 600/600 unique candidate IDs, zero duplicate
`(candidate_id, capability, condition, sample_id)` rows, exactly 2800 rows per candidate, exact
`(capability, condition)` pair coverage per candidate (14, given `visual_grounding`'s missing
`text_only`), 100% `norm_verification_ok` / `scope_isolation_verification_ok` /
`restoration_verification_ok` = `True`, 100% `failure_status == "ok"`, single consistent
`model_name`/`model_revision`, correct radii/scopes, and — for the selection pass specifically —
an explicit cross-check that its 600 candidate IDs are byte-identical to the audit pass's own
600. All checks: **PASS**, zero anomalies, on both datasets.

## Reproducing the final analysis

```
python -m neural_thickets_repro.run_iclr_causal_density_analysis \
    --data-dir /path/to/thickets_iclr_causal_density_backup/results_2026-09-03 \
    --output-dir reports/iclr_causal_density
```

Reads `base_control_report.json`, `results.jsonl[.gz]`, `results_selection.jsonl[.gz]` from
`--data-dir` (the `.gz` files are read directly, streamed, never decompressed to disk); writes
`decision.json` and `analysis_full_output.json` to `--output-dir`. Deterministic given the same
input artifacts — running it again reproduces the identical decision.
