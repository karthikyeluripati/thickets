#!/usr/bin/env bash
# Coarse neural-thicket localization experiment: 7 scopes x 4 relative-L2 radii x GQA.
# NOT part of the repository -- run this on the pod, do not commit it.
# Fails fast: any run erroring, or any validation check failing, stops the whole script
# immediately. No retries, no auto-tuning, no skipping.
set -euo pipefail

cd "$(dirname "$0")"  # run from the repo root (research/neural_thickets_repro), adjust if needed

SCOPES=(full_lm vision_encoder vision_merger lm_early lm_middle lm_late full_vlm)
RADII=(0.005 0.02 0.04 0.07)
N=100
K=1
TEST_SAMPLES=5
CONFIG=configs/gqa_repro.yaml

WORKDIR=results/coarse_thicket_experiment
mkdir -p "$WORKDIR"
RUN_DIRS_FILE="$WORKDIR/run_dirs.tsv"        # scope \t radius \t out_dir
: > "$RUN_DIRS_FILE"

echo "=== Preflight: re-confirm seed identity on THIS machine before spending GPU time ==="
python3 - "$CONFIG" "$N" "$WORKDIR/expected_seeds.json" <<'PYEOF'
import json, sys
sys.path.insert(0, "src")
from neural_thickets_repro.config import load_config
from neural_thickets_repro.candidate_sampling import sample_candidate_seeds

config_path, n, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = load_config(config_path)
seeds = sample_candidate_seeds(n, cfg.reproducibility.global_seed)
assert len(seeds) == len(set(seeds)) == n, "seed list not unique/complete"
with open(out_path, "w") as f:
    json.dump(seeds, f)
print(f"global_seed={cfg.reproducibility.global_seed} n_seeds={len(seeds)} -- OK, wrote {out_path}")
PYEOF

validate_run() {
    local metrics_path="$1" expected_scope="$2" expected_r="$3" expected_n="$4" seeds_file="$5"
    python3 - "$metrics_path" "$expected_scope" "$expected_r" "$expected_n" "$seeds_file" <<'PYEOF'
import json, sys

metrics_path, expected_scope, expected_r, expected_n, seeds_file = sys.argv[1:6]
expected_r = float(expected_r)
expected_n = int(expected_n)

with open(metrics_path) as f:
    m = json.load(f)
with open(seeds_file) as f:
    expected_seeds = json.load(f)

errors = []
if m.get("scope") != expected_scope:
    errors.append(f"scope mismatch: expected {expected_scope}, got {m.get('scope')}")
if m.get("N") != expected_n:
    errors.append(f"N != {expected_n}: got {m.get('N')}")
if m.get("requested_relative_l2") != expected_r:
    errors.append(f"requested_relative_l2 mismatch: expected {expected_r}, got {m.get('requested_relative_l2')}")
if m.get("restoration_mode") != "fixed_base":
    errors.append(f"restoration_mode != fixed_base: got {m.get('restoration_mode')}")
if m.get("noise_semantics") != "upstream_per_tensor_reseed":
    errors.append(f"noise_semantics != upstream_per_tensor_reseed: got {m.get('noise_semantics')}")
if m.get("base_score") is None:
    errors.append("base_score missing")

records = m.get("candidate_records", [])
if len(records) != expected_n:
    errors.append(f"expected {expected_n} candidate records, got {len(records)}")

counts = (m.get("expert_count") or 0) + (m.get("tie_count") or 0) + (m.get("regression_count") or 0)
if counts != expected_n:
    errors.append(f"expert+tie+regression != {expected_n}: got {counts}")

if any(r.get("requested_relative_l2") != expected_r for r in records):
    errors.append(f"not all candidates recorded requested_relative_l2={expected_r}")
if any(r.get("restoration_mode") != "fixed_base" for r in records):
    errors.append("not all candidates used restoration_mode=fixed_base")
if any(r.get("noise_semantics") != "upstream_per_tensor_reseed" for r in records):
    errors.append("not all candidates used noise_semantics=upstream_per_tensor_reseed")

sigmas = {r.get("sigma") for r in records}
if len(sigmas) != 1:
    errors.append(f"derived sigma not constant within run: {sigmas}")

if m.get("candidate_seed_sequence") != expected_seeds:
    errors.append("candidate_seed_sequence does not match the pre-registered 100-seed sequence")

if errors:
    print(f"VALIDATION FAILED: {metrics_path}")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print(
    f"VALIDATION PASSED: scope={m['scope']:<15} r={expected_r:<6} "
    f"base={m['base_score']:.4f} expert_density={m['expert_density']:.4f} "
    f"(experts={m['expert_count']} ties={m['tie_count']} regressions={m['regression_count']}) "
    f"derived_sigma={list(sigmas)[0]:.6g}"
)
PYEOF
}

echo ""
echo "=== Running 28 experiments sequentially (7 scopes x 4 radii) ==="
run_idx=0
for scope in "${SCOPES[@]}"; do
    for r in "${RADII[@]}"; do
        run_idx=$((run_idx + 1))
        echo ""
        echo "--- Run $run_idx/28: scope=$scope r=$r ---"

        log_file="$WORKDIR/log_${scope}_r${r}.txt"
        python -m neural_thickets_repro.run_scoped_randopt \
            --config "$CONFIG" \
            --N "$N" \
            --K "$K" \
            --restoration-mode fixed_base \
            --perturbation-scope "$scope" \
            --perturbation-scale-mode relative_l2 \
            --relative-l2 "$r" \
            --test-samples "$TEST_SAMPLES" \
            2>&1 | tee "$log_file"

        metrics_path=$(grep -oP '(?<=Wrote )\S*thicket_metrics\.json' "$log_file" | tail -n1)
        if [[ -z "$metrics_path" ]]; then
            echo "STOP: could not find 'Wrote .../thicket_metrics.json' in $log_file"
            exit 1
        fi

        validate_run "$metrics_path" "$scope" "$r" "$N" "$WORKDIR/expected_seeds.json"

        out_dir=$(dirname "$metrics_path")
        printf "%s\t%s\t%s\n" "$scope" "$r" "$out_dir" >> "$RUN_DIRS_FILE"
    done
done

echo ""
echo "=== All 28 runs completed and validated. Running the unmodified aggregator, once per radius ==="
mkdir -p "$WORKDIR/aggregate"
for r in "${RADII[@]}"; do
    dirs=$(awk -F'\t' -v r="$r" '$2==r {print $3}' "$RUN_DIRS_FILE")
    echo ""
    echo "--- Aggregate table for r=$r ---"
    python analysis/aggregate_coarse_thicket.py $dirs --out "$WORKDIR/aggregate/table_r${r}.txt"
done

echo ""
echo "=== Building the compact 7x4 expert_density matrix (direct JSON read, not aggregation logic) ==="
python3 - "$RUN_DIRS_FILE" "$WORKDIR/aggregate/expert_density_matrix.txt" "${RADII[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

run_dirs_file, out_path, *radii = sys.argv[1:]
radii = [float(r) for r in radii]

matrix = {}
with open(run_dirs_file) as f:
    for line in f:
        scope, r, out_dir = line.rstrip("\n").split("\t")
        metrics = json.loads((Path(out_dir) / "thicket_metrics.json").read_text())
        matrix.setdefault(scope, {})[float(r)] = metrics["expert_density"]

scope_order = ["full_lm", "vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late", "full_vlm"]
header = "scope".ljust(16) + "".join(f"r={r:<9}" for r in radii)
lines = [header]
for scope in scope_order:
    row = scope.ljust(16) + "".join(f"{matrix[scope][r]:<11.4f}" for r in radii)
    lines.append(row)

text = "\n".join(lines)
print(text)
Path(out_path).write_text(text)
PYEOF

echo ""
echo "=== DONE. Artifacts: ==="
echo "  $WORKDIR/run_dirs.tsv"
echo "  $WORKDIR/aggregate/table_r<radius>.txt   (4 files, one per radius)"
echo "  $WORKDIR/aggregate/expert_density_matrix.txt"
