set -euxo pipefail
grep -q "^rc=0 " /workspace/jobs/02-smoke.rc || { echo SMOKE_FAILED; exit 3; }
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
cd /workspace/repo; MP=$(cat /workspace/model_path.txt)
# cheap first: BASE at both budgets, then the cost projection gate (lock: session <= $11.50)
python -u scripts/g3_eval.py --arms base --model-path $MP --out /workspace/g3/out | tee /workspace/g3/base_timing.log
python - <<'PY'
import json, time
t = {(d['budget']): d['seconds'] for d in map(json.loads, [l for l in open('/workspace/g3/base_timing.log') if l.startswith('{')])}
r = dict(x.split('=') for x in open('/workspace/rate.txt').read().split()); rate = float(r['RATE']); start = int(r['START'])
spent = (time.time() - start) / 3600 * rate
est_s = 60 * (t[256] + t[1024])      # 50 members + SC(n=50) ~ 60 base-pass equivalents per budget (conservative)
proj = spent + est_s / 3600 * rate
open('/workspace/g3/projection.txt', 'w').write(f'{t} spent={spent:.2f} proj={proj:.2f} {"GO" if proj <= 11.5 else "OVER"}\n')
print(open('/workspace/g3/projection.txt').read())
PY
grep -q " GO$" /workspace/g3/projection.txt || { echo PROJECTION_OVER_CAP; exit 4; }
python -u scripts/g3_eval.py --arms sc,members --model-path $MP --out /workspace/g3/out
python -u scripts/g3_analysis.py --upstream /workspace/RandOpt --d /workspace/g3/out --out /workspace/g3/g3_results.json | tail -30
touch /workspace/g3/G3_DONE
