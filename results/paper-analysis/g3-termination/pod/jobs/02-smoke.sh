set -euxo pipefail
grep -q "^rc=0 " /workspace/jobs/01-prep.rc || { echo PREP_FAILED; exit 3; }
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
cd /workspace/repo
python -u scripts/g3_eval.py --limit 5 --k 2 --model-path $(cat /workspace/model_path.txt) --out /workspace/g3/smoke
python -c "import json;d=json.load(open('/workspace/g3/smoke/b1024_rank1.json'));print('SMOKE_OK',len(d),d[0])"
