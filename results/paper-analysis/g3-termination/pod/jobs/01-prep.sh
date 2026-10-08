set -euxo pipefail
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1
cd /workspace/repo
python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen2.5-VL-3B-Instruct',revision='66285546d2b821cf421d4f5eb2576359d3770cd3',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))" | tail -1 > /workspace/model_path.txt
python -u scripts/g2_prep.py
python - <<'PY'
import json
a = json.load(open('/workspace/g2/items.json'))['test']; b = json.load(open('results/paper-analysis/g2-sameRun/pod/g2/items.json'))['test']
assert [r['id'] for r in a] == [r['id'] for r in b], 'item mismatch'; print('ITEMS_MATCH', len(a))
PY
