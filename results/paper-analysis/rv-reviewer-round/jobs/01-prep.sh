#!/usr/bin/env bash
# RV prep: the four models at their pinned revisions, GSM8K data hashes, GQA images + items (as G2/PS).
set -ux
. /workspace/repo/scripts/pod_rv_common.sh
dl Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 q15
dl Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1 q3
dl allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e olmo
dl Qwen/Qwen2.5-VL-3B-Instruct 66285546d2b821cf421d4f5eb2576359d3770cd3 gqa
(cd /workspace/RandOpt && python -u /workspace/repo/scripts/c_prep_gsm8k.py | tee /workspace/data_hashes.txt)
grep -q 'train 7473 c6f812ae33c9159d' /workspace/data_hashes.txt && grep -q 'test 1319 59ec1b7f9357c7a2' /workspace/data_hashes.txt
python -c "import datasets; datasets.load_dataset('openai/gsm8k','main',split='train'); datasets.load_dataset('openai/gsm8k','main',split='test'); print('GSM8K cached')"
python -u scripts/g2_prep.py | tail -2
python - <<'PY'
import hashlib, json
a = json.load(open('/workspace/g2/items.json')); b = json.load(open('results/paper-analysis/g2-sameRun/pod/g2/items.json'))
assert a == b, 'g2 items differ from the committed items.json'; print('G2_ITEMS_MATCH', len(b['selection']), len(b['test']))
PY
mkdir -p /workspace/rv; echo PREP_DONE
