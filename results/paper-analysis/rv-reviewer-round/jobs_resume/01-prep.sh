#!/usr/bin/env bash
set -ux
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1; cd /workspace/repo
MP=$(python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen2.5-3B-Instruct',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))" | tail -1)
ls $MP/*.safetensors && echo $MP > /workspace/mp_q3.txt
(cd /workspace/RandOpt && python -u /workspace/repo/scripts/c_prep_gsm8k.py | tee /workspace/data_hashes.txt)
grep -q 'train 7473 c6f812ae33c9159d' /workspace/data_hashes.txt && grep -q 'test 1319 59ec1b7f9357c7a2' /workspace/data_hashes.txt
python -c "import datasets; datasets.load_dataset('openai/gsm8k','main',split='train'); datasets.load_dataset('openai/gsm8k','main',split='test'); print('GSM8K cached')"
