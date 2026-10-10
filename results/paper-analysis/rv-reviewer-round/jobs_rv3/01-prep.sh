#!/usr/bin/env bash
# RV-3 prep: OLMo-2-0425-1B-Instruct at its pinned revision (download retried and verified), GSM8K hashes.
set -ux
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1; cd /workspace/repo
for i in 1 2 3 4 5; do
  MP=$(python -c "from huggingface_hub import snapshot_download as s;print(s('allenai/OLMo-2-0425-1B-Instruct',revision='48d788eca847d4d7548f375ad03d3c9312f6139e',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))" | tail -1)
  ls "$MP"/*.safetensors 2>/dev/null && python -c "import safetensors.torch as s,glob;[s.load_file(f,device='cpu') for f in glob.glob('$MP/*.safetensors')];print('WEIGHTS_OK')" && break
done
echo "$MP" > /workspace/mp_olmo.txt; ls "$MP"/*.safetensors || exit 1
(cd /workspace/RandOpt && python -u /workspace/repo/scripts/c_prep_gsm8k.py | tee /workspace/data_hashes.txt)
grep -q 'train 7473 c6f812ae33c9159d' /workspace/data_hashes.txt && grep -q 'test 1319 59ec1b7f9357c7a2' /workspace/data_hashes.txt || exit 1
python -c "import datasets; datasets.load_dataset('openai/gsm8k','main',split='train'); datasets.load_dataset('openai/gsm8k','main',split='test'); print('GSM8K cached')"
mkdir -p /workspace/rv
