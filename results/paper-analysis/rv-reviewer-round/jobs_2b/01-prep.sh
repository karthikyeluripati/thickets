#!/usr/bin/env bash
# RV-2b prep: Qwen2.5-1.5B at its pinned revision (download retried and verified), GSM8K hashes.
set -ux
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1; cd /workspace/repo
for i in 1 2 3 4 5; do
  MP=$(python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))" | tail -1)
  ls "$MP"/*.safetensors 2>/dev/null && python -c "import safetensors.torch as s,glob;[s.load_file(f,device='cpu') for f in glob.glob('$MP/*.safetensors')];print('WEIGHTS_OK')" && break
done
echo "$MP" > /workspace/mp_q15.txt; ls "$MP"/*.safetensors || exit 1
(cd /workspace/RandOpt && python -u /workspace/repo/scripts/c_prep_gsm8k.py | tee /workspace/data_hashes.txt)
grep -q 'train 7473 c6f812ae33c9159d' /workspace/data_hashes.txt && grep -q 'test 1319 59ec1b7f9357c7a2' /workspace/data_hashes.txt || exit 1
python -c "import datasets; datasets.load_dataset('openai/gsm8k','main',split='train'); datasets.load_dataset('openai/gsm8k','main',split='test'); print('GSM8K cached')"
mkdir -p /workspace/rv
