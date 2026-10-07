"""Session 1 / B (results/paper-analysis/s1/plan_lock.md): does the per-item first-order prediction explain WHICH fresh
matched items the winner (9504111, sigma 0.002, all parameters incl. vision) changes? Qwen3-VL-8B, M1 hold-out H (600).
Contrast per item (fixed at BASE from M1's stored vLLM base scores): g = gold letter, r_B = strongest wrong letter;
c_j = h_last . (W[g] - W[r_B]) in fp32 (HF). Prediction split: pred_NOV_j = sigma <fold_nonvision(grad c_j), eps_win>,
pred_V_j = sigma <fold_vision(grad c_j), eps_win>. Measurement (analysis, CPU): the winner's vLLM Δc_j from M1 outputs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402

IDS = [32, 33, 34, 35]; LET = 'ABCD'
SEED, SIGMA = 9504111, 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--holdout', default='results/paper-analysis/m1/pod/holdout.jsonl'); ap.add_argument('--base-comp', default='results/paper-analysis/s1/B_comparators.json')
    ap.add_argument('--vllm-check', default='results/paper-analysis/geometry-gpu-a/gpu/vllm_stream_check.json'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import importlib.util
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    rows = [json.loads(l) for l in Path(a.holdout).read_text(encoding='utf-8').splitlines() if l.strip()]
    comp = json.loads(Path(a.base_comp).read_text())
    if a.smoke: rows = rows[:4]
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    proc = AutoProcessor.from_pretrained(path)
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(json.loads(Path(a.vllm_check).read_text())['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    slots = G.hf_slots(mapping); W = model.lm_head.weight
    F = {'NOV': torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda'), 'V': torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda')}

    def hook_for(name):
        off, n, v = slots[name]; tgt = F['V' if cd.assign_group(v) == 'vision' else 'NOV']

        def hook(p):
            tgt[off: off + n].add_(p.grad.reshape(-1)); p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    g = torch.Generator(device='cuda'); g.manual_seed(SEED); e = torch.randn(G.N_MAX, dtype=torch.bfloat16, device='cuda', generator=g).float()
    out = {}
    for r in rows:
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        x = {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors='pt').items()}
        gi, wi = comp[r['uid']]
        for b in F.values(): b.zero_()
        h = model.model(**x).last_hidden_state[0, -1].float(); c = h @ (W[IDS[gi]].float() - W[IDS[wi]].float()); cb = float(c); c.backward()
        out[r['uid']] = {'pred_NOV': float(SIGMA * (F['NOV'] @ e)), 'pred_V': float(SIGMA * (F['V'] @ e)), 'c_base_hf': cb}
    (a.out / 'B_pred.json').write_text(json.dumps(out)); print('B_DONE', json.dumps({'n': len(out), 'minutes': (time.time() - t0) / 60, 'smoke': a.smoke}), flush=True)


if __name__ == '__main__':
    main()
