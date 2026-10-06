"""I5 (results/paper-analysis/i5/plan_lock.md): per-ITEM first-order predictions of the non-vision answer-content tilt for
the Stage-2 probe (96 items) and the 80 Stage-2 perturbations. For each item j: F_j = fold_nonvision(grad B_j) at BASE
(HF fp32 contrast, as Stage 2); pred[j, c] = sigma_c <F_j, eps_c>. Items processed in blocks; vision gradients discarded.
Output: pred_items.npy (96 x 80) + meta.json.
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
from stage12_gradient import tilt_weights  # noqa: E402

IDS = [32, 33, 34, 35]
LET = 'ABCD'
BLOCK = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vllm-check', type=Path, default=Path('results/paper-analysis/geometry-gpu-a/gpu/vllm_stream_check.json'))
    ap.add_argument('--inputs', type=Path, default=Path('results/paper-analysis/stage12/frozen_inputs.json'))
    a = ap.parse_args()
    import importlib.util
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fin = json.loads(a.inputs.read_text()); probe = fin['probe_uids']; perts = fin['perturbations']
    ex = {}
    for s in ('search', 'validation'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    proc = AutoProcessor.from_pretrained(path)
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(json.loads(a.vllm_check.read_text())['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    slots = G.hf_slots(mapping); W = model.lm_head.weight

    def inputs(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        return {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors='pt').items()}
    state = {'target': None}

    def hook_for(name):
        off, n, v = slots[name]; vis = cd.assign_group(v) == 'vision'

        def hook(p):
            if not vis and state['target'] is not None:
                state['target'][off: off + n].add_(p.grad.reshape(-1))
            p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    buf = torch.empty(G.N_MAX, dtype=torch.bfloat16, device='cuda'); pred = np.zeros((len(probe), len(perts)), np.float32); base_B = []
    for b0 in range(0, len(probe), BLOCK):
        blk = probe[b0: b0 + BLOCK]
        Fb = torch.zeros(len(blk), G.N_MAX, dtype=torch.float32, device='cuda')
        for i, u in enumerate(blk):
            r = ex[u]; w = torch.tensor(tilt_weights(r['options']), dtype=torch.float32, device='cuda')
            state['target'] = Fb[i]
            h = model.model(**inputs(r)).last_hidden_state[0, -1]
            Bj = (h.float() @ W[IDS].float().T) @ w; base_B.append(float(Bj)); Bj.backward()
        state['target'] = None
        for k, p_ in enumerate(perts):
            g = torch.Generator(device='cuda'); g.manual_seed(int(p_['seed'])); torch.randn(G.N_MAX, out=buf, generator=g); e = buf.float()
            pred[b0: b0 + len(blk), k] = (p_['sigma'] * (Fb @ e)).cpu().numpy(); del e
        del Fb; np.save(a.out / 'pred_items.npy', pred)
        print(json.dumps({'block': b0, 'minutes': round((time.time() - t0) / 60, 2)}), flush=True)
    (a.out / 'meta.json').write_text(json.dumps({'probe_uids': probe, 'candidate_ids': [p['candidate_id'] for p in perts], 'base_B_fp32': base_B,
                                                 'minutes_total': (time.time() - t0) / 60}))
    print('I5_DONE', flush=True)


if __name__ == '__main__':
    main()
