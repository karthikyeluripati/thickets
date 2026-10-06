"""Stage 3A (results/paper-analysis/stage3/plan_lock.md): for probe sets P96, R, S, fold the gradient of the mean
answer-content tilt into 9 group stream vectors and project every candidate's noise stream onto each.
Output: proj_<set>.npy (n_candidates x 9 groups, already multiplied by sigma) + meta.json.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402
from stage12_gradient import tilt_weights  # noqa: E402

IDS = [32, 33, 34, 35]
LET = 'ABCD'
GROUPS = ['embed', 'L00-05', 'L06-11', 'L12-17', 'L18-23', 'L24-29', 'L30-35', 'final_norm_head', 'vision']


def block_of(name):
    g = cd.assign_group(name)
    if g.startswith('lm_q'):
        i = int(re.search(r'(^|\.)layers\.(\d+)\.', name).group(2)); return f'L{6 * (i // 6):02d}-{6 * (i // 6) + 5:02d}'
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vllm-check', type=Path, default=Path('results/paper-analysis/geometry-gpu-a/gpu/vllm_stream_check.json'))
    ap.add_argument('--inputs', type=Path, default=Path('results/paper-analysis/stage3/frozen_inputs.json'))
    ap.add_argument('--lists', type=Path, default=Path('results/paper-analysis/geometry-gpu-a/candidate_lists.json'))
    a = ap.parse_args()
    import importlib.util
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fin = json.loads(a.inputs.read_text()); cands = json.loads(a.lists.read_text())['ALL_candidates']
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
    assert set(block_of(v) for v in mapping) == set(GROUPS)

    def inputs(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        return {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors='pt').items()}
    F = {g: torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda') for g in GROUPS}

    def hook_for(name):
        off, n, v = slots[name]; tgt = F[block_of(v)]

        def hook(p):
            tgt[off: off + n].add_(p.grad.reshape(-1)); p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    buf = torch.empty(G.N_MAX, dtype=torch.bfloat16, device='cuda'); meta = {'groups': GROUPS, 'candidates': cands, 'sets': {}}
    for X in ('P96', 'R', 'S'):
        uids = fin['probe_sets'][X]; ts = time.time()
        for g in GROUPS:
            F[g].zero_()
        for u in uids:
            r = ex[u]; w = torch.tensor(tilt_weights(r['options']), dtype=torch.float32, device='cuda')
            h = model.model(**inputs(r)).last_hidden_state[0, -1]
            ((h.float() @ W[IDS].float().T) @ w / len(uids)).backward()
        gsec = time.time() - ts
        proj = np.zeros((len(cands), len(GROUPS)), np.float32)
        for k, (cid, seed, sigma) in enumerate(cands):
            gen = torch.Generator(device='cuda'); gen.manual_seed(int(seed)); torch.randn(G.N_MAX, out=buf, generator=gen); e = buf.float()
            proj[k] = [float(sigma * (F[g] @ e)) for g in GROUPS]; del e
        np.save(a.out / f'proj_{X}.npy', proj)
        meta['sets'][X] = {'n_items': len(uids), 'gradient_seconds': gsec, 'projection_seconds': time.time() - ts - gsec,
                           'group_norms': {g: float(F[g].norm()) for g in GROUPS}}
        (a.out / 'meta.json').write_text(json.dumps(meta))
        print(json.dumps({'set': X, **{k: v for k, v in meta['sets'][X].items() if k != 'group_norms'}}), flush=True)
    print(json.dumps({'minutes_total': (time.time() - t0) / 60}), flush=True)


if __name__ == '__main__':
    main()
