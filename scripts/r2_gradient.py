"""R2 (results/paper-analysis/r2/plan_lock.md), Qwen2.5-VL-7B: (M) mapping gate - rebuild seed 9504111 in HF through the
vLLM<->HF mapping and require byte-identical packed tensors vs the vLLM-realized SHA-256 from job 01; then (G) fold the
gradient of the mean 'front' tilt on P96 into 10 group stream vectors and project the frozen perturbations' streams.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402
from r2_common import GROUPS, LETTER_IDS, MODEL, block_of  # noqa: E402
from stage12_gradient import tilt_weights  # noqa: E402

LET = 'ABCD'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--check', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, default=Path('results/paper-analysis/r2/frozen_inputs.json'))
    a = ap.parse_args()
    import importlib.util
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time(); info = {}
    fin = json.loads(a.inputs.read_text()); chk = json.loads(a.check.read_text())
    ex = {}
    for s in ('search', 'validation'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    path = snapshot_download(MODEL, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    proc = AutoProcessor.from_pretrained(path)
    ids = proc.tokenizer.convert_tokens_to_ids(list(LET)); info['letter_ids'] = ids
    assert ids == [LETTER_IDS[L] for L in LET], ids
    IDS = ids
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(chk['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    nmax = max(s['numel'] for s in mapping.values()); info['n_stream'] = nmax
    assert set(block_of(v) for v in mapping) == set(GROUPS)
    # (M) mapping gate: realized candidate bytes per vLLM tensor
    seed, sigma = 9504111, 0.002
    g = torch.Generator(device='cuda'); g.manual_seed(seed); e = torch.randn(nmax, dtype=torch.bfloat16, device='cuda', generator=g)
    ok = 0
    with torch.no_grad():
        for v, spec_ in mapping.items():
            parts = []
            for h, off, n in spec_['parts']:
                p = params[h]; parts.append((p.data + float(sigma) * e[off: off + n].view(p.shape)).contiguous().view(torch.uint8).reshape(-1))
            ok += hashlib.sha256(torch.cat(parts).cpu().numpy().tobytes()).hexdigest() == chk['seeds'][str(seed)]['sha'][v]
    info['mapping_gate'] = {'identical': ok, 'of': len(mapping)}; del e
    (a.out / 'gradient_info.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info), flush=True)
    if ok != len(mapping):
        raise SystemExit('MAPPING_GATE_FAILED')
    slots = G.hf_slots(mapping); W = model.lm_head.weight

    def inputs(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        return {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors='pt').items()}
    F = {gname: torch.zeros(nmax, dtype=torch.float32, device='cuda') for gname in GROUPS}

    def hook_for(name):
        off, n, v = slots[name]; tgt = F[block_of(v)]

        def hook(p):
            tgt[off: off + n].add_(p.grad.reshape(-1)); p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    uids = fin['probe_uids']; ts = time.time()
    for u in uids:
        r = ex[u]; w = torch.tensor(tilt_weights(r['options']), dtype=torch.float32, device='cuda')
        h = model.model(**inputs(r)).last_hidden_state[0, -1]
        ((h.float() @ W[IDS].float().T) @ w / len(uids)).backward()
    gsec = time.time() - ts
    seeds = [(p['candidate_id'], p['seed'], p['sigma']) for p in fin['perturbations']] + [tuple(c) for c in fin['share_seeds']]
    buf = torch.empty(nmax, dtype=torch.bfloat16, device='cuda'); proj = np.zeros((len(seeds), len(GROUPS)), np.float32)
    for k, (cid, sd, sg) in enumerate(seeds):
        gen = torch.Generator(device='cuda'); gen.manual_seed(int(sd)); torch.randn(nmax, out=buf, generator=gen); e = buf.float()
        proj[k] = [float(sg * (F[gname] @ e)) for gname in GROUPS]; del e
    np.save(a.out / 'proj_FRONT.npy', proj)
    meta = {'groups': GROUPS, 'seeds': seeds, 'gradient_seconds': gsec, 'group_norms': {gname: float(F[gname].norm()) for gname in GROUPS},
            'minutes_total': (time.time() - t0) / 60}
    (a.out / 'meta.json').write_text(json.dumps(meta)); print(json.dumps({k: v for k, v in meta.items() if k not in ('seeds',)}), flush=True)


if __name__ == '__main__':
    main()
