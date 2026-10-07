"""G1 gradient job (results/paper-analysis/g1/plan_lock.md), Qwen2.5-VL-3B, language-model-only perturbations.
(M) mapping gate: rebuild seed 9600003 (sigma 0.002, vision untouched) in HF through the vLLM<->HF mapping; every vLLM
tensor must be byte-identical to the realized SHA-256 from g1_check. (G) fold the gradient of the mean direct-prompt
yes-vs-no contrast h_last . (W[Yes] - W[No]) (fp32, at BASE) over two probe sets - SEL: selection yes/no items,
HO: held-out yes/no items - into group stream vectors, and project all 480 candidates' noise streams.
Output: proj_SEL.npy / proj_HO.npy (480 x groups, sigma-scaled), meta.json, gradient_info.json.
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
from stage3_gradient import block_of  # noqa: E402  (6-layer blocks; works for Qwen2.5-VL names)

YES, NO = 9454, 2753
DIRECT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nAnswer with a single word."
GROUPS = ['embed', 'L00-05', 'L06-11', 'L12-17', 'L18-23', 'L24-29', 'L30-35', 'final_norm_head']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--check', type=Path, required=True); ap.add_argument('--model-path', required=True)
    ap.add_argument('--items', default='results/paper-analysis/g1/frozen_items.json')
    ap.add_argument('--cands', default='results/paper-analysis/g1/frozen_candidates.json')
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time(); info = {}
    items = json.loads(Path(a.items).read_text(encoding='utf-8')); chk = json.loads(a.check.read_text())
    cands = json.loads(Path(a.cands).read_text())['candidates']
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    proc = AutoProcessor.from_pretrained(a.model_path)
    assert proc.tokenizer.convert_tokens_to_ids(['Yes', 'No']) == [YES, NO]
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(chk['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    nmax = max(s['numel'] for s in mapping.values()); info['n_stream'] = nmax
    lm = {v: s for v, s in mapping.items() if block_of(v) != 'vision'}
    assert set(block_of(v) for v in lm) == set(GROUPS), sorted(set(block_of(v) for v in lm))
    info['tied_word_embeddings'] = bool(getattr(model.config, 'tie_word_embeddings', False)); info['lm_head_is_embed'] = model.lm_head.weight is model.get_input_embeddings().weight
    seed, sigma = 9600003, 0.002
    g = torch.Generator(device='cuda'); g.manual_seed(seed); e = torch.randn(nmax, dtype=torch.bfloat16, device='cuda', generator=g)
    ok = 0
    with torch.no_grad():
        for v, spec in mapping.items():
            parts = []
            for h, off, n in spec['parts']:
                p = params[h]; val = p.data if block_of(v) == 'vision' else p.data + float(sigma) * e[off: off + n].view(p.shape)
                parts.append(val.contiguous().view(torch.uint8).reshape(-1))
            ok += hashlib.sha256(torch.cat(parts).cpu().numpy().tobytes()).hexdigest() == chk['seeds'][str(seed)]['sha'][v]
    info['mapping_gate'] = {'identical': ok, 'of': len(mapping)}; del e
    (a.out / 'gradient_info.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info), flush=True)
    if ok != len(mapping):
        raise SystemExit('MAPPING_GATE_FAILED')
    slots = G.hf_slots(mapping); W = model.lm_head.weight
    F = {gname: torch.zeros(nmax, dtype=torch.float32, device='cuda') for gname in GROUPS}

    def hook_for(name):
        off, n, v = slots[name]; grp = block_of(v)

        def hook(p):
            if grp != 'vision':
                F[grp][off: off + n].add_(p.grad.reshape(-1))
            p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    meta = {'groups': GROUPS, 'candidates': [c['index'] for c in cands], 'sets': {}}
    buf = torch.empty(nmax, dtype=torch.bfloat16, device='cuda')
    for X, rows in (('SEL', [r for r in items['selection'] if r['yesno']]), ('HO', [r for r in items['heldout'] if r['yesno']])):
        for gname in GROUPS: F[gname].zero_()
        ts = time.time(); base_c = []
        for r in rows:
            text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIRECT.format(q=r['question'])}]}],
                                            tokenize=False, add_generation_prompt=True)
            x = {k: v.cuda() for k, v in proc(text=[text], images=[Image.open(a.images / f"{r['imageId']}.jpg").convert('RGB')], return_tensors='pt').items()}
            h = model.model(**x).last_hidden_state[0, -1]
            c_ = h.float() @ (W[YES].float() - W[NO].float()); base_c.append(float(c_)); (c_ / len(rows)).backward()
        gsec = time.time() - ts
        proj = np.zeros((len(cands), len(GROUPS)), np.float32)
        for k, c in enumerate(cands):
            gen = torch.Generator(device='cuda'); gen.manual_seed(int(c['seed'])); torch.randn(nmax, out=buf, generator=gen); ee = buf.float()
            proj[k] = [float(c['sigma'] * (F[gname] @ ee)) for gname in GROUPS]; del ee
        np.save(a.out / f'proj_{X}.npy', proj)
        meta['sets'][X] = {'n_items': len(rows), 'item_ids': [r['id'] for r in rows], 'base_contrast_fp32': base_c, 'gradient_seconds': gsec,
                           'group_norms': {gname: float(F[gname].norm()) for gname in GROUPS}}
        (a.out / 'meta.json').write_text(json.dumps(meta)); print(json.dumps({'set': X, 'n': len(rows), 'sec': round(gsec, 1)}), flush=True)
    print(json.dumps({'minutes_total': (time.time() - t0) / 60}), flush=True)


if __name__ == '__main__':
    main()
