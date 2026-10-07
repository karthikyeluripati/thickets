"""P0 pilot (results/paper-analysis/p0/plan_lock.md): is the language-model-only perturbation effect on each GQA question's
answer contrast first-order in the noise? Qwen2.5-VL-3B, HF, fp32 contrast.

Gate: seed 9600003 (sigma 0.002, LM only) rebuilt in HF must be byte-identical to the vLLM/RandOpt realization (g1_check).
Contrast per item j (direct one-word prompt), fixed at BASE from full fp32 logits: g_j = gold-variant first token with the
highest base logit; w_j = highest-logit non-gold token; c_j = h_last . (W[g_j] - W[w_j]).
Prediction: pred[j,k] = sigma_k <fold_LM(grad c_j), eps_k>.  Measurement: c_j under each candidate built exactly as RandOpt
(bf16 base + bf16(sigma*eps) on language-model tensors), minus base.
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
from stage3_gradient import block_of  # noqa: E402

DIRECT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nAnswer with a single word."
BLOCK = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--check', type=Path, required=True); ap.add_argument('--model-path', required=True)
    ap.add_argument('--items', default='results/paper-analysis/p0/items.json')
    ap.add_argument('--cands', default='results/paper-analysis/p0/candidates.json')
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time(); info = {}
    items = json.loads(Path(a.items).read_text(encoding='utf-8'))['heldout']; cands = json.loads(Path(a.cands).read_text())['candidates']
    chk = json.loads(a.check.read_text())
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    proc = AutoProcessor.from_pretrained(a.model_path); tok = proc.tokenizer
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(chk['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    nmax = max(s['numel'] for s in mapping.values()); info['n_stream'] = nmax
    info['tied'] = model.lm_head.weight is model.get_input_embeddings().weight
    lmmap = {v: s for v, s in mapping.items() if block_of(v) != 'vision'}

    def eps(seed):
        g = torch.Generator(device='cuda'); g.manual_seed(int(seed)); return torch.randn(nmax, dtype=torch.bfloat16, device='cuda', generator=g)

    def build(seed, sigma, sha_check=False):
        e = eps(seed); ok = 0
        with torch.no_grad():
            for v, spec in mapping.items():
                parts = []
                for h, off, n in spec['parts']:
                    p = params[h]
                    if v in lmmap:
                        p.data.copy_(p.data + float(sigma) * e[off: off + n].view(p.shape))
                    if sha_check: parts.append(p.detach().contiguous().view(torch.uint8).reshape(-1))
                if sha_check:
                    ok += hashlib.sha256(torch.cat(parts).cpu().numpy().tobytes()).hexdigest() == chk['seeds']['9600003']['sha'][v]
        del e; return ok
    base_cpu = {n: p.detach().to('cpu', copy=True) for n, p in params.items()}

    def restore():
        with torch.no_grad():
            for n, p in params.items(): p.data.copy_(base_cpu[n].to(p.device))
    ok = build(9600003, 0.002, sha_check=True); restore()
    info['mapping_gate'] = {'identical': ok, 'of': len(mapping)}
    (a.out / 'p0_info.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info), flush=True)
    if ok != len(mapping): raise SystemExit('MAPPING_GATE_FAILED')
    W = model.lm_head.weight
    xs = []
    for r in items:
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': DIRECT.format(q=r['question'])}]}],
                                        tokenize=False, add_generation_prompt=True)
        xs.append({k: v for k, v in proc(text=[text], images=[Image.open(a.images / f"{r['imageId']}.jpg").convert('RGB')], return_tensors='pt').items()})
    gw = []
    with torch.no_grad():
        for r, x in zip(items, xs):
            h = model.model(**{k: v.cuda() for k, v in x.items()}).last_hidden_state[0, -1].float(); z = h @ W.float().T
            ans = r['answer']; var = {tok.encode(s, add_special_tokens=False)[0] for s in (ans, ans.capitalize(), ans.upper(), ' ' + ans, ' ' + ans.capitalize())}
            g = max(var, key=lambda t: float(z[t])); zz = z.clone(); zz[list(var)] = -1e30; w = int(torch.argmax(zz))
            gw.append((int(g), w, float(z[g] - z[w])))
    info['contrast_tokens'] = [{'id': r['id'], 'g': g, 'w': w, 'g_str': tok.decode([g]), 'w_str': tok.decode([w]), 'c_base': c} for r, (g, w, c) in zip(items, gw)]

    def contrasts():
        out = []
        with torch.no_grad():
            for x, (g, w, _) in zip(xs, gw):
                h = model.model(**{k: v.cuda() for k, v in x.items()}).last_hidden_state[0, -1].float()
                out.append(float(h @ (W[g].float() - W[w].float())))
        return np.array(out)
    cb = contrasts(); info['base_recompute_maxdiff'] = float(np.max(np.abs(cb - np.array([c for _, _, c in gw]))))
    meas = np.zeros((len(items), len(cands)), np.float32)
    for k, c in enumerate(cands):
        build(c['seed'], c['sigma']); meas[:, k] = contrasts() - cb; restore()
        print(json.dumps({'cand': k, 'sigma': c['sigma'], 'mean_abs_dc': float(np.abs(meas[:, k]).mean())}), flush=True)
    restored = all(torch.equal(params[n].detach().cpu(), base_cpu[n]) for n in list(params)[:50])
    info['restored_sample_equal'] = bool(restored)
    np.save(a.out / 'meas.npy', meas)
    slots = G.hf_slots(mapping); state = {'t': None}

    def hook_for(name):
        off, n, v = slots[name]; lm = v in lmmap

        def hook(p):
            if lm and state['t'] is not None: state['t'][off: off + n].add_(p.grad.reshape(-1))
            p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    pred = np.zeros_like(meas)
    for b0 in range(0, len(items), BLOCK):
        blk = list(range(b0, min(b0 + BLOCK, len(items))))
        Fb = torch.zeros(len(blk), nmax, dtype=torch.float32, device='cuda')
        for i, j in enumerate(blk):
            state['t'] = Fb[i]; g, w, _ = gw[j]
            h = model.model(**{k: v.cuda() for k, v in xs[j].items()}).last_hidden_state[0, -1].float()
            (h @ (W[g].float() - W[w].float())).backward()
        state['t'] = None
        for k, c in enumerate(cands):
            e = eps(c['seed']).float(); pred[blk, k] = (c['sigma'] * (Fb @ e)).cpu().numpy(); del e
        del Fb
    np.save(a.out / 'pred.npy', pred)
    info['minutes'] = (time.time() - t0) / 60
    (a.out / 'p0_info.json').write_text(json.dumps(info, indent=1)); print('P0_DONE', json.dumps({k: v for k, v in info.items() if k != 'contrast_tokens'}), flush=True)


if __name__ == '__main__':
    main()
