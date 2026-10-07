"""Session 1 / A (results/paper-analysis/s1/plan_lock.md): is the per-question answer-contrast response to small weight
perturbations first-order on a NON-QWEN model and a NEW benchmark? OLMo-2-0425-1B-Instruct, ARC-Challenge (100 items),
24 perturbations (sigma 0.001/0.002/0.005). HF only.

Perturbation (RandOpt-style shared stream, per HF tensor): every parameter p gets bf16(p + bf16(sigma * eps[:numel].view(p.shape)))
with eps = randn(N_MAX, bf16, cuda, Generator(seed)), i.e. RandOpt's per-tensor reseeding (each tensor's noise is the prefix
of one stream). All parameters are language-model parameters.
Contrast per item j (direct letter prompt), fixed at BASE: g = gold letter, w = strongest wrong letter at base;
c_j = h_last . (W[g] - W[w]) in fp32. Prediction pred[j,k] = sigma_k <fold(grad c_j), eps_k>; fold = sum over tensors of
grad(flattened) placed at offset 0. Measurement: c_j(candidate) - c_j(base).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

MODEL, REV = 'allenai/OLMo-2-0425-1B-Instruct', '48d788eca847d4d7548f375ad03d3c9312f6139e'
LET = 'ABCD'; IDS = [32, 33, 34, 35]
BLOCK = 16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--inputs', default='results/paper-analysis/s1/frozen_A.json'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--model', default=MODEL); ap.add_argument('--rev', default=REV)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fin = json.loads(Path(a.inputs).read_text(encoding='utf-8')); items = fin['items']; cands = fin['candidates']
    if a.smoke: items, cands = items[:5], cands[:3]
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev)
    assert [tok.encode(L, add_special_tokens=False) for L in LET] == [[i] for i in IDS]
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.rev, torch_dtype=torch.bfloat16).cuda()
    params = dict(model.named_parameters()); nmax = max(p.numel() for p in params.values())
    W = model.get_output_embeddings().weight

    def enc(r):
        q = 'Question: ' + r['question'] + '\n' + '\n'.join(f'{LET[i]}. {o}' for i, o in enumerate(r['options'])) + '\nAnswer with the letter of the correct option.'
        s = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
        return tok(s, return_tensors='pt', add_special_tokens=False).to('cuda')
    xs = [enc(r) for r in items]

    def hlast(x):
        return model.model(**x).last_hidden_state[0, -1].float()
    gw, base_acc = [], 0
    with torch.no_grad():
        for r, x in zip(items, xs):
            z = hlast(x) @ W[IDS].float().T; g = r['answer']
            w = max([k for k in range(4) if k != g], key=lambda k: float(z[k])); gw.append((g, w)); base_acc += int(int(torch.argmax(z)) == g)

    def contrasts():
        with torch.no_grad():
            return np.array([float(hlast(x) @ (W[IDS[g]].float() - W[IDS[w]].float())) for x, (g, w) in zip(xs, gw)])

    def eps(seed):
        gen = torch.Generator(device='cuda'); gen.manual_seed(int(seed)); return torch.randn(nmax, dtype=torch.bfloat16, device='cuda', generator=gen)
    base_cpu = {n: p.detach().to('cpu', copy=True) for n, p in params.items()}
    cb = contrasts(); meas = np.zeros((len(items), len(cands)), np.float32)
    for k, c in enumerate(cands):
        e = eps(c['seed'])
        with torch.no_grad():
            for n, p in params.items():
                p.data.copy_(p.data + float(c['sigma']) * e[: p.numel()].view(p.shape))
        meas[:, k] = contrasts() - cb; del e
        with torch.no_grad():
            for n, p in params.items():
                p.data.copy_(base_cpu[n].to(p.device))
    restored = all(torch.equal(params[n].detach().cpu(), base_cpu[n]) for n in params)
    state = {'t': None}
    for n, p in params.items():
        def hook(pp, n=n):
            if state['t'] is not None:
                state['t'][: pp.numel()].add_(pp.grad.reshape(-1))
            pp.grad = None
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook)
    pred = np.zeros_like(meas)
    for b0 in range(0, len(items), BLOCK):
        blk = list(range(b0, min(b0 + BLOCK, len(items)))); Fb = torch.zeros(len(blk), nmax, dtype=torch.float32, device='cuda')
        for i, j in enumerate(blk):
            state['t'] = Fb[i]; g, w = gw[j]
            (hlast(xs[j]) @ (W[IDS[g]].float() - W[IDS[w]].float())).backward()
        state['t'] = None
        for k, c in enumerate(cands):
            e = eps(c['seed']).float(); pred[blk, k] = (c['sigma'] * (Fb @ e)).cpu().numpy(); del e
        del Fb
    np.save(a.out / 'A_pred.npy', pred); np.save(a.out / 'A_meas.npy', meas)
    info = {'n_items': len(items), 'n_cands': len(cands), 'base_acc': base_acc / len(items), 'restored_exact': bool(restored),
            'model': a.model, 'rev': a.rev, 'n_params': len(params), 'nmax': nmax, 'minutes': (time.time() - t0) / 60, 'smoke': a.smoke}
    sig = np.array([c['sigma'] for c in cands])
    for s in sorted(set(sig)):
        m = sig == s; info[f'r_sigma_{s}'] = float(np.corrcoef(pred[:, m].ravel(), meas[:, m].ravel())[0, 1])
    (a.out / 'A_info.json').write_text(json.dumps(info, indent=1)); print('A_DONE', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
