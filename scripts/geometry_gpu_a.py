"""GPU-A phase H (Hugging Face): S2a base fidelity, S2b exact-candidate layout + fidelity, S2c folded gradients.

Gradients come from HF; behaviour is anchored to the stored vLLM outputs. All definitions are frozen in
results/paper-analysis/geometry-gpu-a/plan_lock.md. Outputs are written after every phase/block; an in-script
budget guard stops before the cap. Stops (no imputation) on any fidelity falsifier.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402

ROOT = Path('results/perspective-taking-n5000-20261004')
LET = 'ABCD'
LETTER_IDS = [32, 33, 34, 35]
WIN_SEED = 9504111
BLOCK = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vllm-check', type=Path, required=True)
    ap.add_argument('--usd-per-hour', type=float, required=True); ap.add_argument('--pod-start-epoch', type=float, required=True)
    ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--lists', default='results/paper-analysis/geometry-gpu-a/candidate_lists.json')
    a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'steps': []}

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); (a.out / 'cost_log.json').write_text(json.dumps(log, indent=1)); print(json.dumps(kw)[:600], flush=True)

    def guard(minutes):
        if spent() + minutes / 60 * a.usd_per_hour + 0.10 > a.cap_usd:
            note('STOP_BUDGET', next_minutes=minutes); raise SystemExit('budget stop')

    lists = json.loads(Path(a.lists).read_text())
    vchk = json.loads(a.vllm_check.read_text())
    rows = {}
    for s, ph in (('search', 'SEARCH'), ('validation', 'RERANK'), ('test', 'TEST')):
        rows[ph] = [json.loads(l) for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()]
    stored_base = json.loads(gzip.decompress((ROOT / 'baseline/base.json.gz').read_bytes()))
    sb = {ph: {o['uid']: o for o in stored_base[s]['outputs']} for s, ph in (('search', 'SEARCH'), ('validation', 'RERANK'), ('test', 'TEST'))}
    s1b = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/base_full.json.gz').read_bytes()))}
    s1c = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/candidate_full.json.gz').read_bytes()))}
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    proc = AutoProcessor.from_pretrained(path)
    import importlib.util
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(vchk['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    note('hf_loaded', hf_tensors=len(params))

    def inputs(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        x = proc(text=[text], images=[img], return_tensors='pt')
        return {k: v.cuda() for k, v in x.items()}

    def letter_lp(x, grad=False):
        with torch.set_grad_enabled(grad):
            lg = model(**x, logits_to_keep=1).logits[0, -1].float()
            return torch.log_softmax(lg, -1)[LETTER_IDS]

    # ---------------- S2a: base fidelity (tokens + scores), all 961
    guard(8)
    fid = {}; hf_base = {}
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        tok_ok = 0; d = []; agree = 0
        for r in rows[ph]:
            x = inputs(r)
            tok_ok += hashlib.sha256(json.dumps(x['input_ids'][0].tolist()).encode()).hexdigest() == sb[ph][r['uid']]['prompt_token_ids_sha256']
            lp = letter_lp(x).cpu().numpy(); hf_base[r['uid']] = lp.tolist()
            agree += LET[int(np.argmax(lp))] == sb[ph][r['uid']]['text'].strip()[:1].upper()
            if ph != 'SEARCH':
                d.append(np.abs(lp - np.array([s1b[r['uid']]['letter_logprobs'][L] for L in LET])).max())
        fid[ph] = {'n': len(rows[ph]), 'prompt_tokens_identical': tok_ok, 'argmax_agrees_stored_answer': agree,
                   'max_abs_dlogp_median': float(np.median(d)) if d else None, 'max_abs_dlogp_p99': float(np.quantile(d, .99)) if d else None}
    note('S2a_fidelity', **fid)
    (a.out / 'hf_base_letter_logprobs.json').write_text(json.dumps(hf_base))
    n_all = sum(fid[p]['n'] for p in fid); agree_all = sum(fid[p]['argmax_agrees_stored_answer'] for p in fid)
    if sum(fid[p]['prompt_tokens_identical'] for p in fid) < 0.99 * n_all or agree_all < 0.99 * n_all or \
            max(fid[p]['max_abs_dlogp_median'] for p in ('RERANK', 'TEST')) > 0.05:
        note('NO_GO_S2a'); raise SystemExit('S2a fidelity falsifier')

    # ---------------- S2b: exact candidate in HF layout (bytes) + behaviour
    guard(6)
    base_cpu = {n: p.detach().to('cpu', copy=True) for n, p in params.items()}
    gen = torch.Generator(device='cuda'); gen.manual_seed(WIN_SEED)
    eps = torch.randn(G.N_MAX, dtype=torch.bfloat16, device='cuda', generator=gen)
    sha_ok = 0
    with torch.no_grad():
        for v, spec_ in mapping.items():
            parts = []
            for h, off, n in spec_['parts']:
                p = params[h]
                p.data.copy_(p.data + 0.002 * eps[off: off + n].view(p.shape))
                parts.append(p.detach().contiguous().view(torch.uint8).reshape(-1))
            sha = hashlib.sha256(torch.cat(parts).cpu().numpy().tobytes()).hexdigest()
            sha_ok += sha == vchk['seeds'][str(WIN_SEED)]['sha'][v]
    cand = {'agree': 0, 'dlp': []}
    for ph in ('RERANK', 'TEST'):
        for r in rows[ph]:
            lp = letter_lp(inputs(r)).cpu().numpy()
            cand['agree'] += LET[int(np.argmax(lp))] == s1c[r['uid']]['parsed']
            cand['dlp'].append(float(np.abs(lp - np.array([s1c[r['uid']]['letter_logprobs'][L] for L in LET])).max()))
    note('S2b_candidate', packed_tensors_sha_identical=sha_ok, of=len(mapping), answers_agree=cand['agree'], n=761,
         max_abs_dlogp_median=float(np.median(cand['dlp'])))
    with torch.no_grad():
        for n, p in params.items():
            p.data.copy_(base_cpu[n].to(p.device))
    if sha_ok != len(mapping) or cand['agree'] < 0.99 * 761:
        note('NO_GO_S2b'); raise SystemExit('S2b layout/behaviour falsifier')
    del base_cpu, eps

    # ---------------- S2c: folded per-example gradients and projections
    for p in model.parameters():
        p.requires_grad_(True)
    group_of = {v: cd.assign_group(v) for v in mapping}
    loc = set(json.loads((cd.OUT / 'example_manifest.json').read_text())['localization'])
    setF = {ph: torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda') for ph in ('SEARCH', 'RERANK', 'TEST')}
    buf = torch.empty(G.N_MAX, dtype=torch.bfloat16, device='cuda')

    def eps_into(seed):
        g = torch.Generator(device='cuda'); g.manual_seed(int(seed))
        torch.randn(G.N_MAX, out=buf, generator=g)
        return buf

    comparator = {}
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        cands = lists[f'{ph}_candidates']  # list of [candidate_id, seed, sigma]
        uids = [r['uid'] for r in rows[ph]]
        pred = np.zeros((len(uids), len(cands)), np.float32)
        group_pred = {}
        for b0 in range(0, len(uids), BLOCK):
            guard(2.5)
            blk = rows[ph][b0: b0 + BLOCK]
            Fb = torch.zeros(len(blk), G.N_MAX, dtype=torch.float32, device='cuda')
            for i, r in enumerate(blk):
                y = LET.index(LET[r['answer']])
                base_lp = (np.array([s1b[r['uid']]['letter_logprobs'][L] for L in LET]) if ph != 'SEARCH' else np.array(hf_base[r['uid']]))
                rB = min([k for k in range(4) if k != y], key=lambda k: (-base_lp[k], k)); comparator[r['uid']] = [y, rB]
                model.zero_grad(set_to_none=True)
                lp = letter_lp(inputs(r), grad=True)
                (lp[y] - lp[rB]).backward()
                grads = {n: p.grad for n, p in params.items() if p.grad is not None}
                G.fold_into(Fb[i], grads, mapping)
                if r['uid'] in loc:  # S2d: per-group partial folds for 9504111 only
                    e = eps_into(WIN_SEED)
                    for grp in cd.GROUPS:
                        Fg = torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda')
                        G.fold_into(Fg, grads, {v: s for v, s in mapping.items() if group_of[v] == grp})
                        group_pred.setdefault(r['uid'], {})[grp] = float(0.002 * (Fg @ e.float()))
                        del Fg
            model.zero_grad(set_to_none=True)
            setF[ph] += Fb.sum(0)
            for k, (cid, seed, sigma) in enumerate(cands):
                e = eps_into(seed).float()
                pred[b0: b0 + len(blk), k] = (sigma * (Fb @ e)).cpu().numpy()
            np.save(a.out / f'pred_{ph}.npy', pred)
            del Fb
        (a.out / f'meta_{ph}.json').write_text(json.dumps({'uids': uids, 'candidates': cands, 'comparator': {u: comparator[u] for u in uids}}))
        if group_pred:
            (a.out / f'group_pred_{ph}.json').write_text(json.dumps(group_pred))
        note(f'S2c_{ph}', examples=len(uids), candidates=len(cands))
    # set-level projections onto all 5,000
    allc = lists['ALL_candidates']
    proj = np.zeros((len(allc), 3), np.float32)
    for k, (cid, seed, sigma) in enumerate(allc):
        e = eps_into(seed).float()
        proj[k] = [float(sigma * (setF[ph] @ e)) for ph in ('SEARCH', 'RERANK', 'TEST')]
    np.save(a.out / 'set_projection_all5000.npy', proj)
    norms = {ph: float(setF[ph].norm()) for ph in setF}
    cos = {f'{p}|{q}': float((setF[p] @ setF[q]) / (setF[p].norm() * setF[q].norm())) for p in setF for q in setF if p < q}
    note('S2c_sets', norms=norms, cosines=cos)
    note('DONE')


if __name__ == '__main__':
    main()
