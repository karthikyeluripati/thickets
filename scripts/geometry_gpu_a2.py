"""GPU-A run 2 under amendment A1 (results/paper-analysis/geometry-gpu-a/plan_lock_A1.md).

Order: V0 score-definition verification + S2a tokens (BASE, 961) -> S2b exact candidate bytes -> F1 HF-internal
candidate contrasts (9504111 + 12 controls, 761) -> S2c folded gradients of the fp32 contrast + projections.
Stops on any gate failure; saves after every phase/block; in-script budget guard.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402

ROOT = Path('results/perspective-taking-n5000-20261004')
LET = 'ABCD'
IDS = [32, 33, 34, 35]
WIN_SEED, WIN_SIGMA = 9504111, 0.002
BLOCK = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vllm-check', type=Path, required=True)
    ap.add_argument('--usd-per-hour', type=float, required=True); ap.add_argument('--pod-start-epoch', type=float, required=True)
    ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--lists', default='results/paper-analysis/geometry-gpu-a/candidate_lists.json')
    ap.add_argument('--resume-s2c', action='store_true')
    ap.add_argument('--controls', default='results/paper-analysis/random-control-transfer/frozen_controls.json')
    a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import importlib.util
    a.out.mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'steps': []}

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); (a.out / 'cost_log.json').write_text(json.dumps(log, indent=1)); print(json.dumps(kw)[:700], flush=True)

    def guard(minutes):
        if spent() + minutes / 60 * a.usd_per_hour + 0.10 > a.cap_usd:
            note('STOP_BUDGET', next_minutes=minutes); raise SystemExit('budget stop')

    lists = json.loads(Path(a.lists).read_text()); vchk = json.loads(a.vllm_check.read_text())
    controls = json.loads(Path(a.controls).read_text())['candidates']
    rows = {}
    for s, ph in (('search', 'SEARCH'), ('validation', 'RERANK'), ('test', 'TEST')):
        rows[ph] = [json.loads(l) for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()]
    sb = json.loads(gzip.decompress((ROOT / 'baseline/base.json.gz').read_bytes()))
    sbt = {ph: {o['uid']: o for o in sb[s]['outputs']} for s, ph in (('search', 'SEARCH'), ('validation', 'RERANK'), ('test', 'TEST'))}
    s1b = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/base_full.json.gz').read_bytes()))}
    s1c = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/candidate_full.json.gz').read_bytes()))}
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    proc = AutoProcessor.from_pretrained(path)
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(vchk['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    W = model.lm_head.weight
    note('hf_loaded', hf_tensors=len(params))

    def inputs(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        return {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors='pt').items()}

    def hidden(x):
        return model.model(**x).last_hidden_state[0, -1]

    def letters32(h):
        return h.float() @ W[IDS].float().T  # fp32 letter logits (A-D)

    def contrast32(h, y, rB):
        return h.float() @ (W[IDS[y]].float() - W[IDS[rB]].float())

    def vllm_vec(u):
        return np.array([s1b[u]['letter_logprobs'][L] for L in LET])

    def run_s2c(comp, base_s):
        """Memory-safe S2c: each parameter's grad is folded into the example's stream row by a post-accumulate hook and
        freed at once (the full gradient set is never held); activations are checkpointed. Train mode only enables
        checkpointing (no dropout in Qwen3-VL); the forward is checked against the stored eval-mode fp32 contrast."""
        buf = torch.empty(G.N_MAX, dtype=torch.bfloat16, device='cuda')

        def eps_into(seed):
            g = torch.Generator(device='cuda'); g.manual_seed(int(seed)); torch.randn(G.N_MAX, out=buf, generator=g); return buf
        for p in model.parameters():
            p.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False
        model.train()
        group_of = {v: cd.assign_group(v) for v in mapping}
        sf = G.StreamingFolder(params, mapping, group_of)
        loc = set(json.loads((cd.OUT / 'example_manifest.json').read_text())['localization'])
        setF = {ph: torch.zeros(G.N_MAX, dtype=torch.float32) for ph in ('SEARCH', 'RERANK', 'TEST')}  # CPU during the loop
        fwd_dev = []
        for ph in ('RERANK', 'TEST', 'SEARCH'):
            cands = lists[f'{ph}_candidates']; uids = [r['uid'] for r in rows[ph]]
            pred = np.zeros((len(uids), len(cands)), np.float32); group_pred = {}
            for b0 in range(0, len(uids), BLOCK):
                guard(2.0)
                blk = rows[ph][b0: b0 + BLOCK]
                Fb = torch.zeros(len(blk), G.N_MAX, dtype=torch.float32, device='cuda')
                for i, r in enumerate(blk):
                    u = r['uid']; y, rB = comp[u]
                    sf.target = Fb[i]
                    if u in loc:
                        sf.eps = eps_into(WIN_SEED); sf.sigma = WIN_SIGMA; sf.partials = {}
                    else:
                        sf.eps = None
                    c = contrast32(hidden(inputs(r)), y, rB); fwd_dev.append(abs(float(c) - base_s[u])); c.backward()
                    if u in loc:
                        group_pred[u] = {grp: sf.partials.get(grp, 0.) for grp in cd.GROUPS}
                sf.target = None; sf.eps = None
                setF[ph] += Fb.sum(0).cpu()
                for k, (cid, seed, sigma) in enumerate(cands):
                    e = eps_into(seed).float()
                    pred[b0: b0 + len(blk), k] = (sigma * (Fb @ e)).cpu().numpy(); del e
                np.save(a.out / f'pred_{ph}.npy', pred); del Fb
                if len(fwd_dev) == len(blk) and float(np.median(fwd_dev)) > 0.05:  # catches a real mode difference, not kernel noise
                    note('NO_GO_train_mode_forward', max_abs_dev=max(fwd_dev)); raise SystemExit('train-mode forward deviates')
            (a.out / f'meta_{ph}.json').write_text(json.dumps({'uids': uids, 'candidates': cands, 'comparator': {u: comp[u] for u in uids}}))
            if group_pred:
                (a.out / f'group_pred_{ph}.json').write_text(json.dumps(group_pred))
            note(f'S2c_{ph}', examples=len(uids), candidates=len(cands), fwd_vs_stored_base_max=max(fwd_dev), fwd_vs_stored_base_median=float(np.median(fwd_dev)))
        sf.remove()
        guard(2.0)
        S = torch.stack([setF[ph] for ph in ('SEARCH', 'RERANK', 'TEST')]).cuda(); del setF
        allc = lists['ALL_candidates']; proj = np.zeros((len(allc), 3), np.float32)
        for k, (cid, seed, sigma) in enumerate(allc):
            e = eps_into(seed).float(); proj[k] = (sigma * (S @ e)).cpu().numpy(); del e
        np.save(a.out / 'set_projection_all5000.npy', proj)
        names = ('SEARCH', 'RERANK', 'TEST'); nrm = S.norm(dim=1); Gm = (S @ S.T) / (nrm[:, None] * nrm[None, :])
        cos = {f'{names[i]}|{names[j]}': float(Gm[i, j]) for i in range(3) for j in range(3) if names[i] < names[j]}
        note('S2c_sets', norms={names[i]: float(nrm[i]) for i in range(3)}, cosines=cos)

    if a.resume_s2c:
        bc = json.loads((a.out / 'base_contrast_fp32.json').read_text())
        note('RESUME_S2C'); run_s2c(bc['comparator'], bc['s']); note('DONE'); return

    # ---------------- V0 + S2a on BASE (961)
    guard(8)
    W32 = W.float()
    comp, base_s = {}, {}
    v0a, v0b, v0c, tok, agree = [], [], [], 0, 0
    with torch.no_grad():
        for ph in ('SEARCH', 'RERANK', 'TEST'):
            for r in rows[ph]:
                u = r['uid']; x = inputs(r)
                tok += hashlib.sha256(json.dumps(x['input_ids'][0].tolist()).encode()).hexdigest() == sbt[ph][u]['prompt_token_ids_sha256']
                h = hidden(x); l4 = letters32(h).cpu().numpy()
                y = r['answer']
                ref = vllm_vec(u) if ph != 'SEARCH' else l4
                rB = min([k for k in range(4) if k != y], key=lambda k: (-ref[k], k)); comp[u] = [y, rB]
                s32 = float(l4[y] - l4[rB]); base_s[u] = s32
                lpf = torch.log_softmax(h.float() @ W32.T, -1)
                v0a.append(abs(s32 - float(lpf[IDS[y]] - lpf[IDS[rB]])))
                lpb = torch.log_softmax(model.lm_head(h).float(), -1)
                v0b.append(abs(s32 - float(lpb[IDS[y]] - lpb[IDS[rB]])))
                agree += LET[int(np.argmax(l4))] == sbt[ph][u]['text'].strip()[:1].upper()
                if ph != 'SEARCH':
                    v = vllm_vec(u); v0c.append(abs(s32 - (v[y] - v[rB])))
    del W32
    V0 = {'V0a_max': float(max(v0a)), 'V0b_median': float(np.median(v0b)), 'V0b_p99': float(np.quantile(v0b, .99)),
          'V0c_median': float(np.median(v0c)), 'V0c_p90': float(np.quantile(v0c, .9)), 'V0c_p99': float(np.quantile(v0c, .99)),
          'argmax_fp32_agrees_stored_vllm_answer': agree, 'prompt_tokens_identical': tok, 'n': 961}
    note('V0_S2a', **V0)
    (a.out / 'base_contrast_fp32.json').write_text(json.dumps({'s': base_s, 'comparator': comp}))
    if V0['V0a_max'] > 1e-3 or V0['V0b_median'] > 0.125 or V0['V0b_p99'] > 0.5 or tok < 0.99 * 961:
        note('NO_GO_V0_or_S2a'); raise SystemExit('V0/S2a gate')

    # ---------------- S2b exact candidate bytes + F1 HF-internal candidate contrasts
    base_cpu = {n: p.detach().to('cpu', copy=True) for n, p in params.items()}
    buf = torch.empty(G.N_MAX, dtype=torch.bfloat16, device='cuda')

    def eps_into(seed):
        g = torch.Generator(device='cuda'); g.manual_seed(int(seed)); torch.randn(G.N_MAX, out=buf, generator=g); return buf

    def build(seed, sigma, check_sha=False):
        e = eps_into(seed); ok = 0
        with torch.no_grad():
            for v, sp_ in mapping.items():
                parts = []
                for h_, off, n in sp_['parts']:
                    p = params[h_]; p.data.copy_(p.data + float(sigma) * e[off: off + n].view(p.shape))
                    if check_sha: parts.append(p.detach().contiguous().view(torch.uint8).reshape(-1))
                if check_sha:
                    ok += hashlib.sha256(torch.cat(parts).cpu().numpy().tobytes()).hexdigest() == vchk['seeds'][str(seed)]['sha'][v]
        return ok

    def restore():
        with torch.no_grad():
            for n, p in params.items():
                p.data.copy_(base_cpu[n].to(p.device))

    def cand_contrasts():
        out = {}
        with torch.no_grad():
            for ph in ('RERANK', 'TEST'):
                for r in rows[ph]:
                    u = r['uid']; h = hidden(inputs(r)); l4 = letters32(h).cpu().numpy(); y, rB = comp[u]
                    out[u] = {'s': float(l4[y] - l4[rB]), 'argmax': LET[int(np.argmax(l4))]}
        return out
    guard(5)
    sha_ok = build(WIN_SEED, WIN_SIGMA, check_sha=True)
    cw = cand_contrasts()
    note('S2b', packed_tensors_sha_identical=sha_ok, of=len(mapping), answers_agree_stored_vllm=sum(cw[u]['argmax'] == s1c[u]['parsed'] for u in cw), n=len(cw))
    restore()
    if sha_ok != len(mapping):
        note('NO_GO_S2b'); raise SystemExit('S2b byte gate')
    F1 = {str(WIN_SEED): cw}
    for c in controls:
        guard(3)
        build(c['seed'], c['sigma']); F1[str(c['seed'])] = cand_contrasts(); restore()
        (a.out / 'f1_candidate_contrasts.json').write_text(json.dumps(F1))
        note(f"F1_cand_{c['seed']}")
    (a.out / 'f1_candidate_contrasts.json').write_text(json.dumps(F1))
    restored_ok = all(torch.equal(params[n].detach().cpu(), base_cpu[n]) for n in list(params)[:40])
    note('F1_measured_done', restored_sample_equal=bool(restored_ok))
    del base_cpu

    run_s2c(comp, base_s)
    note('DONE')


if __name__ == '__main__':
    main()
