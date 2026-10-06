"""Post-hoc causal diagnostic of ONE candidate (seed 9504111, sigma 0.002, all parameters incl. vision).

Hybrids of BASE and CANDIDATE are built by COPYING existing tensor values inside the original in-process vLLM
engine with the pinned upstream RandOpt WorkerExtension (the path that reproduced the study byte-for-byte in the
forensic audit). No noise is regenerated per group; no deltas are subtracted. Hybrids are diagnostic objects on
already-inspected evaluation sets, never candidate experts.

CPU parts (importable, unit-tested): parameter partition, example manifests, answer-score contrasts, cyclic option
permutations. GPU part: `main()` runs session 1 (reproduction, controls, coarse interventions on LOCALIZATION).

Usage (CPU):  python scripts/causal_diag_9504111.py manifest
Usage (GPU):  python scripts/causal_diag_9504111.py session1 --images DIR --upstream DIR --out DIR --usd-per-hour X --pod-start-epoch T --cap-usd C
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path('results/perspective-taking-n5000-20261004')
DATA = Path('examples/omnispatial-perspective-taking')
OUT = Path('results/paper-analysis/causal-diagnostic')
MODEL, REV = 'Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
CANDIDATE_ID = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
SEED, SIGMA = 9504111, 0.002
BASE_STATE = '2582817f4966eeaea6723a39f3ef153d18c90c27e2d3adf722625836a45fb3f5'  # forensic Phase Q, vLLM 0.11 in-process
CAND_STATE = '7d7ef38b0154ae91e54c01852085101ea7846165e0be35cfbd9dcf9a551dd831'
LETTERS = 'ABCD'
LETTER_IDS = {'A': 32, 'B': 33, 'C': 34, 'D': 35}  # verified on the pinned tokenizer.json: single tokens, no leading space
N_LAYERS = 36
QUARTERS = ((0, 8), (9, 17), (18, 26), (27, 35))
GROUPS = ('vision', 'embed', 'lm_q1', 'lm_q2', 'lm_q3', 'lm_q4', 'final_norm_head')
MANIFEST_SEED = 20261009


# ------------------------------------------------------------------------------------- parameter partition
def assign_group(name):
    """Frozen, exhaustive rule set. Works for HF names (model.visual.*, model.language_model.*, lm_head.weight) and
    vLLM names (visual.*, language_model.model.*, language_model.lm_head.weight; packed qkv_proj / gate_up_proj).
    'vision' includes patch/pos embeddings, all 27 blocks, the merger and the DeepStack mergers (whose outputs are
    injected into the first language layers)."""
    if re.search(r'(^|\.)visual\.', name):
        return 'vision'
    if name.endswith('embed_tokens.weight'):
        return 'embed'
    m = re.search(r'(^|\.)layers\.(\d+)\.', name)
    if m:
        i = int(m.group(2))
        if not 0 <= i < N_LAYERS:
            raise ValueError(f'layer index out of range: {name}')
        return f'lm_q{1 + next(k for k, (a, b) in enumerate(QUARTERS) if a <= i <= b)}'
    if name.endswith('lm_head.weight') or re.search(r'(^|\.)norm\.weight$', name):
        return 'final_norm_head'
    raise ValueError(f'unassigned parameter: {name}')


def partition(names):
    """Every parameter exactly once; raises on duplicates or unassigned names."""
    if len(set(names)) != len(names):
        raise ValueError('duplicate parameter names')
    out = {g: [] for g in GROUPS}
    for n in names:
        out[assign_group(n)].append(n)
    return out


def hf_to_vllm_names(hf_names):
    """Expected vLLM names after q/k/v and gate/up packing (used only to test the rules on both naming schemes)."""
    out = set()
    for n in hf_names:
        n2 = n.replace('model.language_model.', 'language_model.model.').replace('model.visual.', 'visual.')
        if n2 == 'lm_head.weight':
            n2 = 'language_model.lm_head.weight'
        n2 = re.sub(r'self_attn\.(q|k|v)_proj', 'self_attn.qkv_proj', n2)
        n2 = re.sub(r'mlp\.(gate|up)_proj', 'mlp.gate_up_proj', n2)
        out.add(n2)
    return sorted(out)


# ------------------------------------------------------------------------------------------ example labels
def transition(base_ok, cand_ok, base_ans, cand_ans):
    if base_ok and cand_ok:
        return 'both_correct'
    if cand_ok:
        return 'repair'
    if base_ok:
        return 'regression'
    return 'both_wrong_same' if base_ans == cand_ans else 'both_wrong_diff'


def systematic_half(items, key_fn, rng, take='half', k=None):
    """Proportionally stratified selection: sort by (stratum key, random), take every other item from a random offset
    (take='half', rounds up for odd counts via the offset) or a systematic sample of size k."""
    import numpy as np
    order = sorted(range(len(items)), key=lambda i: (key_fn(items[i]), rng.random()))
    if take == 'half':
        off = int(rng.integers(2))
        return [order[i] for i in range(len(order)) if (i + off) % 2 == 0]
    step = len(order) / k
    start = rng.random() * step
    return [order[int(start + j * step)] for j in range(k)]  # in stratum order


def build_manifest(rows, seed=MANIFEST_SEED, n_unchanged_per_cell=6):
    """rows: dicts with uid, phase (RERANK/TEST), subtask, gold, base, cand.
    LOCALIZATION: about half of every changed-answer cell (repair, regression, both_wrong_diff) per phase, stratified by
    subtask, plus n_unchanged_per_cell examples from each unchanged cell (both_correct, both_wrong_same) per phase.
    MECHANISM_CHECK: the remaining changed-answer examples plus a disjoint, equally sized unchanged sample."""
    import numpy as np
    rng = np.random.default_rng(seed)
    for r in rows:
        r['transition'] = transition(r['base'] == r['gold'], r['cand'] == r['gold'], r['base'], r['cand'])
    loc, mech, cells = [], [], {}
    for ph in ('RERANK', 'TEST'):
        for t in ('repair', 'regression', 'both_wrong_diff', 'both_correct', 'both_wrong_same'):
            cell = [r for r in rows if r['phase'] == ph and r['transition'] == t]
            cell.sort(key=lambda r: r['uid'])
            if t in ('both_correct', 'both_wrong_same'):
                pick = systematic_half(cell, lambda r: r['subtask'], rng, take='k', k=2 * n_unchanged_per_cell)
                l_ids = [cell[i]['uid'] for i in pick[0::2]]; m_ids = [cell[i]['uid'] for i in pick[1::2]]  # alternate within stratum order
            else:
                a = set(systematic_half(cell, lambda r: r['subtask'], rng))
                l_ids = [cell[i]['uid'] for i in sorted(a)]; m_ids = [cell[i]['uid'] for i in range(len(cell)) if i not in a]
            cells[f'{ph}|{t}'] = {'total': len(cell), 'localization': len(l_ids), 'mechanism_check': len(m_ids)}
            loc += l_ids; mech += m_ids
    assert not set(loc) & set(mech)
    return {'seed': seed, 'rule': build_manifest.__doc__, 'cells': cells, 'localization': loc, 'mechanism_check': mech,
            'labels': {r['uid']: {k: r[k] for k in ('phase', 'subtask', 'gold', 'base', 'cand', 'transition')} for r in rows}}


# ------------------------------------------------------------------------------------------ answer scores
def letter_scores(top_logprobs):
    """top_logprobs: {token_id: logprob} at the first generated position (vLLM raw logprobs, top-20).
    Returns per-letter log-probabilities; letters absent from the top-20 get None plus an upper bound."""
    bound = min(top_logprobs.values()) if top_logprobs else None
    return {L: top_logprobs.get(t) for L, t in LETTER_IDS.items()}, bound


def contrast(scores, a, b):
    """Fixed contrast score(a) - score(b) at the same prefix (the prompt); None if either is missing."""
    if a == b:
        return 0.0
    if scores.get(a) is None or scores.get(b) is None:
        return None
    return scores[a] - scores[b]


def margin(scores, gold):
    """Correct-option score minus the strongest incorrect option (missing incorrect letters are below the top-20)."""
    if scores.get(gold) is None:
        return None
    others = [v for k, v in scores.items() if k != gold and v is not None]
    return scores[gold] - max(others) if others else None


# ------------------------------------------------------------------------------------------ route A helpers
def cyclic_orders(n=4):
    return [[(i + s) % n for i in range(n)] for s in range(n)]


def permute_example(options, answer_idx, order):
    """order[j] = original index shown at position j. Returns new options, new gold index, and a map from shown letter
    to original option index (to map predictions back to option CONTENT)."""
    new = [options[i] for i in order]
    return new, order.index(answer_idx), {LETTERS[j]: order[j] for j in range(len(order))}


def order_sensitive(options):
    """Options whose meaning depends on position/letters (flag separately under permutation)."""
    pat = re.compile(r'(all|none|both) of the (above|options)|option [A-D]\b|\b[A-D] and [A-D]\b', re.I)
    return any(pat.search(o) for o in options)


# --------------------------------------------------------------------------------------------- CPU manifest
def make_manifest():
    import pandas as pd
    d = pd.read_parquet('results/paper-analysis/paper_master_predictions.parquet',
                        columns=['candidate_id', 'phase', 'example_id', 'subtask', 'true_option', 'base_prediction', 'candidate_prediction'])
    c = d[(d.candidate_id == CANDIDATE_ID) & d.phase.isin(['RERANK', 'TEST'])]
    rows = [{'uid': str(r.example_id), 'phase': str(r.phase), 'subtask': str(r.subtask), 'gold': str(r.true_option),
             'base': str(r.base_prediction), 'cand': str(r.candidate_prediction)} for r in c.itertuples()]
    man = build_manifest(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'example_manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(man['cells'], indent=1), len(man['localization']), len(man['mechanism_check']))


# ------------------------------------------------------------------------------------------------ GPU side
def _state_hash(worker):
    import torch
    h = hashlib.sha256()
    for n, p in worker.model_runner.model.named_parameters():
        h.update(n.encode()); h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _param_names(worker):
    return [n for n, _ in worker.model_runner.model.named_parameters()]


def _snapshot_candidate(worker):
    """Copy the realized candidate tensors (exact BF16 values produced by the upstream apply_perturbation) to host."""
    worker._cand = {n: p.detach().to('cpu', copy=True) for n, p in worker.model_runner.model.named_parameters()}
    return len(worker._cand)


def _set_state(worker, cand_names):
    """Every parameter is overwritten from a stored source: CANDIDATE values for names in cand_names, BASE otherwise.
    Verifies each tensor equals its intended source exactly. Returns (#tensors from candidate, #mismatches)."""
    import torch
    cand_names = set(cand_names); bad = 0
    for n, p in worker.model_runner.model.named_parameters():
        src = worker._cand[n] if n in cand_names else worker._base_weights[n]
        p.data.copy_(src.to(p.device, non_blocking=False))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for n, p in worker.model_runner.model.named_parameters():
        src = worker._cand[n] if n in cand_names else worker._base_weights[n]
        bad += int(not torch.equal(p.data, src.to(p.device)))
    return len(cand_names), bad


def _mismatch(worker, which):
    """Exact tensor-by-tensor comparison of the live weights with the stored BASE or CANDIDATE snapshot."""
    import torch
    ref = worker._cand if which == 'cand' else worker._base_weights
    return sum(int(not torch.equal(p.data, ref[n].to(p.device))) for n, p in worker.model_runner.model.named_parameters())


def _group_norms(worker, groups):
    import torch
    out = {}
    for g, names in groups.items():
        d2 = b2 = numel = 0.
        for n in names:
            b = worker._base_weights[n].float().to('cuda'); c = worker._cand[n].float().to('cuda')
            d2 += float(((c - b) ** 2).sum()); b2 += float((b ** 2).sum()); numel += b.numel()
        out[g] = {'tensors': len(names), 'params': int(numel), 'delta_l2': d2 ** .5, 'base_l2': b2 ** .5,
                  'relative_delta': (d2 / b2) ** .5 if b2 else None, 'delta_rms': (d2 / numel) ** .5 if numel else None}
    return out


def main_session1(a):
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    a.out.mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'usd_per_hour': a.usd_per_hour, 'cap_usd': a.cap_usd, 'steps': []}

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); (a.out / 'cost_log.json').write_text(json.dumps(log, indent=1)); print(json.dumps(kw)[:600], flush=True)

    def guard(next_minutes):
        if spent() + next_minutes / 60 * a.usd_per_hour > a.cap_usd:
            note('STOP_BUDGET', next_minutes=next_minutes); raise SystemExit('budget stop')

    man = json.loads(Path(a.manifest).read_text())
    rows = {s: [json.loads(l) for l in (DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ('validation', 'test')}
    allrows = [(('RERANK' if s == 'validation' else 'TEST'), r) for s in ('validation', 'test') for r in rows[s]]
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py')
    sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    path = snapshot_download(MODEL, revision=REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.5, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=16384, max_logprobs=20,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights')
    proc = AutoProcessor.from_pretrained(path)

    def prompt(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LETTERS[i]}. {o}'
        return proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)

    req = {r['uid']: {'prompt': prompt(r), 'multi_modal_data': {'image': Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')}}
           for _, r in allrows}
    phase_of = {r['uid']: ph for ph, r in allrows}
    gold = {r['uid']: LETTERS[r['answer']] for _, r in allrows}
    sp_ = SamplingParams(temperature=0, seed=0, max_tokens=16, logprobs=20)

    def run(name, uids):
        t0 = time.time()
        res = llm.generate([req[u] for u in uids], sp_, use_tqdm=False)
        out = []
        for u, o in zip(uids, res):
            g = o.outputs[0]
            lp0 = {int(k): float(v.logprob) for k, v in (g.logprobs[0] if g.logprobs else {}).items()}
            sc, bound = letter_scores(lp0)
            text = g.text
            out.append({'uid': u, 'phase': phase_of[u], 'text': text, 'parsed': (text.strip()[:1].upper() if text.strip()[:1].upper() in LETTERS else ''),
                        'first_token': int(g.token_ids[0]) if g.token_ids else None, 'letter_logprobs': sc, 'top20_floor': bound,
                        'finish': g.finish_reason, 'n_tokens': len(g.token_ids)})
        (a.out / f'{name}.json.gz').write_bytes(gzip.compress(json.dumps(out).encode()))
        note(f'run:{name}', n=len(uids), seconds=round(time.time() - t0, 1), correct=sum(x['parsed'] == gold[x['uid']] for x in out))
        return out

    full = [r['uid'] for _, r in allrows]
    loc = man['localization']
    note('model_loaded')
    # ---- 1. exact-state reproduction
    hb = rpc(_state_hash); note('base_hash', ok=hb == BASE_STATE, hash=hb)
    if hb != BASE_STATE: raise SystemExit('INSTRUMENTATION_MISMATCH: base state')
    guard(6); base = run('base_full', full)
    rpc('apply_perturbation', SEED, SIGMA)
    hc = rpc(_state_hash); note('candidate_hash', ok=hc == CAND_STATE, hash=hc)
    if hc != CAND_STATE: raise SystemExit('INSTRUMENTATION_MISMATCH: candidate state')
    guard(4); cand = run('candidate_full', full)
    stored_b = json.loads(gzip.decompress((ROOT / 'baseline/base.json.gz').read_bytes()))
    sb = {o['uid']: o['text'] for s in ('validation', 'test') for o in stored_b[s]['outputs']}
    sc_ = {}
    for d in ('validation-shard-01', 'test'):
        raw = json.loads(gzip.decompress((ROOT / d / 'candidates' / f'{CANDIDATE_ID}.json.gz').read_bytes()))
        for s, v in raw['splits'].items():
            sc_.update({o['uid']: o['text'] for o in v['outputs']})
    repro = {'base_text_identical': sum(x['text'] == sb[x['uid']] for x in base), 'cand_text_identical': sum(x['text'] == sc_[x['uid']] for x in cand),
             'base_parsed_identical': sum(x['parsed'] == sb[x['uid']].strip()[:1].upper() for x in base),
             'cand_parsed_identical': sum(x['parsed'] == sc_[x['uid']].strip()[:1].upper() for x in cand), 'n': len(full),
             'first_token_is_letter': sum(x['first_token'] in LETTER_IDS.values() for x in base + cand),
             'all_letters_in_top20': sum(all(v is not None for v in x['letter_logprobs'].values()) for x in base + cand)}
    note('reproduction', **repro)
    if repro['base_parsed_identical'] != len(full) or repro['cand_parsed_identical'] != len(full):
        (a.out / 'INSTRUMENTATION_MISMATCH').write_text(json.dumps(repro)); raise SystemExit('INSTRUMENTATION_MISMATCH: answers differ')
    # ---- 2. controls
    rpc(_snapshot_candidate)
    names = rpc(_param_names); groups = partition(names)
    (a.out / 'parameter_groups_vllm.json').write_text(json.dumps({g: v for g, v in groups.items()}, indent=0))
    note('partition', tensors=len(names), per_group={g: len(v) for g, v in groups.items()})
    norms = rpc(_group_norms, groups); (a.out / 'group_norms.json').write_text(json.dumps(norms, indent=1)); note('group_norms')
    rpc('reset_to_base_weights'); rpc('apply_perturbation', SEED, SIGMA)
    note('repeated_reconstruction', mismatches_vs_first_candidate=rpc(_mismatch, 'cand'))
    n, bad = rpc(_set_state, names); note('noop_copy_control_state', from_cand=n, mismatches=bad, mismatches_vs_candidate=rpc(_mismatch, 'cand'))
    guard(2); noop = run('control_noop_candidate_copy_loc', loc)
    n, bad = rpc(_set_state, []); note('reset_to_base_control_state', mismatches=bad, mismatches_vs_base=rpc(_mismatch, 'base'))
    guard(2); rb = run('control_reset_base_loc', loc)
    cb = {x['uid']: x for x in cand}; bb = {x['uid']: x for x in base}
    note('controls', noop_text_identical=sum(x['text'] == cb[x['uid']]['text'] for x in noop), reset_text_identical=sum(x['text'] == bb[x['uid']]['text'] for x in rb), n=len(loc))
    # ---- 3. coarse interventions on LOCALIZATION only
    for g in GROUPS:
        for kind, cn in (('removal', [n for n in names if n not in set(groups[g])]), ('insertion', groups[g])):
            guard(1.5)
            k, bad = rpc(_set_state, cn)
            if bad: note('HYBRID_VERIFY_FAIL', group=g, kind=kind, mismatches=bad); raise SystemExit('hybrid verification failed')
            run(f'{kind}_{g}_loc', loc)
    k, bad = rpc(_set_state, [])
    note('final_restore', mismatches=bad, mismatches_vs_base=rpc(_mismatch, 'base'))
    note('DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['manifest', 'session1'])
    ap.add_argument('--images', type=Path); ap.add_argument('--upstream', type=Path); ap.add_argument('--out', type=Path, default=OUT / 'session1')
    ap.add_argument('--manifest', default=str(OUT / 'example_manifest.json'))
    ap.add_argument('--usd-per-hour', type=float); ap.add_argument('--pod-start-epoch', type=float); ap.add_argument('--cap-usd', type=float)
    a = ap.parse_args()
    make_manifest() if a.stage == 'manifest' else main_session1(a)
