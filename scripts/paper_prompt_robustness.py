"""Part N (paper): official OmniSpatial-style prompt robustness. <= $5 GPU, one H100, one process.

POST_HOC_PROMPT_ROBUSTNESS_DIAGNOSTIC. No search, no new candidates, no reranking.

The prompt and parser follow the official vlms_eval/qwenvl_eval.py at qizekun/OmniSpatial@208bac2:
- prompt = SYS_PROMPTS['manual_cot'] + '\\n' + FORMAT_PROMPTS['re'] + '\\n\\n' + question + '\\nA. ..'
  in one user message [image, text];
- max_new_tokens 8192;
- parser: re.findall(r"Answer\\s*:\\s*([A-D])\\b", IGNORECASE)[-1], falling back to "A".

Model, engine, images and perturbation are identical to the frozen study and the forensic
reproduction.

Steps, each written to disk as soon as it finishes:
 0. Integrity: base state hash; direct-prompt TEST pass must equal the stored base text.
 1. Preflight: manual-CoT greedy on the first 64 TEST questions; projects cost.
 2. Priority 1: base, all 561 TEST, manual-CoT greedy (one deterministic pass).
 3. Priority 2: base RERANK200; candidate 9504111 (sigma 0.002, all parameters) RERANK200 + TEST561.
 4. Optional: base TEST repeats with the checkpoint's own sampling defaults (official
    script behaviour), only while the projected spend stays under budget.
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
MODEL, REV = 'Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
BASE_STATE = '2582817f4966eeaea6723a39f3ef153d18c90c27e2d3adf722625836a45fb3f5'  # forensic Phase Q
CAND_STATE = '7d7ef38b0154ae91e54c01852085101ea7846165e0be35cfbd9dcf9a551dd831'
CAND_SEED, CAND_SIGMA = 9504111, .002
LETTERS = 'ABCD'
PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def rows(split):
    return [json.loads(l) for l in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]


def prompts_module():
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build_prompt(m, r, sys_key, fmt_key):
    text = m.SYS_PROMPTS[sys_key] + '\n' + m.FORMAT_PROMPTS[fmt_key] + '\n\n' + r['question']
    for i, o in enumerate(r['options']):
        text += f'\n{LETTERS[i]}. {o}'
    return text


def official_re(resp):
    """Exactly the official parser: last match, case-insensitive, fallback 'A'. A lower-case match is kept
    as is (and so scored wrong), as in the official code."""
    found = PATTERN.findall(resp)
    return found[-1] if found else 'A'


def strict_re(resp):
    found = PATTERN.findall(resp)
    return found[-1].upper() if found else ''


def state_hash(worker):
    import torch
    h = hashlib.sha256()
    for n, p in worker.model_runner.model.named_parameters():
        h.update(n.encode()); h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--usd-per-hour', type=float, required=True)
    ap.add_argument('--pod-start-epoch', type=float, required=True, help='billing start of the pod (unix seconds)')
    ap.add_argument('--budget', type=float, default=5.0)
    ap.add_argument('--reserve-minutes', type=float, default=8.0, help='allowance for retrieval and stop after the run')
    ap.add_argument('--max-repeats', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'model': MODEL, 'revision': REV, 'usd_per_hour': a.usd_per_hour, 'budget': a.budget, 'steps': []}

    def write(name, obj):
        (a.out / name).write_bytes(gzip.compress(json.dumps(obj).encode()) if name.endswith('.gz') else json.dumps(obj, indent=1).encode())

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); write('cost_log.json', log); print(json.dumps(kw), flush=True)

    path = snapshot_download(MODEL, revision=REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.7, max_model_len=16384, max_num_seqs=128, max_num_batched_tokens=16384,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights')
    proc = AutoProcessor.from_pretrained(path)
    m = prompts_module()
    sets = {'RERANK': rows('validation'), 'TEST': rows('test')}
    imgs = {s: [Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB') for r in rs] for s, rs in sets.items()}

    def reqs(split, sys_key, fmt_key, n=None):
        out = []
        for r, img in list(zip(sets[split], imgs[split]))[:n]:
            p = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': build_prompt(m, r, sys_key, fmt_key)}]}],
                                         tokenize=False, add_generation_prompt=True)
            out.append({'prompt': p, 'multi_modal_data': {'image': img}})
        return out

    def run(split, sp, sys_key='manual_cot', fmt_key='re', n=None):
        t = time.time()
        res = llm.generate(reqs(split, sys_key, fmt_key, n), sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in res]
        finish = [o.outputs[0].finish_reason for o in res]
        ntok = [len(o.outputs[0].token_ids) for o in res]
        gold = [LETTERS[r['answer']] for r in sets[split][:len(texts)]]
        sub = [r['sub_task_type'] for r in sets[split][:len(texts)]]
        if fmt_key == 'direct':
            pred = [t_.strip().upper()[:1] for t_ in texts]; spred = pred
        else:
            pred = [official_re(t_) for t_ in texts]; spred = [strict_re(t_) for t_ in texts]
        by = {}
        for s_, p_, g_ in zip(sub, pred, gold):
            by.setdefault(s_, [0, 0]); by[s_][0] += p_ == g_; by[s_][1] += 1
        return {'split': split, 'n': len(texts), 'correct_official': sum(p == g for p, g in zip(pred, gold)),
                'correct_strict': sum(p == g for p, g in zip(spred, gold)),
                'no_answer_match': sum(1 for x in spred if not x), 'truncated': sum(f == 'length' for f in finish),
                'by_subtask_official': by, 'gen_tokens_total': sum(ntok), 'gen_tokens_max': max(ntok), 'seconds': time.time() - t,
                'uids': [r['uid'] for r in sets[split][:len(texts)]], 'texts': texts, 'pred_official': pred, 'pred_strict': spred,
                'finish': finish}

    def summary(r):
        return {k: v for k, v in r.items() if k not in ('uids', 'texts', 'pred_official', 'pred_strict', 'finish')}

    greedy = SamplingParams(temperature=0, seed=0, max_tokens=8192)
    note('model_loaded')

    # 0. Integrity
    st = rpc(state_hash)
    if st != BASE_STATE: raise SystemExit(f'base state mismatch {st}')
    stored = [o['text'] for o in load(ROOT / 'baseline/base.json.gz')['test']['outputs']]
    d = run('TEST', SamplingParams(temperature=0, seed=0, max_tokens=16), 'none', 'direct')
    d['identical_to_stored_base'] = d['texts'] == stored
    d['mismatched'] = sum(x != y for x, y in zip(d['texts'], stored))
    write('step0_direct_base_test.json.gz', d)
    note('integrity', base_state_ok=True, direct_test_correct=d['correct_official'], direct_identical=d['identical_to_stored_base'], mismatched=d['mismatched'])

    # 1. Preflight
    pf = run('TEST', greedy, n=64)
    per_q = pf['seconds'] / 64
    proj = {'p1_s': 561 * per_q, 'p2_s': (200 + 200 + 561) * per_q + 60, 'repeat_s': 561 * per_q}
    reserve = a.reserve_minutes / 60 * a.usd_per_hour
    proj['projected_p1_p2_usd'] = spent() + (proj['p1_s'] + proj['p2_s']) / 3600 * a.usd_per_hour + reserve
    write('step1_preflight.json', {**summary(pf), **proj})
    note('preflight', seconds_per_question=round(per_q, 3), mean_gen_tokens=pf['gen_tokens_total'] / 64, **{k: round(v, 2) for k, v in proj.items()})

    def affordable(seconds):
        return spent() + seconds / 3600 * a.usd_per_hour * 1.3 + reserve <= a.budget

    # 2. Priority 1
    if not affordable(proj['p1_s']): note('stop_budget_before_p1'); return
    p1 = run('TEST', greedy)
    write('p1_base_test_manualcot_greedy.json.gz', p1); note('p1_base_test', **{k: v for k, v in summary(p1).items() if k != 'by_subtask_official'})
    per_q = p1['seconds'] / 561

    # 3. Priority 2
    if not affordable(961 * per_q + 60): note('stop_budget_before_p2'); return
    b2 = run('RERANK', greedy)
    write('p2_base_rerank_manualcot_greedy.json.gz', b2); note('p2_base_rerank', **{k: v for k, v in summary(b2).items() if k != 'by_subtask_official'})
    rpc('apply_perturbation', CAND_SEED, CAND_SIGMA)
    cs = rpc(state_hash)
    if cs != CAND_STATE: rpc('reset_to_base_weights'); raise SystemExit(f'candidate state mismatch {cs}')
    for split in ('RERANK', 'TEST'):
        c = run(split, greedy); c['state_sha256'] = cs
        write(f'p2_cand9504111_{split.lower()}_manualcot_greedy.json.gz', c)
        note(f'p2_candidate_{split.lower()}', **{k: v for k, v in summary(c).items() if k != 'by_subtask_official'})
    rpc('reset_to_base_weights')
    if rpc(state_hash) != BASE_STATE: raise SystemExit('base not restored')
    note('base_restored')

    # 4. Optional sampled repeats (official script uses the checkpoint's generation defaults)
    sp_def = llm.get_default_sampling_params()
    for k in range(1, a.max_repeats + 1):
        if not affordable(561 * per_q * 1.1): note('stop_budget_repeats', completed=k - 1); break
        sp = sp_def.clone(); sp.max_tokens = 8192; sp.seed = k
        r = run('TEST', sp); r['sampling'] = repr(sp)
        write(f'p1_base_test_manualcot_sampled_seed{k}.json.gz', r)
        note(f'repeat_{k}', **{kk: v for kk, v in summary(r).items() if kk != 'by_subtask_official'})
    note('DONE')


if __name__ == '__main__':
    main()
