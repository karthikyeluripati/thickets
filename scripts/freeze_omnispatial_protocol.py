"""Create the immutable OmniSpatial visual-expert protocol before any OmniSpatial inference."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from thicket_runtime.line_tracing import sha
from thicket_runtime.omnispatial import candidate_plan, ident, verify_inputs
from thicket_runtime.omnispatial_data import (DATASET, DATASET_REVISION, OFFICIAL_BLOBS, REPO, REPO_COMMIT, SPLIT_SEED,
                                              ZIPS, build_prompt, load, sha256)
from thicket_runtime.omnispatial_runtime import QWEN_VL_UTILS_WHEEL_SHA256
from thicket_runtime.visual_runtime import read, write

DATA = 'examples/omnispatial-complex-logic'
DATA_COMMIT = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', DATA], text=True).strip()
parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
old = read('experiments/transfer_aware_visual_protocol.json')
seeds = set(old['previous_candidate_seeds']) | set(range(7300000, 7300400))
for path in Path('results').rglob('*'):
    if path.is_file() and (path.suffix == '.json' or path.name.endswith('.json.gz')):
        raw = path.read_bytes()
        if path.suffix == '.gz': raw = gzip.decompress(raw)
        seeds.update(int(s) for s in re.findall(rb'"seed"\s*:\s*(\d+)', raw))
files = len(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', DATA_COMMIT, '--', DATA], text=True).split())
sets = {s: load(DATA, s) for s in ('search', 'validation', 'test')}
p = {'schema': 'omnispatial-visual-expert-v1', 'parent_commit': parent,
     'question': 'Does the local weight neighborhood of Qwen2.5-VL-7B contain a genuinely better, transferable visual-spatial expert on OmniSpatial Complex Spatial Logic?',
     'benchmark': {'repository': REPO, 'repository_commit': REPO_COMMIT, 'official_blobs': OFFICIAL_BLOBS,
                   'dataset': DATASET, 'dataset_revision': DATASET_REVISION, 'zips': {s: {'file': f, 'lfs_sha256': h} for s, (f, h) in ZIPS.items()},
                   'dimension': 'Complex_Logic (all QA pairs; no sub-task selection)',
                   'official_split_use': 'official train -> SEARCH 60% / VALIDATION 40%; official test -> FINAL TEST 100%'},
     'dataset': DATA, 'dataset_pin': {'commit': DATA_COMMIT, 'manifest_sha256': sha256((Path(DATA)/'manifest.json').read_bytes()), 'files': files},
     'split_seed': SPLIT_SEED,
     'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'model_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
     'processor_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
     **{k: old[k] for k in ('upstream_commit', 'dtype', 'tensor_parallel_size', 'temperature', 'sampling_seed', 'perturbation', 'required_versions')},
     'max_tokens': 16,
     'engine': {'gpu_memory_utilization': .5, 'max_model_len': 17408, 'max_num_seqs': 32, 'max_num_batched_tokens': 17408,
                'min_pixels': 3136, 'max_pixels': 12845056, 'enforce_eager': True, 'prefix_caching': False,
                'mm_preprocessor_cache': False},
     'evaluation': {'path': 'official vlms_eval/qwenvl_eval.py with --prompt_type none --eval_type direct',
                    'prompt': "SYS_PROMPTS['none'] + '\\n' + FORMAT_PROMPTS['direct'] + '\\n\\n' + question + ''.join('\\n{L}. {option}'); user message [image, text]; Qwen chat template with generation prompt",
                    'prompt_source': 'vendored third_party/omnispatial/system_prompts.py, git blob ' + OFFICIAL_BLOBS['vlms_eval/system_prompts.py'],
                    'image_preprocessing': f'official qwen_vl_utils==0.0.8 fetch_image (RGB conversion + smart_resize, factor 28, min 3136, max 12845056 pixels); wheel sha256 {QWEN_VL_UTILS_WHEEL_SHA256}; resized pixels hashed per image',
                    'scorer': "official direct: prediction = response.strip().upper()[:1]; correct iff prediction == chr(65+answer); no fallback letter; non A-D first character counts as invalid and incorrect",
                    'decoding': 'greedy, temperature 0, max_tokens 16. The official script allows 8192 new tokens; greedy decoding is causal and the direct scorer reads only the first non-space character, so a 16-token cap cannot change a score unless the first 16 tokens are all whitespace (finish reasons recorded).',
                    'repeats': 'official script averages 5 sampling repeats; greedy decoding makes repeats identical, so one pass is used',
                    'frozen': 'This prompt/evaluation configuration is fixed before any nonzero candidate inference and is never modified.',
                    'prompt_sha256_by_split': {s: hashlib.sha256(json.dumps([build_prompt(r) for r in rows], ensure_ascii=False).encode()).hexdigest() for s, rows in sets.items()}},
     'populations': {s: {'n': len(rows), 'sha256': ident(rows)} for s, rows in sets.items()},
     'candidates': {'count': 400, 'seed_start': 9100000, 'sigma_mixture': [.00025, .0005, .001, .002],
                    'rule': 'candidate i=0..399: seed 9100000+i, sigma sigma_mixture[i % 4]; exactly 100 per sigma; no sigma sweep; never changed after results',
                    'shard_size': 50, 'identity': '(seed, sigma, base_id, RNG contract randopt-per-tensor-v1)',
                    'reconstruction': 'each candidate from the identical base snapshot; bitwise reset afterwards'},
     'search': {'split': 'search', 'top_k': 30,
                'ranking': "SEARCH correct count descending (fixed n, equivalent to accuracy), then SHA256('visual-rank-v1:' + candidate_id) ascending",
                'freeze': 'commit all 400 records, ranking and top-30 identities/fingerprints before validation inference',
                'no_early_stop': 'never stop on population mean, degradation or one sigma'},
     'validation': {'split': 'validation', 'ranking': 'VALIDATION correct count descending, then frozen search rank',
                    'freeze': 'commit validation scores, rank 1, top 5 and fingerprints before any final-test candidate inference'},
     'test': {'candidates': 'validation rank 1 and ranks 2-5 only; no other candidate after test results are observed'},
     'gate': {'expert_gain': .05, 'positive_sub_tasks': 2, 'answer_share_limit': .8,
              'expert': 'TRANSFERABLE_OMNISPATIAL_EXPERT: final-test gain >= +5 pp; paired bootstrap 95% lower bound > 0; strictly positive accuracy gain in >= 2 fine-grained sub-task types (the official test subset has exactly two: Geometric_Reasoning and Pattern_Recognition, so both must improve); no single true-answer letter supplies >= 80% of positive net correctness gains; all runtime controls pass.',
              'go': 'GO_OMNISPATIAL_VISUAL_EXPERT if any frozen top-5 candidate is a TRANSFERABLE_OMNISPATIAL_EXPERT; otherwise NO_GO_OMNISPATIAL_VISUAL_EXPERT.',
              'answer_bias': 'positive_gain_share(letter) = max(0, net correct gain on that TRUE letter) / sum over letters; expert-validity criterion only, never used to rerank',
              'secondary': 'frozen top-30 validation counts > base, >= +1, >= +3, >= +5 pp; validation-enriched, not unbiased density'},
     'statistics': {'bootstrap_replicates': 10000, 'bootstrap_seed': 20261003,
                    'method': 'paired question percentile bootstrap, common PCG64 int64 indices per split, 2.5/97.5 linear percentiles; exact two-sided McNemar on discordant pairs'},
     'controls': {'zero': 'zero perturbation reproduces base outputs on every evaluated split in every phase; baseline repeat equals base',
                  'repeat': 'first candidate of every shard, validation and test phase reconstructed again with identical state and outputs',
                  'restore': 'bitwise exact snapshot restoration after every candidate (parameters and buffers)',
                  'fingerprints': 'validation and test reject a candidate before generation unless its state fingerprint equals its search fingerprint',
                  'encoding': 'encoder cache cleared before each generation; every image freshly encoded; unique multimodal UUIDs; prefix and multimodal preprocessor caches disabled',
                  'data': 'pinned zips verified by LFS sha256, every image by sha256; per-image resized-pixel and prompt hashes equal the baseline in every phase',
                  'locks': 'dataset, protocol, baseline, search and validation locks committed before the phase that consumes them; analysis checks commit times',
                  'operational': 'any runtime/control failure is operational, never a scientific NO-GO'},
     'audit_findings': read(Path(DATA)/'manifest.json')['audit_findings'],
     'budget': {'baseline': '3 x (705 + 475 + 252) requests (base, zero, base repeat)', 'search': '400 x 705 + 8 x 2 x 705 control',
                'validation': '30 x 475 + 2 x 475 control', 'test': '5 x 252 + 2 x 252 control'},
     'no_base_capability_gate': True,
     'out_of_scope': 'No new optimizer, BO/CMA-ES, model scaling, extra candidates or sigmas, subset or prompt change, or other search algorithm.',
     'previous_candidate_seeds': sorted(seeds)}
if set(s for s, _ in candidate_plan(p)) & seeds: raise ValueError('candidate seed reuse')
verify_inputs(p)
path = Path('experiments/omnispatial_visual_expert_protocol.json')
write(path, p)
print({'protocol_sha256': sha(path), 'dataset_commit': DATA_COMMIT, 'files': files, 'prior_seeds_excluded': len(seeds),
       'populations': {s: v['n'] for s, v in p['populations'].items()}})
