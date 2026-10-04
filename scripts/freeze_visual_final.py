"""Create the immutable final v1 experiment protocol from committed references."""
import gzip
import json
from pathlib import Path
import re
import subprocess

from thicket_runtime.line_tracing import load_split, sha
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.visual_final import rank_hash, verify_inputs
from thicket_runtime.visual_v21 import population_id

parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if parent != 'a6cc79659cd204c9ed4aebf02d941a8c1edf583c':
    raise ValueError('freeze must begin from the requested published head')
old = read('experiments/visual_line_tracing_protocol.json')
reference = Path('results/visual-line-tracing-20261003')
baseline = read(reference/'locks/baseline.json')
calibration = read(reference/'calibration-01/records.json')
best = sorted(calibration, key=lambda r: (-r['correct_count'], rank_hash(r['candidate']['candidate_id'])))[0]
keys = ('model','model_revision','processor_revision','upstream_commit','dtype','tensor_parallel_size',
        'temperature','sampling_seed','max_tokens','max_model_len','max_num_seqs','max_num_batched_tokens',
        'gpu_memory_utilization','image_pixels','perturbation','cache_contract','heldout')
seeds = set()
for path in Path('results').rglob('*'):
    if path.is_file() and (path.suffix == '.json' or path.name.endswith('.json.gz')):
        raw = path.read_bytes()
        if path.suffix == '.gz': raw = gzip.decompress(raw)
        seeds.update(int(s) for s in re.findall(rb'"seed"\s*:\s*(\d+)', raw))
# Exclude the previous protocols' reserved search seeds even though they never ran.
seeds.update(range(3100000,3100060)); seeds.update(range(4100000,4100300))
p = {'schema':'visual-line-tracing-final-v1','parent_commit':parent,
     'hypothesis':'A frozen selection-rank-1 nearby random weight perturbation yields a material held-out visual line-tracing improvement.',
     'dataset':old['dataset'], **{k:old[k] for k in keys},
     'dataset_pin':{'commit':'49bb56e160691e83c786fc7bac40be290c0e8f75','manifest_sha256':sha(Path(old['dataset'])/'manifest.json')},
     'expected_base_id':baseline['base_id'],
     'required_versions':read(reference/'calibration-01/run-manifest.json')['versions'],
     'reference':{'index':str(reference/'sha256.json').replace('\\','/'),'index_sha256':sha(reference/'sha256.json'),
                  'baseline':str(reference/'baseline-v1').replace('\\','/'),
                  'baseline_lock':str(reference/'locks/baseline.json').replace('\\','/'),
                  'calibration_records':str(reference/'calibration-01/records.json').replace('\\','/')},
     'populations':{s:{'n':len(rows),'sha256':population_id(rows)} for s in ('selection','heldout') for rows in [load_split(old['dataset'],s)]},
     'sigma':.0005,'sigma_rationale':'User-fixed from the existing v1 selection right tail; no new calibration or outcome-dependent adjustment.',
     'previous_candidate_seeds':sorted(seeds),
     'diagnostic':{'enabled':True,'selected':best,'heldout_examples':500,'influences_decision_or_search':False},
     'search':{**old['search'],'seed_start':6100000},
     'gate':{'single_expert_gain':.05,'bootstrap_lower_bound_strictly_positive':True,
             'difficulty_levels_with_positive_gain':2,'single_label_positive_net_gain_share_must_be_below':.8,
             'replication_count_at_least_3pp':3,'replication_mean_gain':.03,'replication_one_gain':.05,
             'ensemble_gain_descriptive_only':.07,'rule':'Primary GO, otherwise replication GO, otherwise NO_GO_VISUAL_THICKET_LINE_TRACING; operational failures are not scientific NO-GO.',
             'qualitative_rule_resolution':'Single-label concentration uses positive per-true-label net correct-count gains; no class may account for >=80%. Report every class confusion and prediction marginal. If neither predefined GO passes, report NO-GO and flag any >=5pp rank-1 lacking other primary checks.'},
     'statistics':{'bootstrap_replicates':10000,'bootstrap_seed':20261003,'method':'paired question percentile bootstrap; common 10000x500 PCG64 indices, dtype=int64; 2.5/97.5 percentiles, linear quantile',
                   'mcnemar':'exact two-sided binomial test of discordant pairs, p=0.5; no p-value effect-size substitution',
                   'secondary_intervals':'descriptive, unadjusted; top-10 are selected on the same selection set and share the same held-out questions, not independent replications'},
     'controls':{'zero':'Search selection150 and heldout500 zero runs match old committed base outputs; zero heldout is the required base identity verification, not a new baseline sweep.',
                 'repeats':'First search candidate repeated on selection150; rank-1 repeated on heldout500; diagnostic state matches old calibration fingerprint.',
                 'all_candidates':'Exact snapshot before/after; native state and scope checked; heldout fingerprint checked before generation; encoder freshness for every image.',
                 'budget':'300x150 search +10x500 heldout +1x500 diagnostic; extra control requests only: zero150+zero500+repeat150+repeat500=1300. No new baseline/sigma sweeps.'},
     'no_early_scientific_stop':True,'no_capability_gate':True,'no_population_mean_gate':True,
     'stop':'Finish all 300 candidates and frozen top10 heldout regardless of base, diagnostic, or population means. No rescue or additional experiments.'}
verify_inputs(p)
path = Path('experiments/visual_line_tracing_final_protocol.json')
write(path,p)
root = Path('results/visual-line-tracing-final-20261003')
root.mkdir(exist_ok=False)
(root/'.gitattributes').write_bytes(b'* -text -whitespace\n')
write(root/'locks/sigma.json',{'protocol_sha256':sha(path),'dataset_manifest_sha256':p['dataset_pin']['manifest_sha256'],
      'sigma':p['sigma'],'chosen_by':'explicit user instruction from existing selection-only calibration',
      'source_calibration_sha256':sha(reference/'calibration-01/records.json'),'diagnostic':best,
      'base_output_identities':baseline['output_identities'],'base_summary':baseline['summary'],
      'final_candidate_seeds':list(range(6100000,6100300)),'no_scientific_presearch_gate':True})
print({'protocol_sha256':sha(path),'diagnostic_seed':best['candidate']['seed'],'new_candidates':300,'prior_seeds_excluded':len(seeds)})
