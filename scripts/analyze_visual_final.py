"""Audit every final v1 output and apply the precommitted paired decision."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha, vote
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.visual_final import decision, label_audit, paired, rank_selection, summarize, verify_inputs
from thicket_runtime.visual_final_audit import outputs, phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_final_protocol.json'))
p.add_argument('--out',type=Path,required=True)
a = p.parse_args()
if a.out.exists(): raise FileExistsError(a.out)
committed(a.protocol)
protocol = read(a.protocol)
proof = verify_inputs(protocol)
lock_path = a.root/'locks/selection.json'
committed(lock_path)
selected = read(lock_path)
search = phase(a.root/'search-01',protocol,a.protocol)
ranked = rank_selection(search['records'],protocol)
if (selected['ranked'] != ranked or selected['top10'] != ranked[:10] or selected['top5'] != ranked[:5] or
    selected['best'] != ranked[0] or selected['search_manifest_sha256'] != sha(a.root/'search-01/run-manifest.json') or
    selected['search_checksums_sha256'] != sha(a.root/'search-01/sha256.json')):
    raise ValueError('committed ranking does not reproduce from selection only')
heldout = phase(a.root/'heldout-01',protocol,a.protocol,selected)
diagnostic = phase(a.root/'diagnostic-01',protocol,a.protocol)
manifest = heldout['manifest']
lock_commit = manifest['selection_lock_commit']
if subprocess.check_output(['git','show',lock_commit+':'+lock_path.as_posix()]) != lock_path.read_bytes():
    raise ValueError('heldout did not use the committed ranking bytes')
subprocess.run(['git','merge-base','--is-ancestor',search['manifest']['source']['commit'],lock_commit],check=True)
subprocess.run(['git','merge-base','--is-ancestor',lock_commit,manifest['source']['commit']],check=True)
lock_time = subprocess.check_output(['git','show','-s','--format=%cI',lock_commit],text=True).strip()
if datetime.fromisoformat(lock_time) >= datetime.fromisoformat(manifest['started_utc']):
    raise ValueError('ranking was not committed before held-out inference')
rows = heldout['datasets']['heldout']
historical = read(Path(protocol['reference']['baseline'])/'base.json.gz')
base_outputs = outputs(historical['heldout'],rows)
base = summarize(base_outputs,rows)
if base['correct_count'] != 133: raise ValueError('reference base differs')
indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
    0,500,size=(protocol['statistics']['bootstrap_replicates'],500),dtype=np.int64)
limit = protocol['gate']['single_label_positive_net_gain_share_must_be_below']
individual, values = [], []
for rank,record in enumerate(ranked[:10],1):
    cid = record['candidate']['candidate_id']
    outs = heldout['traces'][cid]['splits']['heldout']['outputs']
    values.append(outs)
    metric = summarize(outs,rows)
    individual.append({'rank':rank,'candidate':record['candidate'],'candidate_state_id':record['candidate_state_id'],
                       'selection_correct_count':record['correct_count'],'selection_accuracy':record['correct_count']/150,
                       'heldout':metric,'comparison':paired(base_outputs,outs,indices),
                       'difficulty_gain':{l:metric['difficulty'][l]['accuracy']-base['difficulty'][l]['accuracy'] for l in ('easy','medium','hard')},
                       'label_audit':label_audit(base_outputs,outs,rows,limit)})
diag_raw = next(iter(diagnostic['traces'].values()))
diag_outputs = diag_raw['splits']['heldout']['outputs']
diag = {'candidate':diag_raw['candidate'],'candidate_state_id':diag_raw['candidate_state_id'],
        'selection_correct_count':50,'heldout':summarize(diag_outputs,rows),'comparison':paired(base_outputs,diag_outputs,indices),
        'label_audit':label_audit(base_outputs,diag_outputs,rows,limit),'diagnostic_only':True}
ensembles, votes, ensemble_vectors = {}, {}, {}
for k in (5,10):
    raw_votes = [vote([v[i]['answer'] for v in values[:k]]) for i in range(500)]
    outs = [{'id':row['id'],'answer':v['answer'],'correct':v['answer']==row['answer'],'finish_reason':'vote'} for row,v in zip(rows,raw_votes)]
    metric = summarize(outs,rows)
    comparison = paired(base_outputs,outs,indices)
    ensembles[str(k)] = {'summary':metric,'comparison':comparison,
                         'mean_individual_accuracy':sum(r['heldout']['accuracy'] for r in individual[:k])/k,
                         'at_least_7pp_additional_result':comparison['gain']>=.07-1e-12,
                         'label_audit':label_audit(base_outputs,outs,rows,limit)}
    votes[str(k)] = raw_votes
    ensemble_vectors[str(k)] = [o['correct'] for o in outs]
counts = [r['correct_count'] for r in ranked]
scores = np.asarray(counts)/150
base_sel = sum(o['correct'] for o in historical['selection']['outputs'])/150
distribution = {'n':300,'base_selection_accuracy':base_sel,'mean_accuracy':float(scores.mean()),
                'standard_deviation':float(scores.std(ddof=1)),'minimum_accuracy':float(scores.min()),
                'best_accuracy':float(scores.max()),'median_accuracy':float(np.median(scores)),
                'quantiles_5_25_50_75_95':np.quantile(scores,[.05,.25,.5,.75,.95]).tolist(),
                'strictly_above_base':int(np.sum(scores>base_sel+1e-12)),
                'at_least_1_3_5pp_counts':{str(pp):int(np.sum(scores>=base_sel+pp/100-1e-12)) for pp in (1,3,5)},
                'histogram_correct_counts':dict(sorted(Counter(counts).items())),
                'descriptive_only_no_population_gate':True}
gate = decision(individual,base,protocol,True)
a.out.mkdir(parents=True)
write(a.out/'manifest.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
      'protocol_sha256':sha(a.protocol),'dataset_proof':proof,'all_raw_generations_rescored':True,
      'ranking_reselected':False,'selection_lock_commit':lock_commit,'selection_lock_committed_at':lock_time,
      'phase_manifest_sha256':{s:sha(a.root/(s+'-01')/'run-manifest.json') for s in ('diagnostic','search','heldout')},
      'numpy':np.__version__,'bootstrap_indices_sha256':__import__('hashlib').sha256(indices.tobytes()).hexdigest(),
      'control_image_requests':1300,'nonzero_distinct_candidates':311,
      'statistical_limitation':'Fixed 500 questions; percentile bootstrap unadjusted secondary comparisons. Top-10 experts are correlated, selection-enriched, and not independent replication cohorts.'})
for name,value in [('base',base),('diagnostic',diag),('distribution',distribution),('individual_experts',individual),
                   ('ensembles',ensembles),('votes',votes),('gate',gate)]:
    write(a.out/(name+'.json'),value)
write(a.out/'per_question.json.gz',{'ids':[r['id'] for r in rows],'true_labels':[r['answer'] for r in rows],
      'base_correct':[o['correct'] for o in base_outputs],
      'individual_correct':[[o['correct'] for o in v] for v in values],
      'diagnostic_correct':[o['correct'] for o in diag_outputs],'ensemble_correct':ensemble_vectors})
lines = ['# Frozen final v1 results','','| Rank | Seed | Selection | Held-out | Gain (pp) | Paired 95% CI (pp) | Wins / losses |',
         '|---|---|---|---|---|---|---|']
for r in individual:
    c = r['comparison']; lo,hi = c['paired_bootstrap95']
    lines.append(f"| {r['rank']} | {r['candidate']['seed']} | {r['selection_correct_count']}/150 | {r['heldout']['correct_count']}/500 | {100*c['gain']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {c['candidate_only_correct']} / {c['base_only_correct']} |")
(a.out/'tables.md').write_bytes(('\n'.join(lines)+'\n').encode())
print(gate)
