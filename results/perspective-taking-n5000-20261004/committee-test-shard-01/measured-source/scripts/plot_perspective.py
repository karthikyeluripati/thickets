"""Figures 1-6 for the Perspective-Taking N=5000 report (static PNG, light surface)."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SIGMAS = [.00025, .0005, .001, .002]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
read = lambda f: json.loads(gzip.decompress(f.read_bytes()) if f.suffix == '.gz' else f.read_bytes())
base = read(a.root / 'locks/baseline.json')['summary']
search, val = read(a.root / 'locks/search.json'), read(a.root / 'locks/validation.json')
final = read(a.root / 'analysis/final_candidates.json')
test_base = read(a.root / 'analysis/test_base.json')
decision = read(a.root / 'analysis/decision.json')
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 10, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(a.out / name, dpi=160)
    plt.close(fig)


# Figure 1: SEARCH accuracy distribution, one panel per sigma (small multiples).
b = base['search']['correct_count']
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
bins = np.arange(min(r['correct_count'] for r in search['records']) - .5, max(r['correct_count'] for r in search['records']) + 1.5)
for ax, s in zip(axes, SIGMAS):
    c = [r['correct_count'] for r in search['records'] if r['candidate']['sigma'] == s]
    ax.hist(c, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=.8)
    ax.axvline(b, color=INK, lw=1.2, ls='--')
    ax.set_title(f'σ = {s:g}  (n={len(c)}, max {max(c)})', fontsize=10)
    ax.set_xlabel('SEARCH correct / 200')
axes[0].set_ylabel('candidates')
fig.suptitle('Figure 1. SEARCH200 accuracy of all 5,000 candidates (dashed line: base 72/200)', x=.01, ha='left', fontsize=11)
save(fig, 'figure1_search_distribution.png')

# Figure 2: SEARCH gain vs VALIDATION gain for the frozen top 50.
st = val['search_to_validation']
fig, ax = plt.subplots(figsize=(6, 4.6))
x, y = 100 * np.array(st['search_gain']), 100 * np.array(st['validation_gain'])
ax.axhline(0, color=INK2, lw=.8); ax.axhline(5, color=ORANGE, lw=.8, ls=':')
ax.scatter(x + np.random.default_rng(0).uniform(-.12, .12, len(x)), y, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
ax.set_xlabel('SEARCH gain over base (pp; ±0.12 jitter to separate ties)'); ax.set_ylabel('VALIDATION gain over base (pp)')
ax.set_title(f"Figure 2. Top 50: search → validation (Pearson {st['pearson']:.2f}, Spearman {st['spearman']:.2f})", loc='left', fontsize=10)
ax.text(ax.get_xlim()[1], 5, '+5 pp ', color=ORANGE, ha='right', va='bottom', fontsize=9)
save(fig, 'figure2_search_vs_validation.png')

# Figure 3: VALIDATION gain vs TEST gain for the frozen top 10.
fig, ax = plt.subplots(figsize=(6, 4.6))
vx, ty = 100 * np.array([r['validation_gain'] for r in final]), 100 * np.array([r['gain'] for r in final])
lo = 100 * np.array([r['paired']['paired_bootstrap95'][0] for r in final]); hi = 100 * np.array([r['paired']['paired_bootstrap95'][1] for r in final])
ax.axhline(0, color=INK2, lw=.8); ax.axhline(5, color=ORANGE, lw=.8, ls=':')
ax.errorbar(vx, ty, yerr=[ty - lo, hi - ty], fmt='o', color=BLUE, ecolor=GRID, elinewidth=2, ms=7, mec=SURFACE, zorder=3)
for r, xx, yy in zip(final, vx, ty):
    ax.annotate(str(r['validation_rank']), (xx, yy), xytext=(5, 4), textcoords='offset points', fontsize=8, color=INK2)
ax.set_xlabel('VALIDATION gain (pp)'); ax.set_ylabel('FINAL TEST gain (pp), paired 95% CI')
ax.set_title('Figure 3. Frozen top 10: validation → official test (labels = validation rank)', loc='left', fontsize=10)
save(fig, 'figure3_validation_vs_test.png')

# Best candidate for figures 4-5: best transferable expert, else frozen validation rank 1.
ranks = decision['transferable_expert_ranks']
best = final[ranks[0] - 1] if ranks else final[0]
label = f"{'best expert' if ranks else 'frozen rank 1'} (seed {best['candidate']['seed']}, σ={best['candidate']['sigma']:g})"

# Figure 4: base vs best by sub-task on TEST.
subs = sorted(test_base['sub_task'])
bv = [100 * test_base['sub_task'][s]['accuracy'] for s in subs]
cv = [100 * best['summary']['sub_task'][s]['accuracy'] for s in subs]
fig, ax = plt.subplots(figsize=(7, 4))
w, idx = .38, np.arange(len(subs))
ax.bar(idx - w / 2 - .01, bv, w, color=BLUE, label='base')
ax.bar(idx + w / 2 + .01, cv, w, color=ORANGE, label=label)
for i, (u, v) in enumerate(zip(bv, cv)):
    ax.text(i + w / 2, v + 1, f'{v - u:+.1f}', ha='center', fontsize=9, color=INK)
ax.set_xticks(idx, [f"{s}\n(n={test_base['sub_task'][s]['n']})" for s in subs]); ax.set_ylabel('TEST accuracy (%)')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2); ax.set_ylim(0, 100); ax.set_title('Figure 4. Official TEST accuracy by Perspective-Taking sub-task', loc='left', fontsize=10)
save(fig, 'figure4_subtask_base_vs_best.png')

# Figure 5: prediction marginals vs true answer distribution on TEST.
pq = read(a.root / 'analysis/per_question.json.gz')
letters = ['A', 'B', 'C', 'D']
cid = best['candidate']['candidate_id']
true = [pq['true_letter'].count(l) for l in letters]
bp = [pq['base_prediction'].count(l) for l in letters]
cp = [pq['candidates'][cid]['prediction'].count(l) for l in letters]
fig, ax = plt.subplots(figsize=(7, 4))
w = .27
ax.bar(np.arange(4) - w - .01, true, w, color=AQUA, label='true answers')
ax.bar(np.arange(4), bp, w, color=BLUE, label='base predictions')
ax.bar(np.arange(4) + w + .01, cp, w, color=ORANGE, label=label)
ax.set_xticks(range(4), letters); ax.set_ylabel('TEST questions'); ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
share = best['label_audit']['largest_positive_gain_share']
ax.set_title(f"Figure 5. Answer-letter distributions on TEST (largest single-letter gain share: {share:.0%})" if share is not None
             else 'Figure 5. Answer-letter distributions on TEST', loc='left', fontsize=10)
save(fig, 'figure5_answer_distributions.png')

# Figure 6: density-audit VALIDATION gains by sigma (small multiples).
gains = val['density_audit']['gains_by_sigma']
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
allg = 100 * np.concatenate([np.array(v) for v in gains.values()])
bins = np.arange(np.floor(allg.min()) - .25, np.ceil(allg.max()) + .75, .5)
for ax, s in zip(axes, SIGMAS):
    g = 100 * np.array(gains[str(s)])
    ax.hist(g, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=.8)
    ax.axvline(0, color=INK, lw=1.2, ls='--')
    ax.set_title(f'σ = {s:g}  (n={len(g)}, ≥+3 pp: {(g >= 3 - 1e-9).sum()})', fontsize=10)
    ax.set_xlabel('VALIDATION gain (pp)')
axes[0].set_ylabel('candidates')
fig.suptitle('Figure 6. Random density audit (500 precommitted candidates): validation gain by sigma', x=.01, ha='left', fontsize=11)
save(fig, 'figure6_density_audit.png')
print(sorted(f.name for f in a.out.glob('*.png')))
