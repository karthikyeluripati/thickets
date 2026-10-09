"""Figure 1: (a) RandOpt vs the cheap route (schematic); (b, c) test accuracy on the two rows where RandOpt beats SC
under its own prompt. Numbers come from the locked result files via thickets_or_tilts.same_run_rows().
Run from the repo root: python paper/figures/scripts/fig_overview.py"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from thickets_or_tilts import C, j, same_run_rows, save  # noqa: E402

NL = '\n'


def main():
    R = {r['name']: r for r in same_run_rows()}
    G4, O2 = j('g4-direct-prompt/pod/g4/g4_results.json'), j('o2-olmo-prompt/pod/o2/o2_results.json')
    rows = [('GQA / Qwen2.5-VL-3B', R['GQA' + NL + 'Qwen2.5-VL-3B'], G4['direct']['base'], 'direct'),
            ('GSM8K / OLMo-2-1B', R['GSM8K' + NL + 'OLMo-2-1B'], O2['boxed']['base'], 'boxed')]
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.32, left=0.02, right=0.98, top=0.86, bottom=0.27)

    # (a) schematic
    ax = fig.add_subplot(gs[0]); ax.set_xlim(-0.3, 10.4); ax.set_ylim(-0.3, 10.6); ax.axis('off')

    def box(x, y, w, h, txt, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1,rounding_size=0.25', fc='white', ec=col, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=6.0, color=C['ink'], linespacing=1.15)

    def arrow(x0, x1, y, col):
        ax.annotate('', (x1, y), (x0, y), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.0, mutation_scale=8))
    flows = [(6.3, C['randopt'], 'RandOpt (weight search)',
              ['5000 random' + NL + 'perturbations', 'score each on' + NL + '200 questions', 'vote of' + NL + 'the top 50'],
              '1,000,000 selection generations'),
             (1.0, C['green'], 'Cheap route (no weight search)',
              ['3 candidate' + NL + 'prompts', 'pick one on' + NL + 'the same 200' + NL + 'questions', 'vote of' + NL + '50 samples'],
              '600 selection generations')]
    for y0, col, head, items, cost in flows:
        ax.text(0.0, y0 + 3.15, head, fontsize=7.2, fontweight='bold', color=col, va='bottom')
        for i, t in enumerate(items):
            x = 0.15 + i * 3.55
            box(x, y0, 2.7, 2.3, t, col)
            if i < 2:
                arrow(x + 2.85, x + 3.45, y0 + 1.15, col)
        ax.text(5.1, y0 - 0.75, cost, ha='center', fontsize=6.0, color=C['muted'])
    ax.text(-0.02, 1.13, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    # (b, c) accuracy ladder
    bars = [("RandOpt (its own prompt)", C['randopt']), ('one answer, prompt chosen on the 200 questions', C['sky']),
            ('SC@50, chosen prompt', C['green']), ('RandOpt searched under the chosen prompt', C['pert'])]
    for k, (title, r, b1, chosen) in enumerate(rows):
        ax = fig.add_subplot(gs[k + 1]); vals = [r['randopt'], b1, r['sc_chosen'], r['randopt_held']]
        for i, (v, (_, col)) in enumerate(zip(vals, bars)):
            ax.bar(i, v, 0.74, color=col, edgecolor='white', lw=0.6)
            ax.text(i, v + 1.2, f'{v:.1f}', ha='center', va='bottom', fontsize=6.6, color=C['ink'])
        ax.axhline(r['base_ro'], color=C['muted'], lw=1.0, ls='--')
        ax.text(3.98, r['base_ro'] + 1.0, f"{r['base_ro']:.1f}", ha='right', va='bottom', fontsize=6.4, color=C['muted'])
        ax.set_xlim(-0.6, 4.0); ax.set_xticks([]); ax.set_ylim(0, 88); ax.grid(axis='x', visible=False)
        ax.set_title(title + NL + f'(chosen prompt: {chosen})', fontsize=7.2)
        if k == 0:
            ax.set_ylabel('test accuracy (%)')
        ax.text(-0.22, 1.13, 'bc'[k], transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    handles = [Patch(color=c, label=lab) for lab, c in bars] + [Line2D([], [], color=C['muted'], ls='--', label="base model, RandOpt's prompt")]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.66, 0.0), ncol=2, frameon=False, fontsize=6.4,
               handlelength=1.4, columnspacing=1.2)
    save(fig, 'fig1_overview')


if __name__ == '__main__':
    main()
