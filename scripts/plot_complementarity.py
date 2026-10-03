"""Static selection/held-out and deployment-quality figures from frozen results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.cli import revision_info, write_json


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    summary_path = args.analysis / 'summary.json'
    summary = json.loads(summary_path.read_bytes())
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'manifest.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv],
               'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
               'matplotlib': matplotlib.__version__, 'numpy': np.__version__})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained', sharey=True)
    for col, (pop, title) in enumerate(zip(('development','validation_a','validation_b'), ('Development','Validation A','Validation B'))):
        data = summary[pop]
        for row, (field, label) in enumerate((('selection_accuracy','200 selection prompts'),('accuracy','300 fresh held-out prompts'))):
            ax = axes[row, col]
            ks = [3,5,10,20]
            ax.plot([3,5,10,20,50], [data[f'standard-k{k}'][field] for k in [3,5,10,20,50]], 'o-', label='Standard top-K', color='tab:blue')
            ax.plot(ks, [data[f'greedy-0-k{k}'][field] for k in ks], 's-', label='Greedy (primary seed 0)', color='tab:orange')
            for seed in (1,2):
                ax.plot(ks,[data[f'greedy-{seed}-k{k}'][field] for k in ks],color='tab:orange',alpha=.35,linestyle=':')
            controls = np.array([[data[f'random-{seed}-k{k}'][field] for k in ks] for seed in (101,102,103)])
            ax.plot(ks, controls.mean(axis=0), '^-', color='tab:green', label='Exact-quality random (mean)')
            ax.fill_between(ks, controls.min(axis=0), controls.max(axis=0), color='tab:green', alpha=.10)
            ax.set(title=f'{title}: {label}', xlabel='Deployed experts K', ylabel='Voting accuracy', ylim=(0,1), xticks=[3,10,20,50])
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Frozen committees: selection gains versus fresh-test transfer\nDotted lines: fixed tie diagnostics; shading: range of three matched random controls')
    fig.savefig(args.out / 'selection-to-heldout.png', dpi=180)
    fig.savefig(args.out / 'selection-to-heldout.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
