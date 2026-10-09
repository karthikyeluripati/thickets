"""Resume RV-2a's selection after the pod stop (results/paper-analysis/rv-reviewer-round/amendment_1.md). The population,
prompt, scorer and ranking are unchanged; only the worker split changes. The pulled select_<w>.jsonl (W = 12) are in
<out>. Residue w (mod 12) is split into sub-workers w, w+12, w+24 (W = 36); each sub-worker's done-file is seeded with
the residue's finished k so nothing is re-scored. Prints one 'slot <gpu> <sub-workers...>' line per slot: LPT
assignment of the sub-workers with work left to 2*G slots, balanced by remaining perturbations."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--gpus', type=int, default=4)
    ap.add_argument('--n', type=int, default=5000); a = ap.parse_args()
    done = {}
    for w in range(12):
        f = a.out / f'select_{w}.jsonl'
        lines = [l for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
        done[w] = (lines, {json.loads(l)['k'] for l in lines})
    jobs = []
    for w in range(12):
        lines, ks = done[w]
        for s in (w, w + 12, w + 24):
            left = [k for k in range(s, a.n, 36) if k not in ks]
            if s != w: (a.out / f'select_{s}.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''))
            if left: jobs.append((len(left), s))
    slots = [[0, []] for _ in range(2 * a.gpus)]
    for n, s in sorted(jobs, reverse=True):
        slot = min(slots, key=lambda x: x[0]); slot[0] += n; slot[1].append(s)
    print('remaining', sum(n for n, _ in jobs), 'sub-workers', len(jobs))
    for i, (n, ss) in enumerate(slots):
        print('slot', i % a.gpus, n, *ss)


if __name__ == '__main__':
    main()
