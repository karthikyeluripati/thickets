"""RV gates for one search row (results/paper-analysis/rv-reviewer-round/plan_lock.md, "Gates"). Runs on the pod after the
fidelity pass (out_fid: base + perturbations k < 24 under RandOpt's prompt), the search-prompt BASE on test (envtest/, o2_eval
format) and the smoke (out: 15 perturbations per worker under the search prompt). Prints FIDELITY (reported), ENV (blocking)
and PROJECTION (blocking) lines; the job reads them from gates.txt."""
import argparse
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--prompt', required=True)
    ap.add_argument('--ro-ref', type=float, required=True)  # PS: greedy BASE selection accuracy under RandOpt's prompt (%)
    ap.add_argument('--sp-ref', type=float, required=True)  # PS: the same under the search prompt (%)
    ap.add_argument('--tb-ref', type=float, required=True)  # Q2/O2: greedy BASE test accuracy under the search prompt (%)
    ap.add_argument('--log', default='none')  # randopt.py log of the same population (fidelity), or none
    ap.add_argument('--spent', type=float, required=True); ap.add_argument('--cap', type=float, required=True)
    ap.add_argument('--workers', type=int, required=True); ap.add_argument('--gpus', type=int, required=True)
    ap.add_argument('--n', type=int, default=5000); ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt'))
    ap.add_argument('--test-pass-sec', type=float, default=60.0)  # one greedy test pass (1319 rows) per selected model
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); gt = [t['ground_truth'] for t in h.load_data(str(a.upstream / 'data/gsm8k/test.parquet'), split='test')]

    def rows(pattern):
        out = {}
        for f in glob.glob(pattern):
            for l in open(f):
                if l.strip():
                    d = json.loads(l); out[d['k']] = d
        return out
    fid = rows(str(a.root / 'out_fid' / 'select_*.jsonl'))
    if a.log != 'none':
        ref = []
        for l in open(a.log, encoding='utf-8', errors='ignore'):
            m = re.search(r'Batch \d+ \| (\d+)/\d+ \| \[(.*)\]', l)
            if m: ref += [float(x.strip("' ")) for x in m.group(2).split(',')]
            if len(ref) >= 24: break
        d = [fid[k]['reward'] - ref[k] for k in range(24) if k in fid]
        mad = float(np.mean(np.abs(d))); print(f'FIDELITY n={len(d)} mean_abs_diff={mad:.4f} signed={np.mean(d):+.4f} ' + ('CONSISTENT' if mad <= 0.03 else 'INCONSISTENT'))
    else:
        print('FIDELITY n=0 (no randopt.py log for this population seed)')
    ro = 100 * json.loads((a.root / 'out_fid' / 'base_select.json').read_text())['reward']
    sp = 100 * json.loads((a.root / 'out' / 'base_select.json').read_text())['reward']
    tb_rec = json.loads((a.root / 'envtest' / f'{a.prompt}_base.json').read_text())
    tb = 100 * float(np.mean([bool(x['a']) and bool(h.is_answer_correct(h.format_answer_for_check(x['a']), gt[i])) for i, x in enumerate(tb_rec)]))
    ok = abs(ro - a.ro_ref) <= 1.0 and abs(sp - a.sp_ref) <= 1.0 and abs(tb - a.tb_ref) <= 1.0
    print(f'ENV randopt_base_sel={ro:.2f} (ref {a.ro_ref}) {a.prompt}_base_sel={sp:.2f} (ref {a.sp_ref}) {a.prompt}_base_test={tb:.2f} (ref {a.tb_ref}) ' + ('PASS' if ok else 'FAIL'))
    smoke = rows(str(a.root / 'out' / 'select_*.jsonl'))
    s = float(np.mean([d['sec'] for d in smoke.values()]))
    rate = float(dict(x.split('=') for x in open('/workspace/rate.txt').read().split())['RATE'])
    remaining = (a.n - len(smoke)) / a.workers * s * 1.10 / 3600 * rate
    test = 51 / a.gpus * a.test_pass_sec / 3600 * rate
    proj = a.spent + remaining + test + 2.0
    print(f'PROJECTION W={a.workers} sec_per_pert={s:.2f} smoke_n={len(smoke)} spent={a.spent:.2f} proj={proj:.2f} cap={a.cap} ' + ('GO' if proj <= a.cap else 'NO_GO'))


if __name__ == '__main__':
    main()
