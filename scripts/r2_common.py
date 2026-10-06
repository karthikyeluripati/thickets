"""R2 replication helpers (results/paper-analysis/r2/plan_lock.md): Qwen2.5-VL-7B-Instruct, Stage-2 probe and class."""
import re

import causal_diag_9504111 as cd

MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
N_LAYERS = 28
GROUPS = ['embed', 'L00-03', 'L04-07', 'L08-11', 'L12-15', 'L16-19', 'L20-23', 'L24-27', 'final_norm_head', 'vision']
MID = ('L04-07', 'L08-11', 'L12-15', 'L16-19')  # relative depth 0.14-0.71 (Stage 3: layers 6-23 of 36 = 0.17-0.67)


def block_of(name):
    if re.search(r'(^|\.)visual\.', name):
        return 'vision'
    if name.endswith('embed_tokens.weight'):
        return 'embed'
    m = re.search(r'(^|\.)layers\.(\d+)\.', name)
    if m:
        i = int(m.group(2))
        if not 0 <= i < N_LAYERS:
            raise ValueError(name)
        return f'L{4 * (i // 4):02d}-{4 * (i // 4) + 3:02d}'
    if name.endswith('lm_head.weight') or re.search(r'(^|\.)norm\.weight$', name):
        return 'final_norm_head'
    raise ValueError(f'unassigned parameter: {name}')


LETTER_IDS = cd.LETTER_IDS
