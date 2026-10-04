"""Freeze the deterministic synthetic benchmark before baseline or candidate inference."""
import argparse
import json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True)
p.add_argument('--version', choices=['v1','v2'], default='v1')
args = p.parse_args()
if args.version == 'v2':
    from thicket_runtime.line_tracing_v2 import freeze_dataset
else:
    from thicket_runtime.line_tracing import freeze_dataset
result = freeze_dataset(args.out)
keys = (('counts','label_counts','seed_and_image_disjoint','straight_endpoint_heuristic_accuracy')
        if args.version == 'v1' else ('schema','counts','label_counts','seed_and_image_disjoint'))
print(json.dumps({k: result[k] for k in keys}))
