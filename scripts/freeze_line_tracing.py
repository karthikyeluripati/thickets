"""Freeze the deterministic synthetic benchmark before baseline or candidate inference."""
import argparse
import json
from thicket_runtime.line_tracing import freeze_dataset

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True)
args = p.parse_args()
result = freeze_dataset(args.out)
print(json.dumps({k: result[k] for k in ('counts','label_counts','seed_and_image_disjoint','straight_endpoint_heuristic_accuracy')}))
