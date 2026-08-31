"""Isolated ICLR causal-density pilot -- tests whether counterfactual (shuffled-image /
text-only) controls reveal that standard RandOpt expert selection at 7B overestimates
causally visual expert density. Lives entirely on branch `iclr-causal-density-pilot`; the
32B S1/S2 implementation (commit 9305cc8) is a read-only dependency, never modified here.

See design.py for the single frozen source of truth this whole package (and
reports/iclr_causal_density/preregistration.md) is generated from.
"""
