"""Visual Thicket experiment framework (Stage 5 of the WACV extension) -- CPU-only
infrastructure that defines the scientific protocol every future GPU perturbation experiment
follows: model-anatomy discovery, the two perturbation modes, perturbation identity, data-role
partitioning, thicket/diversity metrics, low-rank-geometry interfaces, and an experiment-size
estimator.

See ../../VISUAL_THICKET_EXPERIMENT_SPEC.md for the frozen experimental protocol this package
implements. Deliberately separate from ..benchmarks/ (the Capability Benchmark Gate,
measurement-instrument validation) and from ..scopes.py/..scoped_perturbation.py/
..thicket_metrics.py (the frozen Gate-1/2 GQA-only reproduction path, historical RandOpt
compatibility) -- this package generalizes ideas first proven there (scoped relative-L2
perturbation, expert-density statistics) to an architecture-scale-independent, multi-capability
setting, but does not modify or depend on GPU/ray/vllm at import time.
"""
