# Source provenance and corrections to the conceptual picture

Reviewed source: sunrainyg/RandOpt, commit
`4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca`.

- `randopt.py::run_sampling`: seed/sigma selection, then batched perturb RPC,
  generation, restore RPC and dataset-handler scoring. Selection is outside the
  worker. Runtime profiling must not implicitly change the candidate policy.
- `utils/worker_extn.py::perturb_self_weights` and `restore_self_weights`:
  one native-dtype noise tensor at a time, generator reset per parameter, separate
  scale and in-place addition, then synchronization/cache cleanup.
- `core/engine.py::launch_engines`: creates Ray/vLLM engines and stores base weights.
- Worker `store_base_weights`, `apply_perturbation`, `reset_to_base_weights` already
  provide a snapshot-based alternative. Snapshot copy is prior implementation,
  not our contribution.

The reference executor captures these state-operation ideas but differs in
orchestration, tokenization, model layout, buffer handling and inference engine.
It is not a reproduction of the paper's accuracy or throughput.
The original-worker adapter validates the Git blob before loading local source.
Its optional dependency path has not been exercised in the initial CPU session.

Primary references:

- https://github.com/sunrainyg/RandOpt/blob/4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca/utils/worker_extn.py
- https://github.com/sunrainyg/RandOpt/blob/4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca/randopt.py
- https://github.com/sunrainyg/RandOpt/blob/4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca/core/engine.py
- https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
- https://docs.pytorch.org/docs/stable/notes/randomness.html
- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

These establish implementation/numerical background, not an exhaustive novelty
search. Recheck the systems literature after selecting a measured bottleneck.
