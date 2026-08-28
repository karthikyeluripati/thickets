"""Stage-11 32B READINESS + exact-semantics memory architecture -- infrastructure-only milestone
(task spec: "DO NOT RUN 32B FULL. DO NOT RUN 32B ANATOMY. DO NOT ENABLE 72B."). Nothing here runs
a model, touches a GPU, or changes RUNNABLE_SCALES in scaling_common.py -- 32B and 72B remain
exactly as hard-disabled as before this module existed (`ensure_scale_runnable` is untouched).

WHAT THIS MODULE IS: a G1-G8 readiness-GATE framework (task spec Section 14) plus the concrete
inputs those gates need -- a parameter-count/VRAM estimator built from Qwen2.5-VL-32B-Instruct's
REAL, live-fetched HuggingFace config.json (text_config fields; see QWEN25_VL_32B_TEXT_CONFIG's
own `source` field) combined with PUBLICLY DOCUMENTED (not independently re-verified this
session) Qwen2.5-VL vision-tower architecture for the vision-config fields the live fetch did not
return (depth, attention heads) -- flagged explicitly, per field, as CONFIRMED_LIVE vs
ESTIMATED_FROM_PUBLISHED_ARCHITECTURE. The AUTHORITATIVE parameter count for any real go/no-go
decision is `report_scaling_anatomy_audit`'s own live `total_model_elements` (the SAME RPC
already trusted for 3B/7B) -- this module's estimate exists to size the GPU requirement BEFORE
ever touching a real 32B checkpoint, not to replace that live audit.

Every gate function below requires EXPLICIT, INJECTED evidence (a live config dict, a live
VRAM estimate, a live equivalence-test result, ...) -- none of them silently return PASS for
missing/unresolved evidence; an ungiven gate reports NOT_YET_VERIFIED, never PASS. No gate in
this module has, as of this commit, been evaluated against real hardware or a real 32B
checkpoint -- see build_32b_readiness_manifest()'s own docstring for the honest current status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_NOT_YET_VERIFIED = "NOT_YET_VERIFIED"
GATE_VERDICTS = (GATE_PASS, GATE_FAIL, GATE_NOT_YET_VERIFIED)

GATE_IDS: Sequence[str] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")

# =================================================================================================
# Section 1/2: model spec + architecture audit
# =================================================================================================

FROZEN_32B_MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"  # from scaling_common.SCALING_MODEL_REGISTRY["32B"] -- confirmed already registered, revision_ref="main" (not yet pinned)
FROZEN_32B_MODEL_FAMILY = "qwen2_5_vl"

# CONFIRMED_LIVE: fetched from https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/raw/main/config.json this session.
QWEN25_VL_32B_TEXT_CONFIG: Dict[str, Any] = {
    "source": "CONFIRMED_LIVE",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 27648,
    "vocab_size": 152064,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
}

# vision_config hidden_size/out_hidden_size/intermediate_size/patch_size/in_channels are
# CONFIRMED_LIVE from the same fetch. depth (block count) and num_heads were NOT returned by
# that fetch and are ESTIMATED_FROM_PUBLISHED_ARCHITECTURE (the Qwen2.5-VL technical report
# documents a single ViT tower, depth=32, reused unchanged across the 3B/7B/32B/72B family --
# consistent with this project's OWN empirical finding this session that vision+connector
# region_mask_hashes were byte-identical between the real 3B and 7B runs). NOT independently
# re-confirmed against a second live source this session -- flag accordingly in any downstream use.
QWEN25_VL_32B_VISION_CONFIG: Dict[str, Any] = {
    "source_hidden_and_patch_fields": "CONFIRMED_LIVE",
    "source_depth_and_heads": "ESTIMATED_FROM_PUBLISHED_ARCHITECTURE_NOT_LIVE_CONFIRMED",
    "hidden_size": 1280,
    "out_hidden_size": 5120,
    "intermediate_size": 3456,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "depth": 32,
    "num_heads": 16,
}


def estimate_qwen25_vl_32b_parameter_count(
    text_config: Dict[str, Any] = QWEN25_VL_32B_TEXT_CONFIG, vision_config: Dict[str, Any] = QWEN25_VL_32B_VISION_CONFIG,
    *, tie_word_embeddings: bool = False,
) -> Dict[str, Any]:
    """Standard dense-transformer parameter-counting formula (Qwen2/Llama-style: GQA attention +
    SwiGLU MLP + RMSNorm), applied to the CONFIRMED_LIVE text_config, plus a vision-tower
    estimate from the partially-confirmed vision_config. `tie_word_embeddings` defaults to False
    (untied lm_head) as the CONSERVATIVE (larger) VRAM assumption -- Qwen2.5's actual tie setting
    for the 32B variant was not independently confirmed this session; report this explicitly
    rather than guess a specific answer that could understate VRAM need.
    """
    H, L, I, V = text_config["hidden_size"], text_config["num_hidden_layers"], text_config["intermediate_size"], text_config["vocab_size"]
    heads, kv_heads = text_config["num_attention_heads"], text_config["num_key_value_heads"]
    head_dim = H // heads

    attn_per_layer = H * (heads * head_dim) + H * (kv_heads * head_dim) + H * (kv_heads * head_dim) + (heads * head_dim) * H
    mlp_per_layer = H * I + H * I + I * H
    norms_per_layer = 2 * H
    per_layer = attn_per_layer + mlp_per_layer + norms_per_layer
    transformer_total = per_layer * L

    embed = V * H
    lm_head = 0 if tie_word_embeddings else V * H
    final_norm = H
    language_total = transformer_total + embed + lm_head + final_norm

    vh, vi, vd, vheads = vision_config["hidden_size"], vision_config["intermediate_size"], vision_config["depth"], vision_config["num_heads"]
    vhead_dim = vh // vheads
    vision_attn_per_layer = vh * (vheads * vhead_dim) * 3 + (vheads * vhead_dim) * vh  # qkv + o_proj, no GQA in the ViT
    vision_mlp_per_layer = vh * vi + vi * vh
    vision_per_layer = vision_attn_per_layer + vision_mlp_per_layer + 2 * vh
    vision_transformer_total = vision_per_layer * vd
    patch_embed = vision_config["patch_size"] ** 2 * vision_config["temporal_patch_size"] * vision_config["in_channels"] * vh
    out_h = vision_config["out_hidden_size"]
    merger = (vh * 4) * out_h + out_h * out_h  # 2x2 spatial-merge MLP into the LLM embedding space -- estimated structure, not live-confirmed
    vision_total = vision_transformer_total + patch_embed + merger

    total = language_total + vision_total
    return {
        "language_backbone_params": language_total, "vision_tower_params": vision_total, "total_params": total,
        "tie_word_embeddings_assumed": tie_word_embeddings,
        "confidence": "language backbone CONFIRMED_LIVE architecture fields; vision tower PARTIALLY estimated (see QWEN25_VL_32B_VISION_CONFIG). "
                       "AUTHORITATIVE count is report_scaling_anatomy_audit's live total_model_elements, not this estimate.",
    }


# =================================================================================================
# Section 3: hardware feasibility
# =================================================================================================

L40S_TOTAL_VRAM_GIB = 48.0
L40S_USABLE_VRAM_GIB = 44.39  # as stated by the user this session -- attributed, not independently re-derived
BF16_BYTES_PER_PARAM = 2

BASE_SNAPSHOT_OVERHEAD_MULTIPLIER = {"store_base_weights": 2.0, "cpu_base_weights": 1.0}  # legacy GPU-resident clone doubles GPU weight storage; CPU snapshot does not add ANY GPU-resident copy


def estimate_vram_gib(
    total_params: int, *, tensor_parallel_size: int, base_snapshot_mode: str,
    kv_cache_gib: float = 4.0, runtime_overhead_gib: float = 3.0, largest_tensor_transient_gib: float = 1.5,
) -> Dict[str, Any]:
    """Per-GPU VRAM estimate for one tensor-parallel rank. `kv_cache_gib`/`runtime_overhead_gib`/
    `largest_tensor_transient_gib` are DELIBERATELY CONSERVATIVE fixed placeholders (vLLM CUDA
    graphs, NCCL buffers, activation workspace, the single-largest-parameter transient this
    project's own memory-bounded ops already document at 7B) -- they are NOT derived from a live
    32B profiling run (none exists yet); a real smoke attempt should replace them with observed
    values, never trust these blindly as a hard ceiling.
    """
    if base_snapshot_mode not in BASE_SNAPSHOT_OVERHEAD_MULTIPLIER:
        raise ValueError(f"Unknown base_snapshot_mode {base_snapshot_mode!r}")
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")

    weight_bytes_total = total_params * BF16_BYTES_PER_PARAM
    weight_gib_per_gpu = (weight_bytes_total / tensor_parallel_size) / (1024 ** 3)
    base_snapshot_gib_per_gpu = weight_gib_per_gpu * (BASE_SNAPSHOT_OVERHEAD_MULTIPLIER[base_snapshot_mode] - 1.0)

    total_resident_gib = weight_gib_per_gpu + base_snapshot_gib_per_gpu + kv_cache_gib + runtime_overhead_gib + largest_tensor_transient_gib
    headroom_gib = L40S_USABLE_VRAM_GIB - total_resident_gib

    return {
        "tensor_parallel_size": tensor_parallel_size, "base_snapshot_mode": base_snapshot_mode,
        "weight_gib_per_gpu": weight_gib_per_gpu, "base_snapshot_overhead_gib_per_gpu": base_snapshot_gib_per_gpu,
        "kv_cache_gib": kv_cache_gib, "runtime_overhead_gib": runtime_overhead_gib, "largest_tensor_transient_gib": largest_tensor_transient_gib,
        "total_resident_gib_per_gpu": total_resident_gib, "l40s_usable_vram_gib": L40S_USABLE_VRAM_GIB,
        "headroom_gib": headroom_gib, "fits": headroom_gib > 0,
    }


def recommend_min_gpu_count(total_params: int, *, base_snapshot_mode: str, candidate_tp_sizes: Sequence[int] = (1, 2, 4, 8), min_safety_headroom_gib: float = 8.0) -> Dict[str, Any]:
    """The smallest candidate TP size whose estimated headroom exceeds `min_safety_headroom_gib`
    -- a SAFETY MARGIN, not merely "fits with zero room to spare" (KV cache/batch size need real
    slack). Returns the full per-TP-size estimate table so the caller sees the tradeoff, never
    just a bare number with no justification.
    """
    param_estimate = estimate_qwen25_vl_32b_parameter_count()
    table = {tp: estimate_vram_gib(param_estimate["total_params"], tensor_parallel_size=tp, base_snapshot_mode=base_snapshot_mode) for tp in candidate_tp_sizes}
    safe = [tp for tp, est in table.items() if est["headroom_gib"] >= min_safety_headroom_gib]
    recommended = min(safe) if safe else None
    return {
        "total_params_estimate": param_estimate["total_params"], "base_snapshot_mode": base_snapshot_mode,
        "min_safety_headroom_gib": min_safety_headroom_gib, "estimates_by_tp_size": table,
        "recommended_min_tp_size": recommended,
        "note": "recommended_min_tp_size is None if no candidate TP size clears the safety margin -- try a larger candidate_tp_sizes entry." if recommended is None else None,
    }


# =================================================================================================
# Section 14: G1-G8 readiness gates -- every function requires explicit evidence, never assumes
# =================================================================================================


def g1_model_family_audit(spec_model_name: str, spec_model_family: str, live_config_model_type: Optional[str] = None) -> str:
    if spec_model_name != FROZEN_32B_MODEL_NAME or spec_model_family != FROZEN_32B_MODEL_FAMILY:
        return GATE_FAIL
    if live_config_model_type is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if live_config_model_type == "qwen2_5_vl" else GATE_FAIL


def g2_hardware_feasibility(vram_estimate: Optional[Dict[str, Any]], *, min_safety_headroom_gib: float = 8.0) -> str:
    if vram_estimate is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if vram_estimate.get("headroom_gib", -1) >= min_safety_headroom_gib else GATE_FAIL


def g3_cpu_snapshot_bit_equivalence(equivalence_class: Optional[str]) -> str:
    from .thicket.cpu_base_snapshot import EQUIVALENCE_BIT_EXACT

    if equivalence_class is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if equivalence_class == EQUIVALENCE_BIT_EXACT else GATE_FAIL


def g4_distributed_relative_l2_semantics(live_test_passed: Optional[bool]) -> str:
    """Requires a LIVE (multi-rank or genuinely-simulated-multi-shard) confirmation that the
    global relative-L2 ratio holds -- this project's own CPU test suite proves the MATH is
    correct against fake shards/reduces (see tests/test_thicket_distributed_perturbation.py);
    it does NOT constitute live-hardware verification, so this gate defaults to
    NOT_YET_VERIFIED until a caller explicitly supplies a live_test_passed result.
    """
    if live_test_passed is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if live_test_passed else GATE_FAIL


def g5_distributed_rng_semantics(live_test_passed: Optional[bool]) -> str:
    if live_test_passed is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if live_test_passed else GATE_FAIL


def g6_exact_restoration(aggregated_verification: Optional[Dict[str, Any]]) -> str:
    if aggregated_verification is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if aggregated_verification.get("ok") is True else GATE_FAIL


def g7_subset_gate(subset_gate_report: Optional[Dict[str, Any]]) -> str:
    if subset_gate_report is None:
        return GATE_NOT_YET_VERIFIED
    mode = subset_gate_report.get("subset_gate_mode")
    if mode == "smoke_n5_deterministic_repeatability":
        det = subset_gate_report.get("smoke_determinism", {})
        return GATE_PASS if det.get("all_deterministic") and det.get("all_n_matches_expected") else GATE_FAIL
    if mode == "stage8_full_n50_exact_equality":
        eq = subset_gate_report.get("subset_hash_equality", {})
        return GATE_PASS if eq.get("all_match") else GATE_FAIL
    return GATE_NOT_YET_VERIFIED


def g8_tests(pytest_exit_code: Optional[int]) -> str:
    if pytest_exit_code is None:
        return GATE_NOT_YET_VERIFIED
    return GATE_PASS if pytest_exit_code == 0 else GATE_FAIL


_GATE_FUNCTIONS = {"G1": g1_model_family_audit, "G2": g2_hardware_feasibility, "G3": g3_cpu_snapshot_bit_equivalence,
                    "G4": g4_distributed_relative_l2_semantics, "G5": g5_distributed_rng_semantics, "G6": g6_exact_restoration,
                    "G7": g7_subset_gate, "G8": g8_tests}


class Stage32BSmokeNotPermittedError(RuntimeError):
    """Raised by ensure_32b_smoke_permitted -- one or more G1-G8 gates is not GATE_PASS."""


def ensure_32b_smoke_permitted(gate_results: Dict[str, str]) -> None:
    """Task spec Section 14: 'A 32B WHOLE-MODEL SMOKE may only be attempted after G1-G8 PASS. If
    any gate fails: STOP. Do not compensate by changing scientific protocol.' This function is
    the SINGLE enforcement point -- it does not, and structurally cannot, flip RUNNABLE_SCALES in
    scaling_common.py (that gate is untouched and remains 3B/7B-only regardless of this
    function's outcome); it exists for a FUTURE runner-integration step to consult once every
    gate has genuinely been evaluated against live evidence, never as a bypass mechanism today.
    """
    missing = [g for g in GATE_IDS if g not in gate_results]
    if missing:
        raise Stage32BSmokeNotPermittedError(f"Missing gate result(s) for {missing} -- cannot evaluate 32B smoke readiness with an incomplete gate set.")
    not_passing = {g: gate_results[g] for g in GATE_IDS if gate_results[g] != GATE_PASS}
    if not_passing:
        raise Stage32BSmokeNotPermittedError(f"32B smoke NOT permitted -- gate(s) not PASS: {not_passing}")


def ensure_32b_scale_runnable_for_smoke(*, is_smoke: bool, is_anatomy_track: bool, gate_results: Dict[str, str]) -> None:
    """The SINGLE 32B entry-point check a runner's main() calls -- 32B is runnable ONLY when ALL
    of: (a) --smoke was explicitly requested (a full 32B run is never permitted through this
    path, structurally, regardless of gate status), (b) the track is whole_model, never anatomy
    ("DO NOT RUN 32B ANATOMY" -- task spec), (c) every G1-G8 gate is GATE_PASS. Raising here
    means main() must abort BEFORE launching an engine or evaluating any candidate -- 'refuse to
    start perturbations' (task spec Section 14).
    """
    if not is_smoke:
        raise Stage32BSmokeNotPermittedError("32B is runnable ONLY via --smoke -- a full 32B run is not permitted through this path regardless of gate status.")
    if is_anatomy_track:
        raise Stage32BSmokeNotPermittedError("32B anatomy is not permitted -- only track=whole_model smoke may be attempted (task spec: 'DO NOT RUN 32B ANATOMY').")
    ensure_32b_smoke_permitted(gate_results)


# =================================================================================================
# Section 13: pre-flight readiness manifest
# =================================================================================================


@dataclass(frozen=True)
class Stage32BReadinessManifest:
    scale_label: str
    model_name: str
    requested_revision_ref: str
    resolved_revision: Optional[str]
    total_parameters_estimate: int
    region_definitions: Sequence[str]
    intended_tp_size: Optional[int]
    dtype: str
    base_snapshot_mode: str
    perturbation_semantics: str
    radii: Sequence[float]
    capabilities: Sequence[str]
    smoke_counts: Dict[str, int]
    full_counts: Dict[str, int]
    subset_gate_policy: str
    cache_policy: str
    prefix_caching_policy: bool
    restoration_policy: str
    gate_results: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale_label": self.scale_label, "model_name": self.model_name, "requested_revision_ref": self.requested_revision_ref,
            "resolved_revision": self.resolved_revision, "total_parameters_estimate": self.total_parameters_estimate,
            "region_definitions": list(self.region_definitions), "intended_tp_size": self.intended_tp_size, "dtype": self.dtype,
            "base_snapshot_mode": self.base_snapshot_mode, "perturbation_semantics": self.perturbation_semantics,
            "radii": list(self.radii), "capabilities": list(self.capabilities), "smoke_counts": dict(self.smoke_counts),
            "full_counts": dict(self.full_counts), "subset_gate_policy": self.subset_gate_policy, "cache_policy": self.cache_policy,
            "prefix_caching_policy": self.prefix_caching_policy, "restoration_policy": self.restoration_policy,
            "gate_results": dict(self.gate_results), "all_gates_pass": all(v == GATE_PASS for v in self.gate_results.values()) if self.gate_results else False,
        }


def build_32b_readiness_manifest(
    *, resolved_revision: Optional[str] = None, intended_tp_size: Optional[int] = None,
    gate_results: Optional[Dict[str, str]] = None,
) -> Stage32BReadinessManifest:
    """HONEST CURRENT STATUS (read before trusting `all_gates_pass`): calling this with no
    `gate_results` produces a manifest where every gate reads NOT_YET_VERIFIED -- this commit
    provides the gate FRAMEWORK and the estimator inputs, not a live-verified go decision. No
    real 32B checkpoint, GPU, or distributed runtime has been touched this session.
    """
    from .run_stage8_coarse_anatomical_atlas import STAGE8_CAPABILITIES, STAGE8_RADII, STAGE8_REGIONS

    param_estimate = estimate_qwen25_vl_32b_parameter_count()
    resolved_gates = {g: GATE_NOT_YET_VERIFIED for g in GATE_IDS}
    if gate_results:
        resolved_gates.update(gate_results)

    return Stage32BReadinessManifest(
        scale_label="32B", model_name=FROZEN_32B_MODEL_NAME, requested_revision_ref="main", resolved_revision=resolved_revision,
        total_parameters_estimate=param_estimate["total_params"], region_definitions=STAGE8_REGIONS, intended_tp_size=intended_tp_size,
        dtype="bfloat16", base_snapshot_mode="cpu_base_weights", perturbation_semantics="anatomical_relative_l2",
        radii=STAGE8_RADII, capabilities=STAGE8_CAPABILITIES,
        smoke_counts={"s1_perturbations": 3, "s1_rows": 18, "s2_perturbations": 9, "s2_rows": 54},
        full_counts={"s1_perturbations": 192, "s1_rows": 1152, "s2_perturbations": 576, "s2_rows": 3456},
        subset_gate_policy="mode_aware_shared_gate", cache_policy="full_encoder_reset_vllm011_verified_v2",
        prefix_caching_policy=False, restoration_policy="fixed_base_exact", gate_results=resolved_gates,
    )


# =================================================================================================
# 32B engine configuration (task spec Section 3) -- a SEPARATE builder, never a modification of
# run_global_visual_thicket_pilot.build_stage6_engine_config (which 3B/7B still use, unchanged,
# hardcoded tensor_parallel_size=1 and base_snapshot_mode=BASE_SNAPSHOT_MODE="store_base_weights").
# =================================================================================================


def build_32b_engine_config(*, tensor_parallel_size: int = 4, gpu_memory_utilization: float = 0.60) -> Dict[str, Any]:
    """Task spec Section 3, applied literally: BF16 (never quantized), TP=4 default,
    gpu_memory_utilization=0.60 initially, max_model_len=4096, enforce_eager=True (skips
    CUDA-graph capture -- safer first attempt at an unfamiliar memory footprint; graphs can be
    re-enabled once a real smoke has actually succeeded once), prefix caching False,
    base_snapshot_mode=cpu_base_weights (this milestone's fix -- never the legacy GPU-doubling
    mode for 32B).
    """
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
    return {
        "max_model_len": 4096, "gpu_memory_utilization": gpu_memory_utilization, "tensor_parallel_size": tensor_parallel_size,
        "precision": "bfloat16", "enforce_eager": True, "enable_prefix_caching": False,
        "restoration_mode": "fixed_base", "perturbation_semantics": "anatomical_relative_l2", "base_snapshot_mode": "cpu_base_weights",
    }


# =================================================================================================
# THE ONE STRUCTURAL BLOCKER (task spec Section 7's own contingency: "If live TP parameter layout
# makes the current full-shape-noise strategy incorrect or infeasible: STOP AND REPORT. Do NOT
# silently invent a different Gaussian protocol.") -- found by reading the real candidate-lifecycle
# code, not assumed. The ACTUAL per-candidate perturbation call
# (evaluate_one_whole_model_candidate_rpc, run_stage11_whole_model_scaling.py) dispatches
# scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 -- the ITERATIVE bf16-aware
# bracketed radius solver every real 3B/7B run actually used (never the simpler one-shot
# apply_anatomical_relative_l2 thicket.distributed_perturbation.apply_anatomical_relative_l2_
# distributed extends). That solver's bisection loop was built empirically, against REAL bf16
# rounding behavior observed on real GPU hardware (see scoped_anatomical_perturbation.py's own
# "BF16 BRACKETED SOLVER v2" docstring) -- extending IT to a distributed, collective-reduced,
# multi-rank version cannot be responsibly designed blind, without live GPU/TP hardware to verify
# convergence against. Silently swapping 32B onto the simpler one-shot distributed primitive
# instead would use a DIFFERENT radius-realization method than 3B/7B -- breaking cross-scale
# comparability, the exact "silently invent a different protocol" outcome forbidden above.
# =================================================================================================

V3_SOLVER_DISTRIBUTED_EXTENSION_AVAILABLE = False
V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE = (
    "scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3 (the iterative bf16-bracketed "
    "radius solver the real whole-model candidate lifecycle actually calls) has NOT been extended "
    "to a distributed, collective-reduced, multi-TP-rank version. thicket.distributed_perturbation "
    "provides the one-shot apply_anatomical_relative_l2_distributed primitive and proves the "
    "global-norm/RNG/restoration DESIGN is correct, but the v3 solver's bisection loop -- built "
    "empirically against real bf16 rounding behavior on real GPU hardware -- has not itself been "
    "made TP-aware, and doing so blind (without live GPU/TP hardware to verify convergence) risks "
    "either incorrect radius realization or a silent protocol change relative to 3B/7B. STOP AND "
    "REPORT, per task spec Section 7, rather than substitute a different realization method for 32B."
)


def check_v3_solver_distributed_readiness() -> str:
    return GATE_PASS if V3_SOLVER_DISTRIBUTED_EXTENSION_AVAILABLE else GATE_FAIL


def run_32b_readiness_preflight_and_report(*, resolved_revision: str, tensor_parallel_size: int, output_dir: Any) -> Dict[str, Any]:
    """The function run_stage11_whole_model_scaling.main() calls for scale=='32B', BEFORE any
    engine launch -- evaluates every gate that can be determined WITHOUT live GPU evidence
    (today: only G1 partially, via the resolved revision + frozen spec; everything else stays
    NOT_YET_VERIFIED or is forced FAIL by the v3-solver gap above), persists the manifest, and
    returns it. Deliberately does NOT attempt to launch a TP=4 engine just to immediately abort --
    the v3-solver gap is knowable with zero GPU time spent.
    """
    import json
    from pathlib import Path

    gate_results = {g: GATE_NOT_YET_VERIFIED for g in GATE_IDS}
    gate_results["G1"] = g1_model_family_audit(FROZEN_32B_MODEL_NAME, FROZEN_32B_MODEL_FAMILY)  # NOT_YET_VERIFIED without a live config fetch on the pod
    gate_results["G4"] = check_v3_solver_distributed_readiness()  # structurally FAIL today -- see V3_SOLVER_DISTRIBUTED_EXTENSION_NOTE

    manifest = build_32b_readiness_manifest(resolved_revision=resolved_revision, intended_tp_size=tensor_parallel_size, gate_results=gate_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage11_32b_readiness_gate_report.json"
    report_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    return {"manifest": manifest, "report_path": report_path, "blocked_by_v3_solver_gap": not V3_SOLVER_DISTRIBUTED_EXTENSION_AVAILABLE}
