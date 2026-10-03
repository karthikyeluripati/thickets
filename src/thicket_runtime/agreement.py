"""Exact trace agreement and an initial-prefix-only speculative round model.

This module never infers a draft at a corrected candidate prefix from an aligned
suffix of an independently generated base trace. Rounds are an optimistic cost
model, not latency measurements; even one block costs more than one decode token.
"""
from collections import Counter
import math
from statistics import mean

PREFIX_SIZES = (1, 2, 4, 8, 16, 32)
BLOCK_SIZES = (2, 4, 8, 16)


def common_prefix(base, candidate):
    for i, (a, b) in enumerate(zip(base, candidate)):
        if a != b:
            return i
    return min(len(base), len(candidate))


def compare_tokens(base, candidate):
    p = common_prefix(base, candidate)
    equal = list(base) == list(candidate)
    return {
        "common_prefix_tokens": p, "base_tokens": len(base),
        "candidate_tokens": len(candidate),
        "common_prefix_fraction": p / len(candidate) if candidate else None,
        "first_divergence_position": None if equal else p,
        "divergence_kind": ("none_observed" if equal else "token_mismatch"
                            if p < min(len(base), len(candidate)) else "length_mismatch"),
        "complete_observed_token_equality": equal,
        "immediate_divergence": p == 0 and not equal,
        "first_k_match": {str(k): p >= k for k in PREFIX_SIZES},
        "first_k_available": {str(k): min(len(base), len(candidate)) >= k
                              for k in PREFIX_SIZES},
    }


def simulate_prefix(base, candidate, k):
    """One ideal verification per block; corrected token on rejection, no bonus.

    Share only the initial base trajectory. On rejection, emit the correct
    candidate argmax and use ordinary decoding for ALL remaining tokens. A short
    final block is allowed. No free bonus token, no draft regeneration/rejoining,
    no cost assigned to base drafting, prefill, KV maintenance, or weight changes.
    Both reference and speculative round counts exclude prompt prefill and count
    the first output as a round; this is a sequential output-round abstraction.
    """
    if type(k) is not int or k < 1:
        raise ValueError("block size must be a positive integer")
    n, m = len(candidate), len(base)
    prefix = common_prefix(base, candidate)
    pos = calls = accepted = wasted = verified = full_attempts = full_accepts = 0
    rejected = False
    while pos < min(n, m):
        width = min(k, m - pos, n - pos)
        calls += 1
        verified += width
        full_attempts += width == k
        shared = min(width, max(0, prefix - pos))
        accepted += shared
        if shared < width:
            wasted += width - shared - 1  # positions strictly AFTER rejection
            pos += shared + 1             # candidate's corrected argmax
            rejected = True
            break
        full_accepts += width == k
        pos += width
    fallback = n - pos
    rounds = calls + fallback
    return {"block_size": k, "accepted_draft_tokens": accepted,
            "verification_calls": calls, "verified_positions": verified,
            "full_blocks_attempted": full_attempts, "full_blocks_accepted": full_accepts,
            "rejected": rejected, "wasted_positions_after_rejection": wasted,
            "fallback_token_rounds": fallback, "ordinary_output_rounds": n,
            "ideal_candidate_output_rounds": rounds,
            "rounds_saved": n - rounds,
            "round_reduction": (n - rounds) / n if n else None,
            "all_accepted_oracle_rounds": math.ceil(n / k),
            "all_accepted_oracle_reduction": 1 - math.ceil(n / k) / n if n else None}


def distribution(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    def quantile(q):
        pos = (len(v) - 1) * q
        lo = int(pos)
        return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)
    return {"n": len(v), "mean": mean(v), "min": v[0], "max": v[-1],
            **{name: quantile(q) for name, q in
               (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75),
                ("p90", .9), ("p95", .95))}}


def rank_correlation(x, y):
    """Descriptive Spearman correlation with average ranks for ties."""
    if len(x) != len(y):
        raise ValueError("rank vectors must have equal length")
    def ranks(v):
        ordered = sorted(range(len(v)), key=v.__getitem__)
        out, start = [0.] * len(v), 0
        while start < len(v):
            end = start + 1
            while end < len(v) and v[ordered[end]] == v[ordered[start]]:
                end += 1
            for i in ordered[start:end]:
                out[i] = (start + end - 1) / 2
            start = end
        return out
    if not x:
        return None
    a, b = ranks(x), ranks(y)
    ma, mb = mean(a), mean(b)
    numerator = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    denom = math.sqrt(sum((u - ma)**2 for u in a) * sum((v - mb)**2 for v in b))
    return numerator / denom if denom else None


def summarize_pairs(pairs):
    if not pairs:
        return {"pairs": 0}
    n = len(pairs)
    result = {"pairs": n, **{key: distribution(r[key] for r in pairs) for key in
              ("common_prefix_tokens", "common_prefix_fraction", "candidate_tokens",
               "first_divergence_position", "candidate_reward", "base_reward")},
              "common_prefix_histogram": dict(sorted(Counter(str(r["common_prefix_tokens"])
                                                              for r in pairs).items())),
              "observed_output_equality_fraction": mean(r["complete_observed_token_equality"] for r in pairs),
              "uncensored_complete_equality_fraction": mean(r["uncensored_complete_equality"] for r in pairs),
              "candidate_cap_fraction": mean(r["candidate_capped"] for r in pairs),
              "base_cap_fraction": mean(r["base_capped"] for r in pairs),
              "immediate_divergence_fraction": mean(r["immediate_divergence"] for r in pairs),
              "first_k": {}, "simulation": {}}
    for k in PREFIX_SIZES:
        key = str(k)
        eligible = sum(r["first_k_available"][key] for r in pairs)
        matched = sum(r["first_k_match"][key] for r in pairs)
        result["first_k"][key] = {"matched": matched, "eligible": eligible,
                                 "fraction_all_pairs": matched / n,
                                 "fraction_eligible": matched / eligible if eligible else None}
    for k in BLOCK_SIZES:
        rows = [r["simulation"][str(k)] for r in pairs]
        total = sum(r["ordinary_output_rounds"] for r in rows)
        attempts = sum(r["full_blocks_attempted"] for r in rows)
        result["simulation"][str(k)] = {
            "token_weighted_round_reduction": sum(r["rounds_saved"] for r in rows) / total if total else None,
            "full_blocks_attempted": attempts,
            "full_blocks_accepted": sum(r["full_blocks_accepted"] for r in rows),
            "full_block_acceptance": sum(r["full_blocks_accepted"] for r in rows) / attempts if attempts else None,
            **{key: distribution(r[key] for r in rows) for key in
               ("round_reduction", "accepted_draft_tokens", "wasted_positions_after_rejection",
                "ideal_candidate_output_rounds", "ordinary_output_rounds", "all_accepted_oracle_reduction")}}
    return result


def summarize_records(records):
    """Group by workload/scale; pairs are clustered by both prompt and seed."""
    groups = {}
    for record in records:
        for workload, rows in record["workloads"].items():
            groups.setdefault((workload, record["candidate"]["sigma"]), []).append((record, rows))
    output = {}
    for (workload, sigma), members in sorted(groups.items()):
        pairs = [r for _, rows in members for r in rows]
        candidates = [{"candidate_id": r["candidate"]["candidate_id"],
                       "seed": r["candidate"]["seed"],
                       "reward": mean(x["candidate_reward"] for x in rows),
                       "base_reward": mean(x["base_reward"] for x in rows),
                       "mean_prefix": mean(x["common_prefix_tokens"] for x in rows),
                       "mean_prefix_fraction": mean(x["common_prefix_fraction"] or 0 for x in rows)}
                      for r, rows in members]
        cutoff = distribution(c["reward"] for c in candidates)["p75"]
        top = {c["candidate_id"] for c in candidates if c["reward"] >= cutoff}
        better = {c["candidate_id"] for c in candidates if c["reward"] > c["base_reward"]}
        stats = summarize_pairs(pairs)
        stats.update(candidates=len(members), per_candidate=candidates,
                     reward_prefix_spearman=rank_correlation([c["reward"] for c in candidates],
                                                            [c["mean_prefix"] for c in candidates]),
                     high_reward_definition="top quartile by workload mean reward, including all cutoff ties",
                     high_reward_cutoff=cutoff, high_reward_candidates=len(top),
                     high_reward=summarize_pairs([x for r, rows in members
                                                 if r["candidate"]["candidate_id"] in top for x in rows]),
                     above_base_candidates=len(better),
                     above_base=summarize_pairs([x for r, rows in members
                                                if r["candidate"]["candidate_id"] in better for x in rows]))
        output.setdefault(workload, {})[str(sigma)] = stats
    return output


def feasibility_gate(summary, protocol, *, controls_passed, complete):
    """Apply a frozen engineering screen, not a significance/speedup test."""
    config = protocol["verifier_gate"]
    checks = []
    for sigma in protocol["sigmas"]:
        for k in config["block_sizes"]:
            passing = True
            for workload in protocol["workloads"]:
                s = summary.get(workload["name"], {}).get(str(sigma), {})
                reduction = s.get("simulation", {}).get(str(k), {}).get("token_weighted_round_reduction")
                acceptance = s.get("first_k", {}).get(str(k), {}).get("fraction_all_pairs", 0)
                passing &= (s.get("candidates", 0) >= config["minimum_candidates_per_sigma"]
                            and s.get("common_prefix_tokens", {}).get("median", 0) >= config["minimum_median_prefix"]
                            and reduction is not None and reduction >= config["minimum_round_reduction"]
                            and acceptance >= config["minimum_initial_block_acceptance"])
            checks.append({"sigma": sigma, "block_size": k, "both_workloads_pass": bool(passing)})
    enough_data = all(summary.get(w["name"], {}).get(str(s), {}).get("candidates", 0)
                      >= config["minimum_candidates_per_sigma"]
                      for w in protocol["workloads"] for s in protocol["sigmas"])
    valid = controls_passed and complete and enough_data
    passed = bool(valid and any(c["both_workloads_pass"] for c in checks))
    return {"pass": passed, "status": "pass" if passed else "negative" if valid else "inconclusive",
            "controls_passed": controls_passed, "bounded_sweep_complete": complete,
            "enough_candidates": enough_data,
            "checks": checks, "actual_speedup_measured": False}
