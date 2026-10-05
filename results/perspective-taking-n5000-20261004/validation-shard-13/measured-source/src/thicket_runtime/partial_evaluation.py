"""Simple offline evaluation policies with an explicit observation boundary.

Policies receive candidate identities, a prompt order, and observe() only. Full
scores and hidden rewards are consumed solely by the separate evaluator.
"""
from collections import Counter
import hashlib
import math

import numpy as np
from .agreement import rank_correlation


def tie_keys(candidate_ids):
    return [hashlib.sha256(("rank-tie-v1:" + c).encode()).hexdigest() for c in candidate_ids]


def order_scores(scores, keys, ids=None):
    ids = range(len(scores)) if ids is None else ids
    return sorted(ids, key=lambda i: (-float(scores[i]), keys[i]))


class MatrixOracle:
    """Records exactly which finite-matrix cells a simulated policy requests."""
    def __init__(self, rewards, tokens=None):
        self._rewards = np.asarray(rewards, dtype=float)
        if self._rewards.ndim != 2 or not np.isfinite(self._rewards).all() or np.any((self._rewards < 0) | (self._rewards > 1)):
            raise ValueError("finite [0,1] reward matrix required")
        self._tokens = np.ones_like(self._rewards, dtype=int) if tokens is None else np.asarray(tokens, dtype=int)
        if self._tokens.shape != self._rewards.shape or np.any(self._tokens < 0):
            raise ValueError("invalid token matrix")
        self.shape = self._rewards.shape
        self.mask = np.zeros(self.shape, dtype=bool)
        self.calls = []

    def observe(self, candidates, prompts):
        candidates, prompts = list(candidates), list(prompts)
        ix = np.ix_(candidates, prompts)
        if self.mask[ix].any():
            raise ValueError("policy requested a previously observed pair")
        self.mask[ix] = True
        self.calls.append({"candidates": candidates, "prompts": prompts})
        return self._rewards[ix].copy()

    def cost(self):
        return {"evaluations": int(self.mask.sum()), "pair_fraction": float(self.mask.mean()),
                "generated_tokens": int(self._tokens[self.mask].sum()),
                "token_fraction": float(self._tokens[self.mask].sum() / self._tokens.sum()) if self._tokens.sum() else 0.}


def bounds(means, n, total, family, parameter, candidates, stages):
    means = np.asarray(means, dtype=float)
    if n == total:
        return means.copy(), means.copy()
    if family == "hoeffding":
        width = math.sqrt(math.log(2 * candidates * stages / parameter) / (2 * n))
        return np.maximum(0, means - width), np.minimum(1, means + width)
    if family == "wilson":
        z2 = parameter**2
        center = (means + z2 / (2 * n)) / (1 + z2 / n)
        width = parameter * np.sqrt(means * (1 - means) / n + z2 / (4 * n*n)) / (1 + z2 / n)
        return np.maximum(0, center - width), np.minimum(1, center + width)
    raise ValueError("unknown bound family")


def run_policy(observe, shape, candidate_ids, prompt_order, policy, *, k=10, race_stages=(8, 16, 32, 64, 80, 100, 200)):
    """No full reward matrix or full ranking is passed to this function."""
    count, total = shape
    if len(candidate_ids) != count or sorted(prompt_order) != list(range(total)) or not 0 < k <= count:
        raise ValueError("invalid policy dimensions or prompt permutation")
    keys = tie_keys(candidate_ids)
    sums, counts = np.zeros(count), np.zeros(count, dtype=int)
    alive, discarded, history = list(range(count)), [], []
    seen = 0
    family = policy["family"]
    if family in ("subset", "prefix"):
        stages = [min(policy["prompts"], total)]
    elif family == "halving":
        stages, n = [], policy["initial"]
        while n < total:
            stages.append(n)
            n *= 2
        stages.append(total)
    else:
        stages = sorted({min(n, total) for n in race_stages} | {total})
    for n in stages:
        if n <= seen:
            continue
        prompts = list(prompt_order[seen:n])
        observations = observe(alive, prompts)
        sums[alive] += observations.sum(axis=1)
        counts[alive] += len(prompts)
        seen = n
        means = sums / np.maximum(counts, 1)
        removed = []
        if family == "halving" and len(alive) > k:
            ranked = order_scores(means, keys, alive)
            retain = k if n == total else max(k, math.ceil(len(alive) / 2))
            alive, removed = ranked[:retain], ranked[retain:]
        elif family in ("hoeffding", "wilson") and n < total:
            lo, hi = bounds(means[alive], n, total, family, policy["parameter"], count, len(stages))
            frontier = sorted(lo, reverse=True)[k - 1]
            removed = [i for i, upper in zip(alive, hi) if upper < frontier]
            remove_set = set(removed)
            alive = [i for i in alive if i not in remove_set]
            if len(alive) < k:
                raise AssertionError("bound racing cannot eliminate below K")
        if removed:
            discarded.append(order_scores(means, keys, removed))
        history.append({"prompts_per_survivor": n, "survivors": list(alive), "eliminated": removed})
    means = sums / np.maximum(counts, 1)
    ranked = order_scores(means, keys, alive) + [i for group in reversed(discarded) for i in group]
    if sorted(ranked) != list(range(count)):
        raise AssertionError("policy ranking lost candidates")
    return {"ranking": ranked, "observed_means": means.tolist(), "counts": counts.tolist(),
            "eliminated": [i for group in discarded for i in group], "history": history}


def correlation(x, y, method="spearman"):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    if method == "spearman":
        result = rank_correlation(list(x), list(y))
    else:
        a, b = np.asarray(x), np.asarray(y)
        i, j = np.triu_indices(len(a), 1)
        dx, dy = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
        denominator = math.sqrt(np.count_nonzero(dx) * np.count_nonzero(dy))
        result = float(np.sum(dx * dy) / denominator) if denominator else None
    if result is None:
        return None
    return float(result) if np.isfinite(result) else None


def evaluate(result, oracle, full_rewards, candidate_ids, ks=(1, 5, 10, 20)):
    full = np.asarray(full_rewards, dtype=float).mean(axis=1)
    truth = order_scores(full, tie_keys(candidate_ids))
    ranked, eliminated = result["ranking"], set(result["eliminated"])
    selected = ranked[0]
    metrics = {**oracle.cost(), "best_full_reward": float(full[truth[0]]),
               "selected_best_full_reward": float(full[selected]),
               "best_regret": float(full[truth[0]] - full[selected]),
               "top1_recovery": selected == truth[0],
               "spearman": correlation(result["observed_means"], full),
               "kendall_tau_b": correlation(result["observed_means"], full, "kendall"),
               "score_rmse": float(np.sqrt(np.mean((np.array(result["observed_means"]) - full)**2))),
               "mean_absolute_score_error": float(np.mean(np.abs(np.array(result["observed_means"]) - full))),
               "topk": {}}
    for k in ks:
        if k > len(truth):
            continue
        gold, chosen = set(truth[:k]), set(ranked[:k])
        overlap = len(gold & chosen)
        cutoff = full[truth[k - 1]]
        strict = set(np.flatnonzero(full > cutoff))
        boundary = set(np.flatnonzero(full == cutoff))
        tie_credit = len(chosen & strict) + min(k - len(strict), len(chosen & boundary))
        metrics["topk"][str(k)] = {"recall": overlap / k, "overlap": overlap,
            "jaccard": overlap / (2*k - overlap), "tie_aware_recall": tie_credit / k,
            "boundary_tie_count": len(boundary), "full_topk_mean": float(full[truth[:k]].mean()),
            "selected_topk_mean": float(full[ranked[:k]].mean()),
            "false_eliminations": len(gold & eliminated),
            "miss_outside_top2k": len(gold - set(ranked[:min(2*k, len(ranked))])) / k,
            "miss_outside_top5k": len(gold - set(ranked[:min(5*k, len(ranked))])) / k}
    metrics["theoretical_pair_reduction"] = 1 - metrics["pair_fraction"]
    metrics["theoretical_token_reduction"] = 1 - metrics["token_fraction"]
    return metrics


def joint_success(metrics, gate):
    return (metrics["pair_fraction"] <= gate["maximum_pair_fraction"] + 1e-12
            and metrics["topk"]["10"]["recall"] + 1e-12 >= gate["minimum_top10_recall"]
            and metrics["best_regret"] <= gate["maximum_best_regret"] + 1e-12)


def vote_answers(answers, ranking, k):
    """Pinned upstream majority behavior: omit empty answers, stable count ties."""
    answers = np.asarray(answers, dtype=object)
    result = []
    for prompt in range(answers.shape[1]):
        votes = [answers[i, prompt] for i in ranking[:k] if answers[i, prompt]]
        result.append(Counter(votes).most_common(1)[0][0] if votes else "")
    return result
