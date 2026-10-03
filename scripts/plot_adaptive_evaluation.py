"""Static research figures from frozen development and locked validation analyses."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = {"Development": read(args.development / "summary.json"),
                 "Validation A": read(args.validation / "validation_a.summary.json"),
                 "Validation B": read(args.validation / "validation_b.summary.json")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for label, summary in summaries.items():
        keys = sorted((k for k in summary if k.startswith("subset-")), key=lambda k: int(k.split('-')[1]))
        budgets = [int(k.split('-')[1]) for k in keys]
        for ax, field, subfield in ((axes[0], "spearman", None), (axes[1], "topk", "recall"), (axes[2], "best_regret", None)):
            stats = [summary[k][field] if subfield is None else summary[k][field]["10"][subfield] for k in keys]
            ax.plot(budgets, [s["mean"] for s in stats], marker="o", label=label)
            ax.fill_between(budgets, [s["p10"] for s in stats], [s["p90"] for s in stats], alpha=.12)
    for ax, title in zip(axes, ("Partial/full Spearman", "Strict top-10 recall", "Selected-best full-score regret")):
        ax.set(xlabel="Prompts per candidate", title=title)
        ax.grid(alpha=.2)
        ax.axvline(80, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(.9, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(.01, color="black", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Random fixed subsets: means and p10–p90 across prompt orders; not independent confidence intervals")
    fig.savefig(args.out / "subset-quality.png", dpi=170)
    fig.savefig(args.out / "subset-quality.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    lock = read(args.development / "policy-lock.json")
    for ax, label in zip(axes, ("Validation A", "Validation B")):
        seen_families = set()
        for name, s in summaries[label].items():
            if name.startswith("prefix-"):
                continue
            family = name.split('-')[0]
            marker = {"subset": "o", "halving": "s", "hoeffding": "^", "wilson": "D"}[family]
            color = {"subset": "tab:blue", "halving": "tab:orange", "hoeffding": "tab:green", "wilson": "tab:red"}[family]
            point = s["pair_fraction"]["mean"], s["topk"]["10"]["recall"]["mean"]
            ax.scatter(*point, marker=marker, s=45, color=color,
                       label=family if family not in seen_families else None)
            seen_families.add(family)
            if name == lock["chosen_policy_id"]:
                ax.scatter(*point, marker="*", s=160, facecolors="none", edgecolors="black", label="locked policy")
                ax.annotate(name, point, fontsize=8, xytext=(4, 8), textcoords="offset points")
        ax.axvline(.4, color="gray", linestyle="--")
        ax.axhline(.9, color="gray", linestyle="--")
        ax.set(xlabel="Fraction of selection candidate/prompt pairs", ylabel="Mean strict top-10 recall", title=label, xlim=(0, 1.05), ylim=(0, 1.05))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Mean cost/recall; the gate uses per-order joint success, including regret")
    fig.savefig(args.out / "cost-recall.png", dpi=170)
    fig.savefig(args.out / "cost-recall.svg")
    plt.close(fig)
    matrix = read(args.run / "matrix-development.json")
    rewards = np.array([r["rewards"][:200] for r in matrix["rows"]])
    diag = read(args.development / "diagnostics.json")
    truth = diag["full_ranking"]
    trials = read(args.development / "trials.json.gz")
    early = {r["policy"]["prompts"]: r for r in trials if r["policy"]["family"] == "subset" and r["order_seed"] == 100}
    groups = {"Full top candidates": truth[:3], "Middle candidates": truth[len(truth)//2:len(truth)//2+3],
              "Poor candidates": truth[-3:],
              "False-positive early leaders": [i for i in early[4]["result"]["ranking"][:10] if i not in truth[:20]][:3],
              "False-negative late winners": [i for i in truth[:10] if i not in early[32]["result"]["ranking"][:50]][:3]}
    order = early[4]["prompt_order"]
    paths = np.cumsum(rewards[:, order], axis=1) / np.arange(1, 201)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for ax, (name, indices) in zip(axes.flat, groups.items()):
        for i in indices:
            c = matrix["rows"][i]["candidate"]
            ax.plot(range(1, 201), paths[i], label=f"{c['seed']}, sigma={c['sigma']}")
        if not indices:
            ax.text(.5, .5, "No examples under frozen rule", ha="center", transform=ax.transAxes)
        ax.set(title=name, xlabel="Prompts observed", ylabel="Partial mean reward", ylim=(0, 1))
        if indices:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Development candidate trajectories; fixed prompt-order seed 100")
    fig.savefig(args.out / "candidate-trajectories.png", dpi=170)
    fig.savefig(args.out / "candidate-trajectories.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
