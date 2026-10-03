"""Render descriptive agreement distributions from a validated run summary."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary = json.loads((args.run / "summary.json").read_text())
    fig, axes = plt.subplots(len(summary), 2, figsize=(10, 6), layout="constrained")
    for row, (name, scales) in enumerate(summary.items()):
        left, right = axes[row]
        for sigma, s in sorted(scales.items(), key=lambda pair: float(pair[0])):
            points = sorted((int(k), v) for k, v in s["common_prefix_histogram"].items())
            cumulative, x, y = 0, [], []
            for token, count in points:
                cumulative += count
                x.append(token)
                y.append(cumulative / s["pairs"])
            left.step(x, y, where="post", label=f"sigma={sigma}")
            ks = (1, 2, 4, 8, 16, 32)
            right.plot(ks, [s["first_k"][str(k)]["fraction_all_pairs"] for k in ks], marker="o", label=f"sigma={sigma}")
        left.set(xlabel="Common-prefix tokens (view clipped at 64)", ylabel="Empirical cumulative fraction",
                 title=name, xlim=(0, 64), ylim=(0, 1.02))
        right.set(xlabel="Initial draft block size", ylabel="Fraction of all pairs matching full block",
                  title=name, ylim=(0, 1.02), xscale="log")
        right.set_xticks(ks, labels=[str(k) for k in ks])
        for ax in (left, right):
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Independent greedy token agreement; descriptive only, no measured speedup")
    args.out.mkdir(parents=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / ("agreement-distributions." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
