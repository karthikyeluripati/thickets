#!/usr/bin/env python3
"""Clone the official RandOpt repository as an external, gitignored dependency.

We do not vendor (copy) upstream source into this repository -- it has no declared
license (see EXTERNAL_COMMIT.txt). Instead this script clones it locally, pinned to
a specific commit, so eval_base.py / run_randopt.py can invoke it via subprocess.
The clone directory (external/RandOpt/) is listed in .gitignore.
"""
import subprocess
import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/sunrainyg/RandOpt"
PINNED_COMMIT = "536df0a308f3990b6270c991fbb96bd0b779a58e"
CLONE_DIR = Path(__file__).resolve().parent / "RandOpt"


def main() -> int:
    if CLONE_DIR.exists():
        head = subprocess.run(
            ["git", "-C", str(CLONE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        if head.returncode == 0 and head.stdout.strip() == PINNED_COMMIT:
            print(f"Already present at pinned commit {PINNED_COMMIT[:8]}: {CLONE_DIR}")
            return 0
        print(
            f"ERROR: {CLONE_DIR} exists but is not at the pinned commit "
            f"({head.stdout.strip() or 'unknown'} != {PINNED_COMMIT}). "
            "Remove it manually and re-run if you want a clean re-clone.",
            file=sys.stderr,
        )
        return 1

    print(f"Cloning {UPSTREAM_URL} -> {CLONE_DIR} ...")
    subprocess.run(["git", "clone", UPSTREAM_URL, str(CLONE_DIR)], check=True)
    subprocess.run(["git", "-C", str(CLONE_DIR), "checkout", PINNED_COMMIT], check=True)
    print(f"Checked out pinned commit {PINNED_COMMIT[:8]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
