"""Master summary generation across all completed benchmark cards -- reads card.json files
already written by card.write_card() under a results root and writes summary.md (compact
table) + summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def find_cards(results_root: "str | Path") -> List[Path]:
    return sorted(Path(results_root).glob("*/card.json"))


def load_cards(results_root: "str | Path") -> List[dict]:
    return [json.loads(p.read_text()) for p in find_cards(results_root)]


def build_summary_table(cards: List[dict]) -> str:
    columns = ["Capability", "Dataset", "N", "Base", "Repeat", "Image integrity", "Parser", "Visual sanity", "Status"]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("-" * (len(c) + 2) for c in columns) + "|"
    lines = [header, separator]

    for card in cards:
        base = card["base_metrics"].get("primary_metric")
        repeat_metrics = card.get("repeat_metrics") or {}
        repeat = repeat_metrics.get("primary_metric")
        integrity = card["integrity"]
        parser_rate = card["base_metrics"].get("parser_failure_rate", 0.0)
        sanity = card.get("image_sanity")

        image_integrity_str = f"{integrity['n_valid_images']}/{integrity['n_loaded']}"
        base_str = f"{base:.3f}" if base is not None else "N/A"
        repeat_str = f"{repeat:.3f}" if repeat is not None else "N/A"
        parser_str = f"{parser_rate:.1%}"
        sanity_str = f"{sanity['correct_minus_shuffled']:+.3f}" if sanity else "N/A"

        row_values = [
            card["capability"], card["dataset"], str(card["subset_size"]),
            base_str, repeat_str, image_integrity_str, parser_str, sanity_str, card["status"],
        ]
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def build_summary_json(cards: List[dict]) -> dict:
    status_counts = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}
    for card in cards:
        status_counts[card["status"]] = status_counts.get(card["status"], 0) + 1

    return {
        "n_capabilities": len(cards),
        "status_counts": status_counts,
        "all_pass": len(cards) > 0 and status_counts.get("FAIL", 0) == 0 and status_counts.get("NEEDS_REVIEW", 0) == 0,
        "cards": cards,
    }


def write_summary(results_root: "str | Path") -> dict:
    cards = load_cards(results_root)
    table = build_summary_table(cards)
    summary_json = build_summary_json(cards)

    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text(table)
    (root / "summary.json").write_text(json.dumps(summary_json, indent=2))
    return summary_json
