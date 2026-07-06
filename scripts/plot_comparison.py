"""Generate results/comparison_chart.png from baseline and finetuned JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    baseline_path = ROOT / "results" / "baseline.json"
    finetuned_path = ROOT / "results" / "finetuned.json"
    out_path = ROOT / "results" / "comparison_chart.png"

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
    finetuned = json.loads(finetuned_path.read_text(encoding="utf-8"))["metrics"]

    labels = ["ICD micro F1", "CPT micro F1", "ICD exact match", "CPT exact match"]
    keys = [
        "icd10_micro_f1",
        "cpt_micro_f1",
        "exact_set_match_icd10",
        "exact_set_match_cpt",
    ]
    base_vals = [baseline[k] for k in keys]
    ft_vals = [finetuned[k] for k in keys]

    import matplotlib.pyplot as plt

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline")
    ax.bar([i + width / 2 for i in x], ft_vals, width, label="Fine-tuned")
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs fine-tuned (500-note test set)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
