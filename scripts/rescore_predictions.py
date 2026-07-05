"""Re-parse raw_output in an existing predictions JSONL with the latest validator.

Useful after improving parse_model_output / filter_codes without re-running GPU inference.

Usage:
  python scripts/rescore_predictions.py \\
    --input predictions_baseline.jsonl \\
    --output predictions_baseline_rescored.jsonl

  python scripts/eval.py --predictions predictions_baseline_rescored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validate import load_whitelists, process_model_output  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-score predictions from saved raw_output.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    icd_whitelist, cpt_whitelist = load_whitelists()
    rows = load_jsonl(args.input)
    output = args.output or args.input.with_name(f"{args.input.stem}_rescored.jsonl")

    rescored: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("raw_output", "")
        processed = process_model_output(raw, icd_whitelist, cpt_whitelist)
        rescored.append({"id": row["id"], **processed})

    with output.open("w", encoding="utf-8") as f:
        for row in rescored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with_codes = sum(1 for row in rescored if row["icd10"] or row["cpt"])
    print(f"Wrote {len(rescored)} rows to {output}")
    print(f"Rows with at least one whitelisted code: {with_codes}/{len(rescored)}")
    print("Next: python scripts/eval.py --predictions", output)


if __name__ == "__main__":
    main()
