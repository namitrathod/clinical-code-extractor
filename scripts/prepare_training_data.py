"""Convert train/val JSONL into SFT-ready JSONL for QLoRA training.

Each output row keeps note + labels; the Colab notebook applies the Gemma chat
template via src.prompts.format_training_text().

Usage:
  python scripts/prepare_training_data.py
  python scripts/prepare_training_data.py --spot-check 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import build_completion, build_prompt  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    note = row["note"]
    icd10 = row["icd10"]
    cpt = row["cpt"]
    prompt = build_prompt(note)
    completion = build_completion(icd10, cpt)
    return {
        "id": row["id"],
        "note": note,
        "icd10": icd10,
        "cpt": cpt,
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SFT JSONL for QLoRA training.")
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train.jsonl")
    p.add_argument("--val", type=Path, default=ROOT / "data" / "val.jsonl")
    p.add_argument("--train-out", type=Path, default=ROOT / "data" / "train_sft.jsonl")
    p.add_argument("--val-out", type=Path, default=ROOT / "data" / "val_sft.jsonl")
    p.add_argument("--spot-check", type=int, default=0, help="Print N joined examples")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = [to_sft_row(r) for r in load_jsonl(args.train)]
    val_rows = [to_sft_row(r) for r in load_jsonl(args.val)]

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.val_out, val_rows)

    print(f"Wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"Wrote {len(val_rows)} val rows   -> {args.val_out}")

    if args.spot_check:
        print("\n--- Spot check (prompt + completion should join seamlessly) ---")
        for row in train_rows[: args.spot_check]:
            print(f"\n[{row['id']}]")
            print(row["text"][-200:])


if __name__ == "__main__":
    main()
