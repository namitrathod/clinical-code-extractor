"""Score model predictions against gold labels (baseline or fine-tuned)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validate import load_whitelists  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows}


def multilabel_f1(
    gold_sets: list[list[str]],
    pred_sets: list[list[str]],
    vocabulary: list[str],
) -> tuple[float, float]:
    mlb = MultiLabelBinarizer(classes=vocabulary)
    y_true = mlb.fit_transform(gold_sets)
    y_pred = mlb.transform(pred_sets)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return micro, macro


def exact_set_match(gold_sets: list[list[str]], pred_sets: list[list[str]]) -> float:
    if not gold_sets:
        return 0.0
    matches = sum(set(g) == set(p) for g, p in zip(gold_sets, pred_sets))
    return matches / len(gold_sets)


def top_k_primary_recall(
    gold_primary: list[str],
    pred_ordered: list[list[str]],
    k: int,
) -> float:
    if not gold_primary:
        return 0.0
    hits = 0
    for gold, preds in zip(gold_primary, pred_ordered):
        top_k = preds[:k]
        if gold in top_k:
            hits += 1
    return hits / len(gold_primary)


def compute_metrics(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    icd_vocab: list[str],
    cpt_vocab: list[str],
) -> dict[str, Any]:
    gold_by_id = index_by_id(gold_rows)
    pred_by_id = index_by_id(pred_rows)

    missing_preds = [gid for gid in gold_by_id if gid not in pred_by_id]
    if missing_preds:
        raise ValueError(f"Predictions missing {len(missing_preds)} gold ids (e.g. {missing_preds[:3]})")

    ordered_gold = [gold_by_id[pid] for pid in pred_by_id if pid in gold_by_id]
    ordered_pred = [pred_by_id[g["id"]] for g in ordered_gold]

    gold_icd = [g["icd10"] for g in ordered_gold]
    gold_cpt = [g["cpt"] for g in ordered_gold]
    pred_icd = [p.get("icd10", []) for p in ordered_pred]
    pred_cpt = [p.get("cpt", []) for p in ordered_pred]
    pred_icd_pre = [p.get("icd10_pre", p.get("icd10_pred_pre_filter", [])) for p in ordered_pred]
    pred_cpt_pre = [p.get("cpt_pre", p.get("cpt_pred_pre_filter", [])) for p in ordered_pred]
    gold_primary = [g["primary_icd10"] for g in ordered_gold]

    icd_micro, icd_macro = multilabel_f1(gold_icd, pred_icd, icd_vocab)
    cpt_micro, cpt_macro = multilabel_f1(gold_cpt, pred_cpt, cpt_vocab)

    json_valid = [bool(p.get("json_valid", False)) for p in ordered_pred]
    invalid_icd_notes = sum(1 for p in ordered_pred if p.get("invalid_icd10"))
    invalid_cpt_notes = sum(1 for p in ordered_pred if p.get("invalid_cpt"))
    total_invalid_icd_codes = sum(len(p.get("invalid_icd10", [])) for p in ordered_pred)
    total_invalid_cpt_codes = sum(len(p.get("invalid_cpt", [])) for p in ordered_pred)
    total_pre_icd_codes = sum(len(x) for x in pred_icd_pre)
    total_pre_cpt_codes = sum(len(x) for x in pred_cpt_pre)

    return {
        "n_samples": len(ordered_gold),
        "metrics": {
            "icd10_macro_f1": round(icd_macro, 4),
            "icd10_micro_f1": round(icd_micro, 4),
            "cpt_macro_f1": round(cpt_macro, 4),
            "cpt_micro_f1": round(cpt_micro, 4),
            "micro_f1_avg": round((icd_micro + cpt_micro) / 2, 4),
            "exact_set_match_icd10": round(exact_set_match(gold_icd, pred_icd), 4),
            "exact_set_match_cpt": round(exact_set_match(gold_cpt, pred_cpt), 4),
            "top1_recall_primary_icd10": round(top_k_primary_recall(gold_primary, pred_icd, 1), 4),
            "top3_recall_primary_icd10": round(top_k_primary_recall(gold_primary, pred_icd, 3), 4),
            "json_validity_rate": round(sum(json_valid) / len(json_valid), 4),
            "notes_with_invalid_icd10_rate": round(invalid_icd_notes / len(ordered_pred), 4),
            "notes_with_invalid_cpt_rate": round(invalid_cpt_notes / len(ordered_pred), 4),
            "invalid_icd10_code_rate_pre_filter": round(
                total_invalid_icd_codes / max(total_pre_icd_codes, 1), 4
            ),
            "invalid_cpt_code_rate_pre_filter": round(
                total_invalid_cpt_codes / max(total_pre_cpt_codes, 1), 4
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ICD/CPT predictions.")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold", type=Path, default=ROOT / "data" / "test.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "baseline.json")
    p.add_argument("--model-name", type=str, default="google/gemma-4-E4B-it")
    p.add_argument("--split", type=str, default="test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    icd_whitelist, cpt_whitelist = load_whitelists()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.predictions)

    result = compute_metrics(
        gold_rows,
        pred_rows,
        icd_vocab=sorted(icd_whitelist.keys()),
        cpt_vocab=sorted(cpt_whitelist.keys()),
    )
    result["model"] = args.model_name
    result["split"] = args.split
    result["predictions_file"] = str(args.predictions)
    result["gold_file"] = str(args.gold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
