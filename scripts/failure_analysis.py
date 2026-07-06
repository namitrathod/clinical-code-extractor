"""Find worst fine-tuned predictions for README error analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def note_score(gold_icd: set[str], gold_cpt: set[str], pred_icd: set[str], pred_cpt: set[str]) -> float:
    icd_hit = len(gold_icd & pred_icd)
    cpt_hit = len(gold_cpt & pred_cpt)
    icd_denom = max(len(gold_icd | pred_icd), 1)
    cpt_denom = max(len(gold_cpt | pred_cpt), 1)
    return (icd_hit / icd_denom + cpt_hit / cpt_denom) / 2


def diagnose(gold_icd: list[str], gold_cpt: list[str], pred_icd: list[str], pred_cpt: list[str]) -> str:
    if not pred_icd and not pred_cpt:
        return "empty prediction"
    if set(gold_icd) == set(pred_icd) and set(gold_cpt) != set(pred_cpt):
        return "CPT level or procedure mismatch"
    if set(gold_icd) != set(pred_icd) and set(gold_cpt) == set(pred_cpt):
        return "missed or extra ICD code"
    if len(gold_icd) > len(pred_icd):
        return "missed secondary ICD code"
    return "ICD and CPT both partially wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "results" / "predictions_finetuned.jsonl")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "failure_cases.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    gold_by_id = {row["id"]: row for row in load_jsonl(args.gold)}
    preds = load_jsonl(args.predictions)

    scored = []
    for pred in preds:
        gold = gold_by_id[pred["id"]]
        g_icd, g_cpt = set(gold["icd10"]), set(gold["cpt"])
        p_icd, p_cpt = set(pred.get("icd10", [])), set(pred.get("cpt", []))
        scored.append({
            "id": pred["id"],
            "score": note_score(g_icd, g_cpt, p_icd, p_cpt),
            "note_snippet": gold["note"][:200],
            "gold_icd10": gold["icd10"],
            "gold_cpt": gold["cpt"],
            "pred_icd10": pred.get("icd10", []),
            "pred_cpt": pred.get("cpt", []),
            "diagnosis": diagnose(gold["icd10"], gold["cpt"], pred.get("icd10", []), pred.get("cpt", [])),
        })

    scored.sort(key=lambda x: x["score"])
    worst = scored[: args.top_k]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(worst, indent=2), encoding="utf-8")
    print(f"Wrote {len(worst)} failure cases to {args.output}")


if __name__ == "__main__":
    main()
