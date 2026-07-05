"""Run base Gemma 4 E4B on a JSONL split and write predictions for eval.py.

Usage (Colab T4 recommended for full test set):
  python scripts/run_baseline.py --gold data/test.jsonl --output predictions_baseline.jsonl

Smoke test locally:
  python scripts/run_baseline.py --gold data/test.jsonl --output predictions_baseline.jsonl --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import format_for_model  # noqa: E402
from src.validate import filter_codes, load_whitelists, parse_model_output  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_one(model, tokenizer, note: str, max_new_tokens: int) -> str:
    prompt = format_for_model(tokenizer, note, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline inference with base Gemma model.")
    p.add_argument("--gold", type=Path, default=ROOT / "data" / "test.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "predictions_baseline.jsonl")
    p.add_argument("--model", type=str, default="google/gemma-4-E4B-it")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--4bit", dest="use_4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    icd_whitelist, cpt_whitelist = load_whitelists()
    rows = load_jsonl(args.gold)
    if args.max_samples:
        rows = rows[: args.max_samples]

    print(f"Loading model {args.model} (4bit={args.use_4bit})...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
    )
    model.eval()

    predictions: list[dict[str, Any]] = []
    start = time.time()

    for i, row in enumerate(rows, start=1):
        raw = generate_one(model, tokenizer, row["note"], args.max_new_tokens)
        parsed = parse_model_output(raw)
        filtered = filter_codes(parsed, icd_whitelist, cpt_whitelist)

        predictions.append({
            "id": row["id"],
            "raw_output": raw,
            "json_valid": filtered["json_valid"],
            "icd10_pre": filtered["icd10_pre"],
            "cpt_pre": filtered["cpt_pre"],
            "icd10": filtered["icd10"],
            "cpt": filtered["cpt"],
            "invalid_icd10": filtered["invalid_icd10"],
            "invalid_cpt": filtered["invalid_cpt"],
        })

        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - start
            print(f"  {i}/{len(rows)} ({elapsed:.0f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print("Next: python scripts/eval.py --predictions", args.output)


if __name__ == "__main__":
    main()
