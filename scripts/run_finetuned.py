"""Run fine-tuned Gemma + LoRA adapter on a JSONL split.

Usage (Colab T4):
  python scripts/run_finetuned.py \\
    --adapter your-username/gemma4-icd-cpt-qlora \\
    --gold data/test.jsonl \\
    --output predictions_finetuned.jsonl

Then:
  python scripts/eval.py --predictions predictions_finetuned.jsonl --output results/finetuned.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_baseline import generate_one, load_jsonl  # noqa: E402
from src.validate import load_whitelists, process_model_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuned inference with LoRA adapter.")
    p.add_argument("--adapter", type=str, required=True, help="HF Hub id or local checkpoint path")
    p.add_argument("--gold", type=Path, default=ROOT / "data" / "test.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "predictions_finetuned.jsonl")
    p.add_argument("--model", type=str, default="google/gemma-4-E4B-it")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    icd_whitelist, cpt_whitelist = load_whitelists()
    rows = load_jsonl(args.gold)
    if args.max_samples:
        rows = rows[: args.max_samples]

    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading base {args.model} + adapter {args.adapter}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    predictions = []
    start = time.time()
    for i, row in enumerate(rows, start=1):
        raw = generate_one(model, tokenizer, row["note"], args.max_new_tokens)
        filtered = process_model_output(raw, icd_whitelist, cpt_whitelist)
        predictions.append({"id": row["id"], **filtered})
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time() - start:.0f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        import json

        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print("Next: python scripts/eval.py --predictions", args.output, "--output results/finetuned.json")


if __name__ == "__main__":
    main()
