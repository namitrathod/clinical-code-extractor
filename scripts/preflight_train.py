"""Pre-flight checks before QLoRA training on GPU.

Run locally (no GPU needed for most checks):
  python scripts/preflight_train.py
  python scripts/preflight_train.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import (  # noqa: E402
    build_completion,
    build_user_content,
    format_for_model,
    format_training_text,
)
from src.validate import load_whitelists, process_model_output  # noqa: E402

MODEL_ID = "google/gemma-4-E4B-it"
PLACEHOLDER_HUB = "your-username/gemma4-icd-cpt-qlora"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_files() -> list[str]:
    issues: list[str] = []
    required = [
        ROOT / "data" / "train.jsonl",
        ROOT / "data" / "val.jsonl",
        ROOT / "data" / "test.jsonl",
        ROOT / "codes" / "icd10_whitelist.json",
        ROOT / "codes" / "cpt_whitelist.json",
        ROOT / "src" / "prompts.py",
        ROOT / "src" / "validate.py",
        ROOT / "scripts" / "prepare_training_data.py",
    ]
    for path in required:
        if not path.exists():
            issues.append(f"MISSING: {path.relative_to(ROOT)}")
    return issues


def check_raw_data() -> list[str]:
    issues: list[str] = []
    icd_w, cpt_w = load_whitelists()
    for split in ("train", "val"):
        rows = load_jsonl(ROOT / "data" / f"{split}.jsonl")
        if split == "train" and len(rows) < 1000:
            issues.append(f"WARN: train.jsonl only has {len(rows)} rows")
        for row in rows[:50]:
            for field in ("id", "note", "icd10", "cpt", "primary_icd10"):
                if field not in row:
                    issues.append(f"{split} row missing field '{field}'")
                    break
            for code in row.get("icd10", []):
                if code not in icd_w:
                    issues.append(f"Gold ICD {code} not in whitelist ({row['id']})")
                    break
            for code in row.get("cpt", []):
                if code not in cpt_w:
                    issues.append(f"Gold CPT {code} not in whitelist ({row['id']})")
                    break
    return issues


def check_sft_data() -> list[str]:
    issues: list[str] = []
    from scripts.prepare_training_data import to_sft_row

    for split in ("train", "val"):
        src = load_jsonl(ROOT / "data" / f"{split}.jsonl")
        for row in src[:20]:
            sft = to_sft_row(row)
            completion = json.loads(sft["completion"])
            if set(completion.keys()) != {"icd10", "cpt"}:
                issues.append(f"SFT completion wrong keys: {completion.keys()}")
            if completion["icd10"] != row["icd10"] or completion["cpt"] != row["cpt"]:
                issues.append(f"SFT labels mismatch for {row['id']}")
            if not sft["text"].endswith(sft["completion"]):
                issues.append(f"prompt+completion join broken for {row['id']}")
    return issues


def check_imports() -> list[str]:
    issues: list[str] = []
    for pkg in ("transformers", "peft", "trl", "datasets", "bitsandbytes", "accelerate"):
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"Package not installed locally: {pkg} (Colab pip cell installs it)")
    try:
        import trl
        from trl import SFTConfig, SFTTrainer

        ver = getattr(trl, "__version__", "unknown")
        cfg_fields = set(SFTConfig.__dataclass_fields__)
        if "max_seq_length" not in cfg_fields and "max_length" not in cfg_fields:
            issues.append("TRL SFTConfig missing max_length — check trl version")
        elif "max_seq_length" in cfg_fields and "max_length" not in cfg_fields:
            issues.append("INFO: older TRL uses max_seq_length in notebook")
        elif "max_length" in cfg_fields:
            issues.append("INFO: TRL uses max_length (not max_seq_length) in SFTConfig")
        if "assistant_only_loss" not in cfg_fields:
            issues.append(f"TRL {ver}: assistant_only_loss not in SFTConfig — set False or upgrade trl")
        sig = SFTTrainer.__init__.__code__.co_varnames
        if "processing_class" not in sig and "tokenizer" in sig:
            issues.append("SFTTrainer expects tokenizer= not processing_class= — update notebook")
    except ImportError:
        pass
    return issues


def check_tokenizer_offline(max_seq_length: int) -> list[str]:
    """Token-length check using transformers if model config is reachable."""
    issues: list[str] = []
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:
        issues.append(f"SKIP token check (need HF access): {exc}")
        return issues

    train = load_jsonl(ROOT / "data" / "train.jsonl")
    lengths: list[int] = []
    over: list[tuple[str, int]] = []
    for row in train:
        text = format_training_text(tokenizer, row["note"], row["icd10"], row["cpt"])
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        if n > max_seq_length:
            over.append((row["id"], n))

    if lengths:
        lengths.sort()
        p95 = lengths[int(0.95 * len(lengths))]
        mx = lengths[-1]
        issues.append(
            f"INFO: token lengths p50={lengths[len(lengths)//2]}, p95={p95}, max={mx} "
            f"(max_seq_length={max_seq_length})"
        )
        if over:
            pct = 100 * len(over) / len(train)
            issues.append(
                f"WARN: {len(over)}/{len(train)} ({pct:.1f}%) train rows exceed max_seq_length={max_seq_length}"
            )
            if pct > 5:
                issues.append("FIX: increase MAX_SEQ_LENGTH to 2048 on H100/A100")

    # inference prompt should differ from training (generation prompt only)
    sample = train[0]
    infer = format_for_model(tokenizer, sample["note"], tokenize=False)
    train_text = format_training_text(tokenizer, sample["note"], sample["icd10"], sample["cpt"])
    if infer == train_text:
        issues.append("CRITICAL: inference prompt equals full training text — chat template bug")
    elif infer not in train_text and not train_text.startswith(infer[:100]):
        issues.append("INFO: inference prompt is prefix of training format (expected for chat SFT)")

    return issues


def check_notebook_gotchas() -> list[str]:
    issues: list[str] = []
    nb = (ROOT / "notebooks" / "train_qlora.ipynb").read_text(encoding="utf-8")
    if PLACEHOLDER_HUB in nb and "raise ValueError" not in nb:
        issues.append(f"BLOCKER: placeholder HUB_MODEL_ID without runtime validation")
    if "git clone https://github.com/namitrathod/clinical-code-extractor" in nb:
        if "git pull" not in nb:
            issues.append("BLOCKER: notebook clones GitHub but never git pull — push code first")
        else:
            issues.append(
                "INFO: notebook clones/pulls GitHub — push ALL local changes before Colab run"
            )
    if "H100" not in nb and "get_device_name" not in nb:
        issues.append("WARN: no H100/A100 auto-detect in notebook")
    if "formatting_func(examples)" not in nb and "isinstance(examples" not in nb:
        issues.append("WARN: formatting_func may not handle batched rows")
    if "trainer.model" not in nb and "PeftModel.from_pretrained" in nb:
        issues.append("WARN: smoke test reloads base model — may OOM after training")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat WARN as failure")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    sections = [
        ("Files", check_files()),
        ("Raw data", check_raw_data()),
        ("SFT format", check_sft_data()),
        ("Python packages / TRL API", check_imports()),
        ("Tokenizer lengths", check_tokenizer_offline(args.max_seq_length)),
        ("Notebook gotchas", check_notebook_gotchas()),
    ]

    blockers: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    print("=" * 60)
    print("QLoRA PRE-FLIGHT CHECK")
    print("=" * 60)
    for title, items in sections:
        print(f"\n[{title}]")
        if not items:
            print("  OK")
            continue
        for msg in items:
            if msg.startswith("BLOCKER") or msg.startswith("CRITICAL") or msg.startswith("MISSING"):
                blockers.append(msg)
                print(f"  FAIL: {msg}")
            elif msg.startswith("WARN") or msg.startswith("FIX") or msg.startswith("SKIP"):
                warnings.append(msg)
                print(f"  WARN: {msg}")
            else:
                info.append(msg)
                print(f"  OK:   {msg}")

    print("\n" + "=" * 60)
    print(f"Blockers: {len(blockers)} | Warnings: {len(warnings)} | Info: {len(info)}")
    if blockers:
        print("\nDO NOT START TRAINING until blockers are fixed.")
        sys.exit(1)
    if args.strict and warnings:
        print("\nStrict mode: warnings treated as failure.")
        sys.exit(1)
    if warnings:
        print("\nWarnings present — review before H100 run.")
    else:
        print("\nAll clear — safe to train.")
    sys.exit(0)


if __name__ == "__main__":
    main()
