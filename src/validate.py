"""Parse model JSON output and filter billing codes against whitelists."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


def load_whitelists(codes_dir: Path | None = None) -> tuple[dict[str, str], dict[str, str]]:
    root = codes_dir or Path(__file__).resolve().parent.parent / "codes"
    icd10 = json.loads((root / "icd10_whitelist.json").read_text(encoding="utf-8"))
    cpt = json.loads((root / "cpt_whitelist.json").read_text(encoding="utf-8"))
    return icd10, cpt


def _normalize_code_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        code = str(item).strip()
        if code:
            out.append(code)
    return out


def parse_model_output(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from generated text."""
    if not text or not text.strip():
        return None

    # Try full string first (clean outputs)
    for candidate in [text.strip(), *JSON_OBJECT_PATTERN.findall(text)]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def filter_codes(
    parsed: dict[str, Any] | None,
    icd_whitelist: dict[str, str],
    cpt_whitelist: dict[str, str],
) -> dict[str, Any]:
    """Filter to whitelisted codes; return pre- and post-filter lists."""
    if parsed is None:
        return {
            "json_valid": False,
            "icd10_pre": [],
            "cpt_pre": [],
            "icd10": [],
            "cpt": [],
            "invalid_icd10": [],
            "invalid_cpt": [],
        }

    icd_pre = _normalize_code_list(parsed.get("icd10"))
    cpt_pre = _normalize_code_list(parsed.get("cpt"))

    icd_post: list[str] = []
    cpt_post: list[str] = []
    invalid_icd: list[str] = []
    invalid_cpt: list[str] = []

    for code in icd_pre:
        if code in icd_whitelist:
            if code not in icd_post:
                icd_post.append(code)
        else:
            invalid_icd.append(code)

    for code in cpt_pre:
        if code in cpt_whitelist:
            if code not in cpt_post:
                cpt_post.append(code)
        else:
            invalid_cpt.append(code)

    return {
        "json_valid": True,
        "icd10_pre": icd_pre,
        "cpt_pre": cpt_pre,
        "icd10": icd_post,
        "cpt": cpt_post,
        "invalid_icd10": invalid_icd,
        "invalid_cpt": invalid_cpt,
    }
