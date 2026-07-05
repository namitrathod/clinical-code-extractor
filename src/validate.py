"""Parse model JSON output and filter billing codes against whitelists."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ICD_KEYS = ("icd10", "diagnosis_codes", "diagnosis_code", "icd_codes", "diagnosis")
CPT_KEYS = ("cpt", "procedure_codes", "procedure_code", "cpt_codes", "procedures")
BILLING_KEYS = ICD_KEYS + CPT_KEYS


def load_whitelists(codes_dir: Path | None = None) -> tuple[dict[str, str], dict[str, str]]:
    root = codes_dir or Path(__file__).resolve().parent.parent / "codes"
    icd10 = json.loads((root / "icd10_whitelist.json").read_text(encoding="utf-8"))
    cpt = json.loads((root / "cpt_whitelist.json").read_text(encoding="utf-8"))
    return icd10, cpt


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_objects(text: str) -> list[str]:
    """Return balanced {...} substrings, largest first."""
    objects: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        in_string = False
        escape = False
        for j in range(i, len(text)):
            char = text[j]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : j + 1])
                    i = j + 1
                    break
        else:
            i += 1
    objects.sort(key=len, reverse=True)
    return objects


def _billing_score(parsed: dict[str, Any]) -> int:
    score = sum(10 for key in BILLING_KEYS if key in parsed)
    if "code" in parsed:
        score += 1
    return score


def _extract_codes_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        code = value.strip()
        return [code] if code else []
    if isinstance(value, dict):
        if "code" in value:
            return _extract_codes_from_value(value["code"])
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_extract_codes_from_value(item))
        return out
    return []


def canonicalize_parsed_output(parsed: dict[str, Any]) -> dict[str, list[str]]:
    """Map alternate model schemas to icd10/cpt string lists."""
    icd10: list[str] = []
    cpt: list[str] = []

    for key in ICD_KEYS:
        if key in parsed:
            icd10.extend(_extract_codes_from_value(parsed[key]))
    for key in CPT_KEYS:
        if key in parsed:
            cpt.extend(_extract_codes_from_value(parsed[key]))

    if not icd10 and not cpt and "code" in parsed:
        code = str(parsed["code"]).strip()
        if re.match(r"^[A-TV-Z]\d", code, re.IGNORECASE):
            icd10 = [code]
        elif re.search(r"\d{4,5}", code):
            cpt = [code]

    return {"icd10": icd10, "cpt": cpt}


def _normalize_icd_for_whitelist(code: str, whitelist: dict[str, str]) -> str:
    code = code.strip().upper()
    if code in whitelist:
        return code
    if f"{code}0" in whitelist:
        return f"{code}0"
    if "." in code:
        base, suffix = code.rsplit(".", 1)
        if len(suffix) == 1:
            candidate = f"{base}.{suffix}0"
            if candidate in whitelist:
                return candidate
    return code


def _normalize_cpt_code(code: str) -> str:
    code = code.strip()
    match = re.search(r"\d{5}", code)
    return match.group(0) if match else code


def _dedupe_preserve_order(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def parse_model_output(text: str) -> dict[str, Any] | None:
    """Extract the best JSON object from generated text."""
    if not text or not text.strip():
        return None

    cleaned = _strip_markdown_fences(text)
    candidate_texts = [cleaned, * _extract_json_objects(cleaned), *_extract_json_objects(text)]

    best: tuple[int, int, dict[str, Any]] | None = None
    seen_texts: set[str] = set()
    for candidate_text in candidate_texts:
        if not candidate_text or candidate_text in seen_texts:
            continue
        seen_texts.add(candidate_text)
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        score = (_billing_score(parsed), len(candidate_text))
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], parsed)

    return best[2] if best else None


def filter_codes(
    parsed: dict[str, Any] | None,
    icd_whitelist: dict[str, str],
    cpt_whitelist: dict[str, str],
) -> dict[str, Any]:
    """Filter to whitelisted codes; return pre- and post-filter lists."""
    empty = {
        "json_valid": False,
        "icd10_pre": [],
        "cpt_pre": [],
        "icd10": [],
        "cpt": [],
        "invalid_icd10": [],
        "invalid_cpt": [],
    }
    if parsed is None:
        return empty

    canonical = canonicalize_parsed_output(parsed)
    icd_pre = _dedupe_preserve_order(
        _normalize_icd_for_whitelist(code, icd_whitelist) for code in canonical["icd10"]
    )
    cpt_pre = _dedupe_preserve_order(_normalize_cpt_code(code) for code in canonical["cpt"])

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

    has_billing_signal = bool(icd_pre or cpt_pre or _billing_score(parsed) > 0)
    return {
        "json_valid": has_billing_signal,
        "icd10_pre": icd_pre,
        "cpt_pre": cpt_pre,
        "icd10": icd_post,
        "cpt": cpt_post,
        "invalid_icd10": invalid_icd,
        "invalid_cpt": invalid_cpt,
    }


def process_model_output(
    raw: str,
    icd_whitelist: dict[str, str],
    cpt_whitelist: dict[str, str],
) -> dict[str, Any]:
    """Parse raw model text and return filtered prediction fields."""
    parsed = parse_model_output(raw)
    result = filter_codes(parsed, icd_whitelist, cpt_whitelist)
    return {
        "raw_output": raw,
        "json_valid": result["json_valid"],
        "icd10_pre": result["icd10_pre"],
        "cpt_pre": result["cpt_pre"],
        "icd10": result["icd10"],
        "cpt": result["cpt"],
        "invalid_icd10": result["invalid_icd10"],
        "invalid_cpt": result["invalid_cpt"],
    }
