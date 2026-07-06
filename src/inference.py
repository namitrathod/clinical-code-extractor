"""Load fine-tuned model and extract billing codes from clinical notes."""

from __future__ import annotations

import os
from typing import Any

from src.validate import load_whitelists, process_model_output

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_ADAPTER_ID = "namitrathod/gemma4-icd-cpt-qlora"


class CodeExtractor:
    """Lazy-loaded Gemma + LoRA wrapper for demo and CLI inference."""

    def __init__(
        self,
        model_id: str | None = None,
        adapter_id: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id or os.environ.get("CLINICAL_MODEL_ID", DEFAULT_MODEL_ID)
        self.adapter_id = adapter_id or os.environ.get("CLINICAL_ADAPTER_ID", DEFAULT_ADAPTER_ID)
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._icd_whitelist: dict[str, str] | None = None
        self._cpt_whitelist: dict[str, str] | None = None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from scripts.run_baseline import generate_one

        self._generate_one = generate_one

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant,
            device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base, self.adapter_id)
        self._model.eval()
        self._icd_whitelist, self._cpt_whitelist = load_whitelists()

    def extract_codes(self, note: str, *, retry: bool = True) -> dict[str, Any]:
        """Generate ICD-10/CPT codes for a clinical note."""
        note = (note or "").strip()
        if not note:
            return self._empty_result("Empty note.")

        self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._icd_whitelist is not None
        assert self._cpt_whitelist is not None

        raw = self._generate_one(self._model, self._tokenizer, note, self.max_new_tokens)
        result = process_model_output(raw, self._icd_whitelist, self._cpt_whitelist)

        if retry and not result["json_valid"]:
            raw = self._generate_one(self._model, self._tokenizer, note, self.max_new_tokens)
            result = process_model_output(raw, self._icd_whitelist, self._cpt_whitelist)

        return self._format_result(result)

    def _format_result(self, result: dict[str, Any]) -> dict[str, Any]:
        assert self._icd_whitelist is not None
        assert self._cpt_whitelist is not None

        icd10 = result["icd10"]
        cpt = result["cpt"]
        return {
            "icd10": icd10,
            "cpt": cpt,
            "json_valid": result["json_valid"],
            "raw_output": result["raw_output"],
            "descriptions": {
                "icd10": {code: self._icd_whitelist.get(code, "") for code in icd10},
                "cpt": {code: self._cpt_whitelist.get(code, "") for code in cpt},
            },
        }

    @staticmethod
    def _empty_result(message: str) -> dict[str, Any]:
        return {
            "icd10": [],
            "cpt": [],
            "json_valid": False,
            "raw_output": message,
            "descriptions": {"icd10": {}, "cpt": {}},
        }


_default_extractor: CodeExtractor | None = None


def get_extractor() -> CodeExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = CodeExtractor()
    return _default_extractor


def extract_codes(note: str) -> dict[str, Any]:
    """Convenience wrapper using the shared extractor instance."""
    return get_extractor().extract_codes(note)
