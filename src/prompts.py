"""Prompt templates for clinical coding — single source of truth.

The *content* inside the user message must stay identical across:
  - baseline inference
  - QLoRA training data prep
  - fine-tuned inference
  - Gradio demo

For Gemma instruction models, wrap with format_for_model() so chat tokens
(<start_of_turn> etc.) are applied consistently at inference time.
"""

from __future__ import annotations

from typing import Any

USER_PROMPT_TEMPLATE = """### Clinical note:
{note}

### Task
Return billing codes as one JSON object with exactly these keys:
- "icd10": list of ICD-10-CM code strings
- "cpt": list of CPT code strings
No markdown, no explanations, no other keys.

### Billing codes (JSON only):
"""


def build_user_content(note: str) -> str:
    """Instruction text shown to the model (without chat wrapper)."""
    return USER_PROMPT_TEMPLATE.format(note=note.strip())


def build_prompt(note: str) -> str:
    """Plain prompt prefix used in SFT (prompt + JSON completion concatenated)."""
    return build_user_content(note)


def build_completion(icd10: list[str], cpt: list[str]) -> str:
    """Gold JSON completion appended during training."""
    import json

    return json.dumps({"icd10": icd10, "cpt": cpt}, separators=(",", ": "))


def build_training_messages(note: str, icd10: list[str], cpt: list[str]) -> list[dict[str, str]]:
    """Chat messages for SFT (user prompt + assistant JSON completion)."""
    return [
        {"role": "user", "content": build_user_content(note)},
        {"role": "assistant", "content": build_completion(icd10, cpt)},
    ]


def format_training_text(tokenizer: Any, note: str, icd10: list[str], cpt: list[str]) -> str:
    """Full chat-formatted training example (used by SFTTrainer formatting_func)."""
    messages = build_training_messages(note, icd10, cpt)
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_for_model(tokenizer: Any, note: str, *, tokenize: bool = False) -> str | list[int]:
    """Apply the model's chat template (required for Gemma 4 instruction models)."""
    messages = [{"role": "user", "content": build_user_content(note)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
    )
