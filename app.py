"""Gradio demo — paste a clinical note, get ICD-10 + CPT billing codes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr

from src.inference import extract_codes, get_extractor

# Hand-written examples — NOT from the training/test dataset
EXAMPLES = [
    [
        "Follow-up visit. 58yo M with type 2 diabetes returns for routine management. "
        "Reports improved glucose control on metformin. Exam: A1c 7.1%, BP 128/82. "
        "Assessment: Type 2 diabetes mellitus without complications. "
        "Plan: continue metformin, recheck A1c in 3 months."
    ],
    [
        "UC visit — 34yo F, 3 days sore throat and cough. Focused exam: erythematous "
        "oropharynx, no wheezing, afebrile. Assessment: Acute upper respiratory infection. "
        "Plan: supportive care, return if worsening."
    ],
    [
        "Annual wellness visit. 45yo F, no acute complaints. Exam unremarkable, vitals "
        "stable. Assessment: Encounter for general adult medical examination without "
        "abnormal findings. Plan: preventive counseling, flu vaccine administered."
    ],
]


def run_extraction(note: str) -> tuple[list[list[str]], list[list[str]], str]:
    if not note or not note.strip():
        return [], [], "Please paste a clinical note."

    try:
        result = extract_codes(note)
    except Exception as exc:
        return [], [], f"Error: {exc}"

    icd_rows = [
        [code, result["descriptions"]["icd10"].get(code, "")]
        for code in result["icd10"]
    ]
    cpt_rows = [
        [code, result["descriptions"]["cpt"].get(code, "")]
        for code in result["cpt"]
    ]

    status = "Valid JSON" if result["json_valid"] else "Could not parse JSON — check raw output"
    if not icd_rows and not cpt_rows:
        status = "No whitelisted codes extracted. " + status

    raw_preview = result["raw_output"][:500]
    footer = f"\n\n---\n{status}\n\nRaw model output:\n{raw_preview}"
    return icd_rows, cpt_rows, footer


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Clinical Code Extractor") as demo:
        gr.Markdown(
            "# Clinical Code Extractor\n"
            "Paste a clinical note → get **ICD-10** and **CPT** billing codes.\n\n"
            "Fine-tuned Gemma 4 E4B (QLoRA) · codes filtered to a curated whitelist · "
            "**Not for clinical use.**"
        )
        with gr.Row():
            note_input = gr.Textbox(
                label="Clinical note",
                lines=10,
                placeholder="Paste a doctor's note here...",
            )
        extract_btn = gr.Button("Extract codes", variant="primary")
        with gr.Row():
            icd_table = gr.Dataframe(
                headers=["ICD-10", "Description"],
                label="ICD-10 codes",
                interactive=False,
            )
            cpt_table = gr.Dataframe(
                headers=["CPT", "Description"],
                label="CPT codes",
                interactive=False,
            )
        details = gr.Textbox(label="Details", lines=6, interactive=False)

        gr.Examples(examples=EXAMPLES, inputs=note_input, label="Try an example")

        extract_btn.click(
            fn=run_extraction,
            inputs=note_input,
            outputs=[icd_table, cpt_table, details],
        )
        note_input.submit(
            fn=run_extraction,
            inputs=note_input,
            outputs=[icd_table, cpt_table, details],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    print("Loading model (first request may take 1–2 min on CPU)...")
    get_extractor().load()
    print("Model ready.")
    demo.launch()
