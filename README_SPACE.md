---
title: Clinical Code Extractor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
python_version: 3.12
pinned: false
---

# Clinical Code Extractor

Paste a clinical note → get **ICD-10** and **CPT** billing codes.

Fine-tuned Gemma 4 E4B (QLoRA) · [Adapter on HF Hub](https://huggingface.co/namitrathod/gemma4-icd-cpt-qlora)

> **Note:** Free CPU hardware — first request may take **2–5 minutes** while the 4-bit model loads. Not for clinical use.
