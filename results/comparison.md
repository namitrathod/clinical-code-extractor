# Baseline vs fine-tuned comparison

Test set: 500 held-out notes | Model: `google/gemma-4-E4B-it` + QLoRA adapter `namitrathod/gemma4-icd-cpt-qlora`

| Metric | Baseline | Fine-tuned | Delta |
|--------|----------|------------|-------|
| ICD-10 micro F1 | 0.4846 | 0.9370 | +0.4524 |
| ICD-10 macro F1 | 0.4196 | 0.9754 | +0.5558 |
| CPT micro F1 | 0.1756 | 0.5838 | +0.4082 |
| CPT macro F1 | 0.0327 | 0.4366 | +0.4039 |
| Micro F1 avg | 0.3301 | 0.7604 | +0.4303 |
| Exact set match (ICD) | 0.292 | 0.848 | +0.556 |
| Exact set match (CPT) | 0.122 | 0.364 | +0.242 |
| Top-1 primary ICD recall | 0.384 | 1.000 | +0.616 |
| JSON validity rate | 1.000 | 1.000 | 0.000 |
| Invalid ICD rate | 0.610 | 0.000 | -0.610 |
| Invalid CPT rate | 0.156 | 0.000 | -0.156 |

## Interpretation

- **ICD-10**: QLoRA dramatically improved code accuracy — primary diagnosis recall reached 100% and exact set match rose to 85%.
- **CPT**: Improved from 18% to 58% micro F1, but visit-level CPT codes (99202 vs 99203 vs 99204) remain the hardest category.
- **Format**: Both runs output valid JSON; fine-tuning eliminated invalid/hallucinated codes (61% → 0% invalid ICD rate on baseline).

## Artifacts

- Baseline: `results/baseline.json`, `predictions_baseline.jsonl`
- Fine-tuned: `results/finetuned.json`, `predictions_finetuned.jsonl`
- Adapter: https://huggingface.co/namitrathod/gemma4-icd-cpt-qlora
