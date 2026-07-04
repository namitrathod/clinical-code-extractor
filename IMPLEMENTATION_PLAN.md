# ICD-10 / CPT Code Extractor — Full Implementation Plan

**Project:** Fine-tune Gemma 4 E4B with QLoRA to map free-text clinical notes to ICD-10 and CPT billing codes.

**Target pitch (fill in real numbers after eval):**

> *"Fine-tuned Gemma 4 E4B with QLoRA maps clinical notes to ICD-10/CPT codes; macro-F1 improved from **X → Y** on a held-out synthetic eval set vs the base model (top-3 recall **Z**)."*

| Constraint | Value |
|------------|-------|
| Budget | ~$0 (Colab credits: ~15–25 of 290 available) |
| Timeline | ~3 weeks of focused evenings (~40–50 hours) |
| Base model | `google/gemma-4-E4B-it` (fallback: `microsoft/Phi-3-mini-4k-instruct`) |
| Training method | QLoRA (4-bit + PEFT) |
| GPU | Colab A100 (training) / T4 (inference); Kaggle free T4 as fallback |
| Hosting | HuggingFace Hub (adapter) + HuggingFace Spaces (demo) |
| Data | Synthetic only (public ICD/CPT tables) |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Success Criteria](#2-success-criteria)
3. [Scope (v1)](#3-scope-v1)
4. [Architecture](#4-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Data Strategy](#6-data-strategy)
7. [Training Plan](#7-training-plan)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Inference & Validation](#9-inference--validation)
10. [Deployment](#10-deployment)
11. [Week-by-Week Timeline](#11-week-by-week-timeline)
12. [Task Checklist](#12-task-checklist)
13. [Design Decisions](#13-design-decisions)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Phase 2 — Pipeline Extension](#15-phase-2--pipeline-extension)
16. [README Outline](#16-readme-outline)
17. [Dependencies](#17-dependencies)

---

## 1. Problem Statement

Hospitals must translate free-text clinical documentation into standardized billing codes:

- **ICD-10-CM** — diagnosis codes (70,000+ codes)
- **CPT** — procedure / visit codes (10,000+ codes)

Insurance reimbursement depends on accurate coding. Human medical coders are expensive (~$50k/year), slow, and error-prone. Coding errors cause denied claims and lost revenue.

**This project automates:** clinical note in → structured ICD-10 + CPT codes out.

**Commercial context:** Revenue cycle management (RCM) is a large health-tech market. Every US health system has this problem.

---

## 2. Success Criteria

### Minimum bar (ship v1)

| Criterion | Target |
|-----------|--------|
| Macro-F1 (ICD-10) — fine-tuned | ≥ 0.55 |
| Absolute F1 improvement vs baseline | ≥ 0.20 |
| JSON validity rate (no retry) | ≥ 95% |
| Invalid code rate after whitelist filter | 0% |
| Live demo | Works on 3 hand-written notes not in training set |
| Artifacts published | HF adapter, eval results, Gradio Space |

### Hire-ready deliverables

- [ ] Held-out test eval: baseline vs fine-tuned (same prompt, same test set)
- [ ] `results/comparison.md` with metric table + 1 chart
- [ ] 10 failure-case error analysis in README
- [ ] One-sentence metric pitch with real numbers
- [ ] Public HuggingFace Space demo

---

## 3. Scope (v1)

### In scope

- Synthetic clinical notes paired with ICD-10 + CPT labels
- QLoRA fine-tuning of Gemma 4 E4B
- Baseline evaluation (no fine-tuning)
- Code whitelist validation at inference
- Gradio demo on HuggingFace Spaces

### Out of scope (v1 — do not build yet)

- SOAP note generator (Phase 2)
- FHIR bundle output (Phase 2)
- NeMo Guardrails / PHI handling (Phase 2)
- Real patient data / HIPAA compliance
- Full 70k ICD code space

### Scope limits (critical for 3-week timeline)

| Dimension | v1 limit |
|-----------|----------|
| ICD-10 codes | Top **50–100** by frequency in synthetic generator |
| CPT codes | **20–30** common E/M + procedure codes |
| Note length | 100–400 words |
| Training notes | 3,000–5,000 |
| Validation notes | 500 |
| Test notes | 500 (held out, never used for tuning) |
| Train / val / test split | 70 / 15 / 15, stratified by primary ICD |

---

## 4. Architecture

```
┌─────────────────────┐
│  Public Code Tables │
│  (ICD-10 + CPT CSV) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Synthetic Data     │     │  Instruction Format │
│  Generator          │────▶│  (JSONL for train)  │
│  (templates + seeds)│     └──────────┬──────────┘
└─────────────────────┘                │
                                       ▼
                            ┌─────────────────────┐
                            │  Baseline Eval      │
                            │  (base Gemma 4 E4B) │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  QLoRA Fine-tune    │
                            │  (Kaggle T4)        │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Fine-tuned Eval    │
                            │  (same test set)    │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Inference +        │
                            │  Whitelist Validate │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  HF Space Demo      │
                            │  (Gradio)           │
                            └─────────────────────┘
```

### Inference flow

1. Accept raw clinical note text
2. Format with fixed prompt template
3. Generate model output (JSON)
4. Parse JSON; retry once on failure
5. Filter codes against whitelist
6. Return `{ icd10: [...], cpt: [...] }` with optional code descriptions

---

## 5. Repository Structure

```
finetune/
├── IMPLEMENTATION_PLAN.md      # This file
├── README.md                   # Portfolio-facing documentation
├── requirements.txt
├── .gitignore
│
├── codes/
│   ├── icd10_full.csv          # Raw download (optional, gitignored if large)
│   ├── cpt_full.csv
│   ├── icd10_whitelist.json    # Curated v1 code subset + descriptions
│   └── cpt_whitelist.json
│
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl              # Consider gitignoring; document how to regenerate
│
├── scripts/
│   ├── build_code_tables.py    # Download/parse ICD+CPT → whitelist JSON
│   ├── generate_synthetic_data.py
│   ├── prepare_training_data.py
│   ├── run_baseline.py
│   └── eval.py                 # Shared eval for baseline + fine-tuned
│
├── src/
│   ├── __init__.py
│   ├── prompts.py              # Prompt templates (train == inference)
│   ├── inference.py            # Load model, generate, parse
│   └── validate.py             # Whitelist filter, dedupe, format check
│
├── notebooks/
│   └── train_qlora.ipynb       # Kaggle training notebook
│
├── results/
│   ├── baseline.json
│   ├── finetuned.json
│   ├── comparison.md
│   └── failure_cases.json      # 10 examples for error analysis
│
└── app.py                      # HuggingFace Spaces Gradio demo
```

---

## 6. Data Strategy

### 6.1 Code table sources

| Source | Use |
|--------|-----|
| CMS ICD-10-CM files | Official diagnosis codes + descriptions |
| AMA CPT (or curated open subset) | Procedure codes — use a small open mirror for v1 |

**Script:** `scripts/build_code_tables.py`

**Output:** `codes/icd10_whitelist.json`, `codes/cpt_whitelist.json`

```json
{
  "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
  "I10": "Essential (primary) hypertension"
}
```

### 6.2 Synthetic note generation

**Principle:** Quality > quantity. Notes must look like real chart text, not `"Patient has E11.65."`

**Per-code generation:**
1. Seed from official ICD/CPT short description
2. Apply 3–5 note templates (brief visit, detailed H&P, follow-up, urgent care)
3. Inject variability: age, sex, abbreviations, comorbidities, irrelevant social history
4. Assign CPT based on visit complexity keywords (not random)

**Multi-label:** ~30% of notes include 2 ICD codes (primary + secondary).

**Example record (`data/train.jsonl`):**

```json
{
  "id": "note_0042",
  "note": "67yo M c/o polyuria and polydipsia x 3 weeks. PMH: HTN. Exam: BMI 31, fasting glucose 248. Assessment: T2DM with hyperglycemia, uncontrolled. Plan: start metformin, diabetes education, recheck A1c in 3 months.",
  "icd10": ["E11.65", "I10"],
  "cpt": ["99214"],
  "primary_icd10": "E11.65",
  "template_id": "follow_up_02"
}
```

**Script:** `scripts/generate_synthetic_data.py`

**Targets:**

| Split | Count | Purpose |
|-------|-------|---------|
| train | 3,000–5,000 | Fine-tuning |
| val | 500 | Early stopping / hyperparameter checks |
| test | 500 | Final metrics only — never tune on this |

**Split strategy:** Stratify by `primary_icd10`. Ensure each code has representation in all splits.

### 6.3 Training format (instruction tuning)

**Script:** `scripts/prepare_training_data.py`

Convert each record to:

```
### Clinical note:
{note}

### Billing codes (JSON only):
{"icd10": ["E11.65", "I10"], "cpt": ["99214"]}
```

**Rules:**
- Identical prompt structure for training, baseline eval, and fine-tuned inference
- No shuffle of sections between stages
- Store as JSONL with `text` field (full prompt + completion) for SFT

### 6.4 Leakage prevention

- Hold out **entire template families** from test set where possible (not just random rows)
- Do not copy identical note text across splits
- Document seed + generator version in README

---

## 7. Training Plan

### 7.1 Environment

- **Platform:** Colab (A100 for training via compute credits; T4 for debugging/inference). Kaggle free T4 as fallback.
- **Notebook:** `notebooks/train_qlora.ipynb`
- **Upload:** Training JSONL via Drive mount or HF dataset

### 7.2 Hyperparameters (starting point)

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-4-E4B-it` |
| Quantization | 4-bit (bitsandbytes) |
| LoRA rank (r) | 16–32 |
| LoRA alpha | 32–64 |
| LoRA target modules | `all-linear` (recommended for Gemma family) |
| Epochs | 2–3 |
| Learning rate | 2e-4 |
| Batch size | 8 + grad accum 2 on A100 (4 + accum 4 on T4) |
| Max sequence length | 2048 on A100 (1024 on T4) |
| Optimizer | AdamW |
| Warmup | 5% of steps |

**Gemma-specific note:** use the tokenizer's chat template (`tokenizer.apply_chat_template`) to wrap the prompt — Gemma instruction models expect `<start_of_turn>` formatting. Keep the note/JSON content identical to `src/prompts.py`; only the wrapper differs. Follow the official Gemma 4 TRL fine-tuning recipe.

### 7.3 Training steps

1. Load base model in 4-bit
2. Attach LoRA adapter via PEFT
3. Train on `data/train.jsonl` (instruction format)
4. Validate loss on `data/val.jsonl` each epoch
5. Save adapter checkpoints to HuggingFace Hub each epoch
6. Select best checkpoint by val loss (not test)

### 7.4 HF Hub artifacts

| Artifact | Path |
|----------|------|
| LoRA adapter | `your-username/gemma4-icd-cpt-qlora` |
| Optional dataset | `your-username/clinical-coding-synthetic-v1` |

---

## 8. Evaluation Protocol

**Script:** `scripts/eval.py` (shared by baseline and fine-tuned)

### 8.1 Metrics

| Metric | Description |
|--------|-------------|
| **Macro-F1 (ICD)** | Per-code F1 averaged across codes |
| **Macro-F1 (CPT)** | Same for procedure codes |
| **Micro-F1** | Global TP/FP/FN across all predictions |
| **Exact-set match** | Predicted code set equals gold set |
| **Top-1 recall (ICD)** | Primary ICD correct |
| **Top-3 recall (ICD)** | Gold primary ICD in top 3 predictions |
| **JSON validity rate** | Parses without retry |
| **Invalid code rate (pre-filter)** | Hallucinated codes before whitelist |
| **Invalid code rate (post-filter)** | Should be 0% |

### 8.2 Evaluation rules

1. Run **baseline first** on test set — save `results/baseline.json`
2. Run **fine-tuned** on same test set — save `results/finetuned.json`
3. Same prompt, same temperature, same max tokens for both
4. Never tune hyperparameters using test set
5. Save 10 worst failures to `results/failure_cases.json`

### 8.3 Output format (`results/baseline.json`)

```json
{
  "model": "google/gemma-4-E4B-it",
  "split": "test",
  "n_samples": 500,
  "metrics": {
    "icd10_macro_f1": 0.31,
    "cpt_macro_f1": 0.28,
    "micro_f1": 0.35,
    "exact_set_match": 0.12,
    "top1_recall": 0.45,
    "top3_recall": 0.72,
    "json_validity": 0.88
  }
}
```

### 8.4 Comparison doc (`results/comparison.md`)

Include:
- Side-by-side metric table
- Bar chart: baseline vs fine-tuned (ICD macro-F1, CPT macro-F1)
- 3–5 sentence interpretation
- Link to failure cases

---

## 9. Inference & Validation

### 9.1 `src/prompts.py`

Single source of truth for prompt template. Used by train prep, eval, and demo.

### 9.2 `src/inference.py`

```python
# Pseudocode
def extract_codes(note: str, model, tokenizer) -> dict:
    prompt = build_prompt(note)
    output = generate(model, tokenizer, prompt)
    parsed = parse_json(output)
    if parsed is None:
        output = generate(model, tokenizer, prompt)  # one retry
        parsed = parse_json(output)
    return validate_codes(parsed)
```

### 9.3 `src/validate.py`

- Parse JSON; extract `icd10` and `cpt` lists
- Filter against whitelist
- Dedupe
- Reject malformed code strings
- Return empty lists on total failure (don't crash demo)

---

## 10. Deployment

### 10.1 HuggingFace Space (Gradio)

**File:** `app.py`

**UI elements:**
- Text area: paste clinical note
- Button: Extract codes
- Output: ICD-10 + CPT tables with descriptions
- 3 preset examples (diabetes, URI, wellness visit)

**Space `README.md` badge + link in main README**

### 10.2 Space requirements

```
requirements.txt (Space):
torch
transformers
peft
bitsandbytes
accelerate
gradio
```

Load base Gemma 4 E4B + LoRA adapter from HF Hub at startup. If CPU-tier latency is unacceptable, consider a Gemma 4 E2B adapter for the demo.

---

## 11. Week-by-Week Timeline

### Week 1 — Data + Baseline

| Day | Task | Output |
|-----|------|--------|
| 1 | Set up repo, requirements, .gitignore | Scaffold |
| 2 | `build_code_tables.py` — curate 50–100 ICD, 20–30 CPT | `codes/*.json` |
| 3–4 | `generate_synthetic_data.py` — templates + variability | `data/*.jsonl` |
| 5 | `prepare_training_data.py` — instruction format | Training JSONL |
| 6 | `eval.py` + `run_baseline.py` | `results/baseline.json` |
| 7 | Buffer / fix data issues / spot-check notes | Clean test set |

### Week 2 — Fine-tune + Eval

| Day | Task | Output |
|-----|------|--------|
| 1–2 | `train_qlora.ipynb` — first training run | Adapter v1 on HF |
| 3 | Debug training (loss, overfit, prompt mismatch) | Adapter v2 if needed |
| 4 | Run fine-tuned eval on test set | `results/finetuned.json` |
| 5 | Write `comparison.md`, failure analysis | Results package |
| 6–7 | `inference.py` + `validate.py` | Local inference works |

### Week 3 — Demo + Portfolio

| Day | Task | Output |
|-----|------|--------|
| 1–2 | `app.py` Gradio demo | Local demo |
| 3 | Deploy HF Space | Public URL |
| 4 | Write README (metrics, limitations, architecture) | Portfolio README |
| 5 | Hand-test 3 novel notes | Demo validated |
| 6 | LinkedIn post draft (optional) | Shareable pitch |
| 7 | Buffer / polish / record short screen capture | Final ship |

---

## 12. Task Checklist

### Phase A — Foundation
- [ ] Initialize repo structure
- [ ] Add `requirements.txt` and `.gitignore`
- [ ] Download and parse ICD-10 / CPT tables
- [ ] Build whitelist JSON files

### Phase B — Data
- [ ] Implement note templates (≥ 4 styles)
- [ ] Implement multi-label assignment logic
- [ ] Generate train / val / test splits
- [ ] Manual review: 20 random notes for realism

### Phase C — Baseline
- [ ] Implement `prompts.py`
- [ ] Implement `eval.py`
- [ ] Run baseline on test set
- [ ] Record `results/baseline.json`

### Phase D — Training
- [ ] Prepare instruction-tuning JSONL
- [ ] Create Kaggle notebook
- [ ] Train QLoRA adapter
- [ ] Upload adapter to HF Hub

### Phase E — Fine-tuned Eval
- [ ] Run eval on test set with adapter
- [ ] Record `results/finetuned.json`
- [ ] Write `comparison.md`
- [ ] Document 10 failure cases

### Phase F — Ship
- [ ] Build Gradio `app.py`
- [ ] Deploy HuggingFace Space
- [ ] Write README with metric sentence
- [ ] Verify demo on unseen notes

---

## 13. Design Decisions

### Why Gemma 4 E4B?
- Current-generation model (April 2026) — signals up-to-date skills
- QLoRA fine-tunes in ~10–14 GB VRAM — works on free T4, fast on A100
- Apache 2.0 license — clean for a commercial revenue-cycle pitch
- Strong instruction-following for JSON output
- Fallback: Phi-3 Mini (mature recipes) if the Gemma recipe fights back — pipeline is model-agnostic

### Why QLoRA?
- Fits in 16GB T4 VRAM
- Standard industry approach — interviewers know it
- Adapter-only upload keeps HF artifacts small

### Why synthetic data?
- $0 cost, no HIPAA risk
- Full control over labels (clean supervision signal)
- Reproducible — anyone can regenerate with your scripts

### Why whitelist validation?
- LLMs hallucinate invalid codes
- Post-filter is standard in production coding systems
- Report pre-filter invalid rate to show the problem; post-filter for production metrics

### Why narrow code set?
- 70k codes = impossible to evaluate well in 3 weeks
- 50–100 codes = learnable, measurable, still impressive if metrics are honest

### CPT assignment in synthetic data
- Tie CPT to visit complexity keywords (`comprehensive exam`, `30 min visit`) — not random
- Gives the model a learnable signal

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Kaggle GPU timeout | Medium | High | Save checkpoints each epoch to HF Hub |
| JSON parse failures | High | Medium | Retry once; track JSON validity separately |
| Metrics look "too good" (leakage) | Medium | High | Template-level holdout; manual spot checks |
| CPT harder than ICD | High | Medium | Report separately; consider ICD-only fallback |
| Scope creep (SOAP, FHIR) | High | High | Strict v1 boundary — Phase 2 only after ship |
| Hallucinated codes | High | Low | Whitelist filter always on |
| Interview skepticism on synthetic data | Medium | Medium | README: limitations + path to real data / human-in-the-loop |

---

## 15. Phase 2 — Pipeline Extension

**Do not start until v1 metrics are published.**

| Step | Addition |
|------|----------|
| 2a | SOAP note generator (second LoRA adapter) |
| 2b | FastAPI pipeline: raw note → SOAP → codes |
| 2c | End-to-end metric (raw note through full pipeline) |
| 2d | FHIR R4 Bundle output (`DocumentReference` + `Claim`) |
| 2e | NeMo Guardrails on API boundary |
| 2f | Expand code vocabulary to 500+ ICD |

**End-to-end pitch (Phase 2):**

> *"Built an end-to-end clinical AI pipeline: raw doctor notes → structured SOAP → ICD-10/CPT codes → FHIR R4 bundle, with 0.68 coding F1 end-to-end vs 0.31 baseline."*

---

## 16. README Outline

When v1 ships, README should include:

1. **One-sentence pitch** (with real metrics)
2. **Problem** — why coding matters (2 paragraphs)
3. **Approach** — Gemma 4 E4B + QLoRA + synthetic data
4. **Results** — table + chart (baseline vs fine-tuned)
5. **Demo** — link to HF Space
6. **How to reproduce** — data gen → train → eval commands
7. **Error analysis** — 3 failure examples with explanation
8. **Limitations** — synthetic only, narrow code set, not clinical advice
9. **Phase 2 roadmap** — SOAP + FHIR pipeline

---

## 17. Dependencies

### `requirements.txt`

```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.28.0
datasets>=2.18.0
trl>=0.8.0
scikit-learn>=1.4.0
pandas>=2.0.0
tqdm>=4.66.0
gradio>=4.0.0
huggingface_hub>=0.22.0
```

### Accounts needed

- [ ] HuggingFace account (model + Space hosting)
- [ ] Kaggle account (free GPU)
- [ ] GitHub account (portfolio repo)

---

## Quick Reference — Commands (to implement)

```bash
# 1. Build code whitelists
python scripts/build_code_tables.py

# 2. Generate synthetic dataset
python scripts/generate_synthetic_data.py --train 4000 --val 500 --test 500

# 3. Prepare instruction-tuning format
python scripts/prepare_training_data.py

# 4. Run baseline evaluation
python scripts/run_baseline.py --split test --output results/baseline.json

# 5. Run fine-tuned evaluation (after training)
python scripts/eval.py --model your-username/gemma4-icd-cpt-qlora --split test --output results/finetuned.json

# 6. Launch local demo
python app.py
```

---

*Last updated: project kickoff. Update metric placeholders in Section 2 after eval is complete.*
