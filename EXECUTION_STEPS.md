# Execution Steps — Milestone-by-Milestone Guide

Companion to `IMPLEMENTATION_PLAN.md`. That file is the **spec** (what and why).
This file is the **runbook** (exact steps, in order, with done-checks).

**Rule:** Complete every step in a milestone before starting the next. Each milestone ends with a **Done check** — if it fails, fix before moving on.

---

## Milestone 1 — Repo Scaffold + Code Tables

**Time:** 1 evening (~2–3 hours) | **Needs:** Python 3.10+, nothing else

### Step 1.1 — Create the folder structure

```powershell
cd c:\Users\91898\Desktop\project\finetune
mkdir codes, data, scripts, src, notebooks, results
```

### Step 1.2 — Create `requirements.txt`

Copy the dependency list from `IMPLEMENTATION_PLAN.md` Section 17. Install only what Milestones 1–2 need for now:

```powershell
pip install pandas tqdm scikit-learn
```

(Install torch/transformers/peft later — they're only needed from Milestone 3.)

### Step 1.3 — Create `.gitignore`

```
__pycache__/
*.pyc
.venv/
venv/
codes/icd10_full.csv
codes/cpt_full.csv
*.ipynb_checkpoints
```

Keep `data/*.jsonl` tracked for now (small enough); gitignore later if it grows.

### Step 1.4 — Initialize git

```powershell
git init
git add .
git commit -m "Initial scaffold with implementation plan"
```

### Step 1.5 — Curate the ICD-10 whitelist (the real work)

Create `scripts/build_code_tables.py`. It should:

1. Define ~50 common ICD-10-CM codes **hardcoded or from a downloaded CMS CSV**.
   Recommended v1 approach: hardcode a curated dict — faster, zero download issues.
2. Cover 8–10 clinical categories so notes are varied:
   - **Endocrine:** E11.9, E11.65, E78.5, E03.9
   - **Cardiovascular:** I10, I25.10, I48.91, I50.9
   - **Respiratory:** J06.9, J45.909, J44.9, J20.9
   - **Musculoskeletal:** M54.5, M25.561, M17.11
   - **Mental health:** F41.1, F32.9, G47.00
   - **GI:** K21.9, K59.00
   - **GU/renal:** N39.0, N18.3
   - **Infectious:** A49.9, B34.9
   - **Wellness/misc:** Z00.00, Z23, R51.9, R05.9
   - ...expand each category to reach ~50 total
3. Write `codes/icd10_whitelist.json` as `{"code": "description"}`.

### Step 1.6 — Curate the CPT whitelist

Same script. ~20 codes:

- **Office visits (E/M):** 99202–99205 (new patient), 99212–99215 (established)
- **Preventive:** 99385, 99386, 99395, 99396
- **Common procedures:** 36415 (blood draw), 71046 (chest X-ray), 93000 (ECG), 81002 (urinalysis), 90471 (immunization admin), 96127 (mental health screen)

Write `codes/cpt_whitelist.json`.

### Step 1.7 — Run and verify

```powershell
python scripts/build_code_tables.py
```

### ✅ Done check (Milestone 1)

- [ ] `codes/icd10_whitelist.json` exists, ~50 entries, valid JSON
- [ ] `codes/cpt_whitelist.json` exists, ~20 entries, valid JSON
- [ ] Every code has a human-readable description
- [ ] Committed to git

---

## Milestone 2 — Synthetic Data Generator

**Time:** 1–2 evenings (~4–6 hours) | **Needs:** Milestone 1 done

### Step 2.1 — Design the condition profiles

Create `scripts/generate_synthetic_data.py`. First, define a **condition profile** per ICD code:

```python
CONDITION_PROFILES = {
    "E11.65": {
        "name": "Type 2 diabetes with hyperglycemia",
        "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred vision"],
        "exam_findings": ["BMI {bmi}", "fasting glucose {glucose} mg/dL", "A1c {a1c}%"],
        "plans": ["start metformin", "increase metformin dose", "add insulin glargine",
                  "diabetes education referral", "recheck A1c in 3 months"],
        "common_comorbidities": ["I10", "E78.5"],   # HTN, hyperlipidemia
        "typical_cpt": ["99213", "99214", "36415"],
        "vitals_ranges": {"glucose": (180, 350), "a1c": (7.5, 11.2), "bmi": (27, 38)},
    },
    # ... one profile per ICD code
}
```

This is the longest step. Budget 2–3 hours. Quality here decides whether the whole project is credible.

### Step 2.2 — Write 4–5 note templates

Each template is a different documentation style:

1. **`brief_visit`** — 3–4 sentences, heavy abbreviations ("67yo M c/o...")
2. **`detailed_hp`** — full History & Physical: CC, HPI, PMH, exam, assessment, plan
3. **`follow_up`** — references prior visit, medication adjustment
4. **`urgent_care`** — acute complaint, focused exam
5. **`wellness`** — annual physical, links to Z-codes and preventive CPT

Template rules:
- Insert **irrelevant noise** (social history, unrelated stable conditions) in ~40% of notes
- Randomize name/age/sex, vitals within profile ranges
- Mix abbreviation density (some notes say "hypertension", others "HTN")

### Step 2.3 — Implement label assignment

- Primary ICD = the profile the note was generated from
- ~30% of notes: add 1 secondary ICD from `common_comorbidities` **and mention it in the note text** (critical — label must be supported by text)
- CPT: pick from `typical_cpt` based on template (wellness template → preventive CPT; urgent care → 99203/99204; blood test mentioned → add 36415)

### Step 2.4 — Implement the split logic

- Generate ~5,000 notes total
- Split **70/15/15 stratified by primary ICD** so every code appears in train, val, and test
- **Template-level holdout:** for each condition, reserve one template variant that appears ONLY in the test set. Store `template_id` in each record so you can prove it.
- Fix the random seed (`random.seed(42)`) and print it — reproducibility is a resume claim

### Step 2.5 — Output format

Each line of `data/train.jsonl` / `val.jsonl` / `test.jsonl`:

```json
{"id": "note_00042", "note": "...", "icd10": ["E11.65", "I10"], "cpt": ["99214"], "primary_icd10": "E11.65", "template_id": "follow_up_02"}
```

### Step 2.6 — Run and manually review

```powershell
python scripts/generate_synthetic_data.py --train 4000 --val 500 --test 500 --seed 42
```

Then **read 20 random notes yourself.** Ask for each:
- Would a doctor recognize this as a plausible (if simple) note?
- Is every labeled code actually supported by the note text?
- Do any two notes read identically? (If yes, add more variability.)

### Step 2.7 — Write a stats printout

At the end of generation, print: notes per split, ICD code distribution, % multi-label, avg note length. Save to `data/dataset_stats.json`. You'll cite these numbers on your resume.

### ✅ Done check (Milestone 2)

- [ ] `data/train.jsonl` ~4,000 lines, `val.jsonl` ~500, `test.jsonl` ~500
- [ ] Every ICD code appears in all 3 splits
- [ ] Test set contains template variants absent from train
- [ ] 20-note manual review passed
- [ ] `data/dataset_stats.json` saved
- [ ] Committed to git
- [ ] **Resume bullet 2 ("Built a reproducible synthetic-data pipeline...") is now TRUE**

---

## Milestone 3 — Baseline Evaluation

**Time:** 1 evening (~3 hours) + GPU inference time | **Needs:** Milestone 2 done, Kaggle or Colab account

### Step 3.1 — Write `src/prompts.py`

One function, used EVERYWHERE (training prep, baseline, fine-tuned eval, demo):

```python
def build_prompt(note: str) -> str:
    return f"""### Clinical note:
{note}

### Billing codes (JSON only):
"""
```

Never fork this template. Train/inference mismatch is the #1 silent killer of fine-tune results.

### Step 3.2 — Write `src/validate.py`

Functions:
- `parse_model_output(text) -> dict | None` — extract first JSON object from generated text, return None on failure
- `filter_codes(parsed, icd_whitelist, cpt_whitelist) -> dict` — keep only whitelisted codes, dedupe, preserve order
- Track and return both **pre-filter** and **post-filter** code lists (you need the pre-filter list to measure hallucination rate)

### Step 3.3 — Write `scripts/eval.py`

Takes `--predictions <file> --gold data/test.jsonl --output results/<name>.json`. Computes:

- Macro-F1 and micro-F1 for ICD and CPT separately (use `sklearn.metrics` with multi-label binarization over the whitelist vocabulary)
- Exact-set match rate
- Top-1 / top-3 recall for primary ICD (order predictions as generated)
- JSON validity rate (from the generation log)
- Invalid-code rate pre-filter

Keep generation and scoring separate: a `run_model.py`/notebook produces a predictions JSONL, `eval.py` scores it. This lets you re-score without re-running the GPU.

### Step 3.4 — Run baseline generation on Colab

Create a Colab notebook on a **T4 runtime** (cheap on credits — don't use A100 for inference):

1. Upload `data/test.jsonl` + `codes/*.json` (Drive mount or direct upload)
2. Load `google/gemma-4-E4B-it` in 4-bit (wrap prompts with `tokenizer.apply_chat_template` — Gemma expects `<start_of_turn>` formatting)
3. For each test note: `build_prompt(note)` → generate (temperature 0, max_new_tokens ~128) → save raw output
4. Download `predictions_baseline.jsonl`

Expect ~30–60 min for 500 notes on a T4 (~1–2 credits). Kaggle free T4 works too.

### Step 3.5 — Score it

```powershell
python scripts/eval.py --predictions predictions_baseline.jsonl --gold data/test.jsonl --output results/baseline.json
```

### Step 3.6 — Record the headline numbers

Note these three — they're your interview material:
1. Baseline ICD macro-F1 (expect ~0.2–0.4)
2. JSON validity rate (expect 80–95%)
3. **Invalid-code (hallucination) rate** — expect 5–20%. This is your best stat.

### ✅ Done check (Milestone 3)

- [ ] `results/baseline.json` exists with all metrics
- [ ] Hallucination rate measured pre-filter
- [ ] Prompt template frozen in `src/prompts.py`
- [ ] Committed to git
- [ ] **Resume bullet about "quantified X% invalid-code hallucination rate" is now TRUE**

---

## Milestone 4 — QLoRA Fine-tuning

**Time:** 1–2 evenings + ~2–4 GPU hours | **Needs:** Milestone 3 done, HuggingFace account with write token

### Step 4.1 — Prepare training data

Write `scripts/prepare_training_data.py`:

- For each train/val record: `text = build_prompt(note) + json.dumps({"icd10": [...], "cpt": [...]})`
- Use the SAME `build_prompt` from `src/prompts.py`
- Output `data/train_sft.jsonl` and `data/val_sft.jsonl` with a single `text` field
- Spot-check 3 lines by eye: prompt and completion should join seamlessly

### Step 4.2 — Create the Colab training notebook

`notebooks/train_qlora.ipynb` (Colab, using your compute credits). Cells in order:

1. `pip install -q transformers peft bitsandbytes trl accelerate datasets`
2. Login to HF Hub (`huggingface_hub.login()` with your write token stored as a Colab Secret)
3. Load `google/gemma-4-E4B-it` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` — start from the official Gemma 4 TRL fine-tuning recipe (HF blog / Google docs)
4. LoRA config: `r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05, task_type="CAUSAL_LM"`
5. `SFTTrainer` (trl) with: epochs=2, lr=2e-4, batch=8 + grad accum 2 (A100; use batch=4 + grad accum 4 on T4), max_seq_length=2048 (A100; 1024 on T4), warmup_ratio=0.05, `eval_strategy="epoch"`, `save_strategy="epoch"`, `push_to_hub=True`, `hub_model_id="your-username/gemma4-icd-cpt-qlora"`
   - Format each sample with `tokenizer.apply_chat_template` (user turn = clinical note prompt, model turn = JSON codes) so training matches Gemma's expected chat format
6. `trainer.train()`

### Step 4.3 — Enable GPU and launch

- **Write and debug the notebook on a T4 (or CPU) runtime first** — credits burn per connected hour, not per computed hour
- When ready: Runtime → Change runtime type → **A100**, then run everything top to bottom
- Expect ~30–45 min for 4,000 samples × 2 epochs on A100 (~6–9 credits); ~1–3 hours on T4
- **Checkpoint every epoch pushes to HF Hub** — if the session dies, you resume from the last epoch
- **Disconnect the runtime as soon as training finishes** to stop credit burn
- Fallback: Kaggle free T4 (30 GPU hrs/week) works with the T4 settings above if credits run out

### Step 4.4 — Sanity-check the run

While/after training, verify:
- Train loss decreases smoothly (not flat, not instant-zero)
- Val loss decreases and doesn't sharply rise (rise = overfitting → stop at earlier checkpoint)
- If loss is flat: check the `text` field format — most common bug is a malformed prompt join

### Step 4.5 — Quick smoke test

In the same notebook, load the adapter and run 3 test notes through generation. Output should be clean JSON with plausible codes. If it outputs prose or garbage, the training format is wrong — fix before Milestone 5.

### ✅ Done check (Milestone 4)

- [ ] Adapter on HF Hub (`your-username/gemma4-icd-cpt-qlora`)
- [ ] Loss curves sane (screenshot/save them — README material)
- [ ] 3-note smoke test produces valid JSON
- [ ] Notebook committed to git

---

## Milestone 5 — Fine-tuned Eval + Comparison

**Time:** 1 evening (~2–3 hours) + ~1 GPU hour | **Needs:** Milestone 4 done

### Step 5.1 — Generate predictions with the adapter

Same generation notebook as Step 3.4, with ONE change: load the LoRA adapter on top of the base model (`PeftModel.from_pretrained(base, "your-username/gemma4-icd-cpt-qlora")`).

Same test set, same prompt, same temperature 0, same max tokens. Download `predictions_finetuned.jsonl`.

### Step 5.2 — Score with the identical eval script

```powershell
python scripts/eval.py --predictions predictions_finetuned.jsonl --gold data/test.jsonl --output results/finetuned.json
```

### Step 5.3 — Write `results/comparison.md`

Structure:
1. Table: every metric, baseline column vs fine-tuned column, delta column
2. One bar chart (matplotlib: baseline vs fine-tuned for ICD macro-F1, CPT macro-F1, exact-set match) → save `results/comparison_chart.png`
3. 3–5 sentences of interpretation: where the model improved most, where it's still weak

### Step 5.4 — Failure analysis

Script or manual: find the 10 test notes with worst prediction overlap. For each, record note, gold codes, predicted codes, and a one-line diagnosis (missed secondary code / wrong CPT complexity / hallucination / JSON failure). Save `results/failure_cases.json`.

Common patterns you'll likely find (and can discuss in interviews):
- Secondary diagnoses mentioned briefly get missed
- CPT complexity levels (99213 vs 99214) confuse the model
- Rare codes underperform vs common ones (macro-F1 shows this)

### Step 5.5 — Decision gate

- If ICD macro-F1 gain **≥ 0.15**: proceed to Milestone 6. Ship it.
- If less: debug in this order — (1) prompt mismatch between train and inference, (2) JSON parsing eating valid outputs, (3) training loss never converged, (4) labels not supported by note text. Fix, re-run Milestone 4, re-eval. Do NOT tune on test set.

### ✅ Done check (Milestone 5)

- [ ] `results/finetuned.json`, `comparison.md`, `comparison_chart.png`, `failure_cases.json` all exist
- [ ] Real numbers for the pitch: "macro-F1 improved from X → Y"
- [ ] Committed to git
- [ ] **Resume bullet 1 upgrades from "Fine-tuning" to "Fine-tuned ... improved macro-F1 from X → Y"**

---

## Milestone 6 — Demo + README + Ship

**Time:** 1–2 evenings (~4–5 hours) | **Needs:** Milestone 5 done

### Step 6.1 — Write `src/inference.py`

Single entry point used by the demo:

```python
def extract_codes(note: str) -> dict:
    # build_prompt → generate → parse_model_output → filter_codes
    # one retry on parse failure; return {"icd10": [...], "cpt": [...], "descriptions": {...}}
```

### Step 6.2 — Build `app.py` (Gradio)

- `gr.Textbox` (note input, 8 lines) → button → two `gr.Dataframe` outputs (ICD table: code + description; CPT table: code + description)
- 3 preset examples via `gr.Examples`: diabetes follow-up, URI urgent care, annual wellness — **written by hand, NOT from your dataset**
- Test locally first with the adapter loaded from HF Hub

### Step 6.3 — Deploy to HuggingFace Spaces

1. Create Space (Gradio SDK, free CPU tier — or ZeroGPU if available)
2. Push `app.py`, Space `requirements.txt`, and `codes/*.json`
3. CPU inference for Gemma 4 E4B 4-bit is slow (~1–2 min/request on free tier) — acceptable for a demo; note it in the Space description. If too slow, train a Gemma 4 E2B adapter for the demo (same data, ~half the size)
4. Verify the public URL works from another device/incognito

### Step 6.4 — Write the portfolio README

Follow `IMPLEMENTATION_PLAN.md` Section 16. Order matters:
1. **First line = metric pitch with real numbers**
2. Demo link + GIF/screenshot
3. Problem (2 short paragraphs)
4. Results table + chart
5. Approach (data → QLoRA → whitelist validation)
6. Reproduce instructions (the 6 commands)
7. Error analysis (3 best failure cases from `failure_cases.json`)
8. Limitations (synthetic data, 50-code vocabulary, not clinical advice) + Phase 2 roadmap

### Step 6.5 — Final validation pass

- [ ] Run all 6 quick-reference commands from scratch — do they work?
- [ ] Demo handles: empty input, non-medical text ("my car broke down"), a real-looking note
- [ ] Every number on your resume matches a file in `results/`
- [ ] Push final repo to GitHub (public)

### Step 6.6 — Publish

- GitHub repo public with README
- HF Hub: adapter model card links back to GitHub
- Optional: LinkedIn post — lead with the metric sentence, link to the Space

### ✅ Done check (Milestone 6) — SHIPPED

- [ ] Public GitHub repo, public HF Space, adapter on HF Hub
- [ ] README first line has real X → Y numbers
- [ ] Demo verified on 3 hand-written notes
- [ ] Resume bullets final and 100% backed by artifacts

---

## Progress Tracker

| Milestone | Status | Date done | Key artifact |
|-----------|--------|-----------|--------------|
| 1 — Scaffold + code tables | ☐ Not started | | `codes/*.json` |
| 2 — Synthetic data | ☐ Not started | | `data/*.jsonl` |
| 3 — Baseline eval | ☐ Not started | | `results/baseline.json` |
| 4 — QLoRA training | ☐ Not started | | Adapter on HF Hub |
| 5 — Fine-tuned eval | ☐ Not started | | `results/comparison.md` |
| 6 — Demo + ship | ☐ Not started | | Public Space URL |

Update this table as you go — it doubles as your accountability log.
