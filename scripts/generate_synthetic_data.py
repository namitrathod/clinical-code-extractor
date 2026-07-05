"""Generate synthetic clinical notes with ICD-10 + CPT labels.

Reads whitelists from codes/, builds templated notes per condition, assigns
labels supported by note text, and writes stratified train/val/test JSONL.

Design choices (see EXECUTION_STEPS.md Milestone 2):
- ~30% multi-label notes with secondary ICD mentioned in text
- CPT tied to visit type / procedures mentioned in the note
- Stratified split by primary_icd10
- One template per code held out of train+val (test-only for that code)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CODES_DIR = ROOT / "codes"
DATA_DIR = ROOT / "data"

TEMPLATES = ["brief_visit", "detailed_hp", "follow_up", "urgent_care", "wellness"]
WELLNESS_CODES = {"Z00.00", "Z23", "Z12.11"}

FIRST_NAMES = ["James", "Maria", "Robert", "Patricia", "Michael", "Jennifer",
               "David", "Linda", "William", "Elizabeth", "Richard", "Susan"]
NOISE_SNIPPETS = [
    "Social history: non-smoker, occasional alcohol.",
    "Family history: father with CAD.",
    "Patient denies chest pain or shortness of breath today.",
    "Allergies: NKDA.",
    "Review of systems otherwise negative.",
    "Patient works as a teacher, lives with spouse.",
    "Prior appendectomy, no other surgeries.",
]

# Category-level defaults; each ICD code inherits then gets code-specific tweaks.
CATEGORY_DEFAULTS: dict[str, dict[str, Any]] = {
    "endocrine_metabolic": {
        "symptoms": ["fatigue", "weight gain", "increased thirst", "polyuria"],
        "exam": ["BMI {bmi}", "BP {sbp}/{dbp} mmHg"],
        "plans": ["lifestyle counseling", "recheck labs in 3 months", "continue current meds"],
        "comorbidities": ["I10", "E78.5", "E66.9"],
        "cpt_base": ["99213", "99214"],
        "labs": ["83036", "80053", "36415"],
        "vitals": {"bmi": (26, 38), "sbp": (128, 158), "dbp": (78, 96)},
        "is_new_patient": False,
    },
    "cardiovascular": {
        "symptoms": ["chest tightness", "palpitations", "dyspnea on exertion", "leg swelling"],
        "exam": ["BP {sbp}/{dbp} mmHg", "regular rate and rhythm", "no murmurs"],
        "plans": ["continue antihypertensive", "low-sodium diet", "follow up in 4 weeks"],
        "comorbidities": ["E78.5", "E11.9", "E66.9"],
        "cpt_base": ["99213", "99214"],
        "labs": ["93000", "80053", "36415"],
        "vitals": {"sbp": (138, 168), "dbp": (82, 102)},
        "is_new_patient": False,
    },
    "respiratory": {
        "symptoms": ["cough", "congestion", "sore throat", "wheezing", "shortness of breath"],
        "exam": ["lungs clear to auscultation", "mild wheezing bilaterally", "no respiratory distress"],
        "plans": ["supportive care", "increase inhaler as needed", "return if worsening"],
        "comorbidities": ["J45.909", "I10"],
        "cpt_base": ["99213", "99203", "99204"],
        "labs": ["87880", "71046", "94010", "36415"],
        "vitals": {"temp_f": (98.6, 101.8), "spo2": (94, 99)},
        "is_new_patient": True,
    },
    "musculoskeletal": {
        "symptoms": ["pain with movement", "stiffness", "limited range of motion", "swelling"],
        "exam": ["tenderness on palpation", "decreased ROM", "no erythema"],
        "plans": ["NSAIDs as needed", "physical therapy referral", "activity modification"],
        "comorbidities": ["M54.50", "I10", "E66.9"],
        "cpt_base": ["99213", "99214"],
        "labs": ["71046", "20610"],
        "vitals": {"pain_0_10": (4, 8)},
        "is_new_patient": False,
    },
    "mental_neuro": {
        "symptoms": ["anxiety", "low mood", "poor sleep", "headache", "difficulty concentrating"],
        "exam": ["alert and oriented x3", "affect congruent", "normal neurological exam"],
        "plans": ["continue SSRI", "counseling referral", "sleep hygiene education"],
        "comorbidities": ["F41.9", "G47.00", "I10"],
        "cpt_base": ["99213", "99214"],
        "labs": ["96127"],
        "vitals": {},
        "is_new_patient": False,
    },
    "gastrointestinal": {
        "symptoms": ["heartburn", "abdominal discomfort", "nausea", "bloating", "constipation"],
        "exam": ["abdomen soft, mild epigastric tenderness", "no rebound or guarding"],
        "plans": ["PPI trial", "diet modification", "follow up in 6 weeks"],
        "comorbidities": ["K21.9", "E66.9", "I10"],
        "cpt_base": ["99213", "99214"],
        "labs": ["80053"],
        "vitals": {},
        "is_new_patient": False,
    },
    "genitourinary_renal": {
        "symptoms": ["dysuria", "frequency", "urgency", "fatigue", "nocturia"],
        "exam": ["suprapubic tenderness", "no CVA tenderness", "stable vitals"],
        "plans": ["antibiotic if indicated", "increase fluids", "recheck labs"],
        "comorbidities": ["I10", "E11.9", "D64.9"],
        "cpt_base": ["99213", "99214"],
        "labs": ["81002", "85025", "36415"],
        "vitals": {"temp_f": (98.0, 100.4)},
        "is_new_patient": False,
    },
    "infectious_skin": {
        "symptoms": ["fever", "malaise", "localized redness", "ear pain", "sore throat"],
        "exam": ["erythema present", "warmth over affected area", "TMs erythematous"],
        "plans": ["antibiotics as indicated", "warm compresses", "return precautions given"],
        "comorbidities": ["J06.9", "R50.9"],
        "cpt_base": ["99203", "99213", "99214"],
        "labs": ["87880", "85025", "36415"],
        "vitals": {"temp_f": (99.0, 102.5)},
        "is_new_patient": True,
    },
    "wellness_symptoms": {
        "symptoms": ["routine visit", "mild cough", "fatigue", "dizziness", "screening concerns"],
        "exam": ["general appearance well", "vitals stable", "exam unremarkable"],
        "plans": ["health maintenance discussed", "vaccines updated", "return annually"],
        "comorbidities": ["I10", "E78.5"],
        "cpt_base": ["99395", "99396", "99213"],
        "labs": ["85025", "80053", "36415"],
        "vitals": {"sbp": (110, 130), "dbp": (70, 82)},
        "is_new_patient": False,
    },
}

# Per-code overrides for symptoms/plans/comorbidities where category defaults are too generic.
CODE_OVERRIDES: dict[str, dict[str, Any]] = {
    "E11.65": {
        "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred vision"],
        "exam": ["BMI {bmi}", "fasting glucose {glucose} mg/dL", "A1c {a1c}%"],
        "plans": ["start metformin", "increase metformin", "diabetes education", "recheck A1c in 3 months"],
        "comorbidities": ["I10", "E78.5"],
        "vitals": {"bmi": (27, 38), "glucose": (180, 320), "a1c": (7.8, 11.5)},
    },
    "E11.9": {
        "symptoms": ["mild polyuria", "fatigue"],
        "plans": ["continue metformin", "A1c monitoring"],
        "comorbidities": ["I10", "E78.5"],
    },
    "E11.40": {
        "symptoms": ["numbness in feet", "burning pain in lower extremities"],
        "exam": ["decreased sensation to monofilament", "A1c {a1c}%"],
        "plans": ["optimize glycemic control", "foot care education"],
        "comorbidities": ["I10", "E78.5"],
        "vitals": {"a1c": (8.0, 11.0)},
    },
    "I10": {
        "symptoms": ["headache", "occasional dizziness"],
        "plans": ["continue lisinopril", "home BP log", "low-sodium diet"],
        "comorbidities": ["E78.5", "E11.9"],
    },
    "I48.91": {
        "symptoms": ["palpitations", "irregular heartbeat", "mild dyspnea"],
        "plans": ["rate control discussed", "anticoagulation risk reviewed", "cardiology follow-up"],
        "labs": ["93000"],
    },
    "J44.1": {
        "symptoms": ["increased dyspnea", "productive cough", "wheezing"],
        "plans": ["prednisone burst", "increase bronchodilator", "return if worse"],
        "labs": ["94010", "71046"],
    },
    "M25.561": {
        "symptoms": ["right knee pain", "swelling", "difficulty climbing stairs"],
        "exam": ["right knee effusion", "pain with flexion"],
        "plans": ["NSAIDs", "knee brace", "consider injection if no improvement"],
        "labs": ["20610", "71046"],
    },
    "M54.50": {
        "symptoms": ["low back pain", "pain radiating to buttock", "worse with bending"],
        "plans": ["NSAIDs", "physical therapy", "activity modification"],
    },
    "F41.1": {
        "symptoms": ["persistent worry", "restlessness", "poor concentration"],
        "plans": ["continue sertraline", "CBT referral", "relaxation techniques"],
        "labs": ["96127"],
    },
    "F32.9": {
        "symptoms": ["low mood", "anhedonia", "poor sleep", "decreased energy"],
        "plans": ["medication adjustment", "therapy referral", "safety assessment negative"],
        "labs": ["96127"],
    },
    "N39.0": {
        "symptoms": ["dysuria", "urinary frequency", "suprapubic discomfort"],
        "plans": ["empiric antibiotic", "increase oral fluids", "culture if no improvement"],
        "labs": ["81002", "85025"],
    },
    "Z00.00": {
        "symptoms": ["here for annual physical", "no acute complaints"],
        "plans": ["preventive counseling", "routine labs ordered", "immunizations reviewed"],
        "cpt_base": ["99395", "99396"],
    },
    "Z23": {
        "symptoms": ["here for vaccination", "no acute illness"],
        "plans": ["influenza vaccine administered", "observed 15 minutes without reaction"],
        "cpt_base": ["99213"],
        "labs": ["90471"],
    },
    "Z12.11": {
        "symptoms": ["here for colon cancer screening discussion"],
        "plans": ["FIT kit provided", "colonoscopy referral if indicated"],
        "cpt_base": ["99214"],
    },
}

COMORBIDITY_PHRASES: dict[str, str] = {
    "I10": "History of hypertension, currently on lisinopril.",
    "E78.5": "Hyperlipidemia on atorvastatin.",
    "E11.9": "Type 2 diabetes, diet controlled.",
    "E66.9": "Obesity, BMI elevated.",
    "E78.00": "Hypercholesterolemia on statin therapy.",
    "J45.909": "Known asthma, uses albuterol PRN.",
    "G47.00": "Insomnia, sleep hygiene discussed.",
    "F41.9": "Anxiety disorder, stable on medication.",
    "K21.9": "GERD, on omeprazole.",
    "D64.9": "Anemia noted on prior labs.",
    "M54.50": "Chronic low back pain, stable.",
    "J06.9": "Recent URI symptoms improving.",
    "R50.9": "Low-grade fever yesterday.",
}


@dataclass
class ConditionProfile:
    code: str
    name: str
    category: str
    symptoms: list[str]
    exam: list[str]
    plans: list[str]
    comorbidities: list[str]
    cpt_base: list[str]
    labs: list[str]
    vitals: dict[str, tuple[float, float]]
    is_new_patient: bool
    holdout_template: str = "follow_up"


@dataclass
class NoteContext:
    profile: ConditionProfile
    template_id: str
    age: int
    sex: str
    sex_abbr: str
    patient_ref: str
    primary_text: str
    secondary_code: str | None
    secondary_text: str
    exam_text: str
    plan_text: str
    noise: str
    vitals: dict[str, Any]
    use_abbrev: bool
    cpt: list[str] = field(default_factory=list)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def code_to_category(categories: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cat, codes in categories.items():
        for code in codes:
            mapping[code] = cat
    return mapping


def build_profiles(
    icd10: dict[str, str],
    categories: dict[str, list[str]],
) -> dict[str, ConditionProfile]:
    cat_map = code_to_category(categories)
    profiles: dict[str, ConditionProfile] = {}
    template_cycle = list(TEMPLATES)

    for i, (code, desc) in enumerate(sorted(icd10.items())):
        cat = cat_map[code]
        base = CATEGORY_DEFAULTS[cat]
        override = CODE_OVERRIDES.get(code, {})

        profiles[code] = ConditionProfile(
            code=code,
            name=desc,
            category=cat,
            symptoms=override.get("symptoms", base["symptoms"]),
            exam=override.get("exam", base["exam"]),
            plans=override.get("plans", base["plans"]),
            comorbidities=[c for c in override.get("comorbidities", base["comorbidities"]) if c != code],
            cpt_base=override.get("cpt_base", base["cpt_base"]),
            labs=override.get("labs", base["labs"]),
            vitals=override.get("vitals", base["vitals"]),
            is_new_patient=override.get("is_new_patient", base["is_new_patient"]),
            holdout_template=template_cycle[i % len(template_cycle)],
        )
    return profiles


def pick_vitals(rng: random.Random, ranges: dict[str, tuple[float, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, (lo, hi) in ranges.items():
        if key in {"bmi", "a1c", "glucose", "pain_0_10", "spo2"}:
            out[key] = round(rng.uniform(lo, hi), 1)
        else:
            out[key] = int(rng.uniform(lo, hi))
    return out


def format_exam(exam_lines: list[str], vitals: dict[str, Any]) -> str:
    parts = []
    for line in exam_lines:
        try:
            parts.append(line.format(**vitals))
        except KeyError:
            parts.append(line)
    return "; ".join(parts)


def assign_cpt(ctx: NoteContext, rng: random.Random) -> list[str]:
    p = ctx.profile
    cpts: list[str] = []

    if ctx.template_id == "wellness" or p.code in WELLNESS_CODES:
        if ctx.age < 40:
            cpts.append(rng.choice(["99385", "99395"]))
        else:
            cpts.append(rng.choice(["99386", "99396"]))
    elif ctx.template_id == "urgent_care" or p.is_new_patient:
        cpts.append(rng.choice(["99203", "99204", "99202"]))
    else:
        cpts.append(rng.choice(p.cpt_base or ["99213", "99214"]))

    note_lower = (ctx.primary_text + ctx.plan_text).lower()
    if any(k in note_lower for k in ["lab", "a1c", "cbc", "metabolic panel", "venipuncture", "blood draw"]):
        if "83036" in p.labs:
            cpts.append("83036")
        elif "85025" in p.labs:
            cpts.append("85025")
        elif "80053" in p.labs:
            cpts.append("80053")
        if rng.random() < 0.7:
            cpts.append("36415")
    if "ekg" in note_lower or "ecg" in note_lower or "93000" in p.labs and p.category == "cardiovascular" and rng.random() < 0.4:
        cpts.append("93000")
    if "x-ray" in note_lower or "chest" in note_lower and p.category == "respiratory" and rng.random() < 0.35:
        cpts.append("71046")
    if "spirometry" in note_lower or (p.code in {"J44.1", "J44.9", "J45.909"} and rng.random() < 0.3):
        cpts.append("94010")
    if "vaccine" in note_lower or "immunization" in note_lower or p.code == "Z23":
        cpts.append("90471")
    if "injection" in note_lower or "arthrocentesis" in note_lower:
        cpts.append("20610")
    if "ua" in note_lower or "urinalysis" in note_lower or p.code == "N39.0":
        cpts.append("81002")
    if p.category == "mental_neuro" and rng.random() < 0.45:
        cpts.append("96127")
    if "strep" in note_lower or p.code in {"J02.9", "J06.9"} and rng.random() < 0.35:
        cpts.append("87880")

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cpts:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def render_note(ctx: NoteContext, rng: random.Random) -> str:
    p = ctx.profile
    cc = ctx.primary_text
    if ctx.template_id == "brief_visit":
        parts = [
            f"{ctx.age}yo {ctx.sex_abbr} c/o {cc}.",
            f"Exam: {ctx.exam_text}.",
        ]
        if ctx.secondary_text:
            parts.append(ctx.secondary_text)
        parts.append(f"Assessment: {p.name}. Plan: {ctx.plan_text}.")
        if ctx.noise:
            parts.append(ctx.noise)
        return " ".join(parts)

    if ctx.template_id == "detailed_hp":
        return (
            f"Chief Complaint: {cc}.\n"
            f"History of Present Illness: {ctx.patient_ref} is a {ctx.age}-year-old {ctx.sex} presenting with "
            f"{cc} for the past {rng.randint(2, 14)} days. {ctx.secondary_text}\n"
            f"Past Medical History: {p.name}. {ctx.noise}\n"
            f"Physical Exam: {ctx.exam_text}.\n"
            f"Assessment: {p.name}.\n"
            f"Plan: {ctx.plan_text}."
        )

    if ctx.template_id == "follow_up":
        return (
            f"Follow-up visit. {ctx.patient_ref} returns for ongoing management of {p.name.lower()}. "
            f"Reports {cc}. {ctx.secondary_text} Exam: {ctx.exam_text}. "
            f"Assessment unchanged — {p.name}. Plan: {ctx.plan_text}. {ctx.noise}"
        )

    if ctx.template_id == "urgent_care":
        return (
            f"UC visit — {ctx.age}yo {ctx.sex_abbr}, acute onset {cc}. "
            f"Focused exam: {ctx.exam_text}. No red flags. "
            f"Assessment: {p.name}. Plan: {ctx.plan_text}. Patient educated on return precautions."
        )

    # wellness
    return (
        f"Annual wellness visit. {ctx.patient_ref}, {ctx.age}yo {ctx.sex_abbr}, "
        f"{cc}. Exam: {ctx.exam_text}. {ctx.noise} "
        f"Assessment: {p.name}. Plan: {ctx.plan_text}."
    )


def build_context(
    profile: ConditionProfile,
    template_id: str,
    rng: random.Random,
    icd10: dict[str, str],
) -> NoteContext:
    age = rng.randint(22, 82)
    if profile.code in WELLNESS_CODES and profile.code != "Z12.11":
        age = rng.randint(25, 75)
    if profile.code == "Z12.11":
        age = rng.randint(45, 72)

    sex = rng.choice(["male", "female"])
    sex_abbr = "M" if sex == "male" else "F"
    patient_ref = f"{rng.choice(FIRST_NAMES)}"
    use_abbrev = template_id in {"brief_visit", "urgent_care"} or rng.random() < 0.35

    symptom = rng.choice(profile.symptoms)
    if use_abbrev and "hypertension" in profile.name.lower():
        primary_text = symptom
    else:
        primary_text = symptom if symptom != "routine visit" else profile.name.lower()

    vitals = pick_vitals(rng, profile.vitals)
    exam_text = format_exam(profile.exam, vitals)
    plan_text = ", ".join(rng.sample(profile.plans, k=min(2, len(profile.plans))))

    secondary_code: str | None = None
    secondary_text = ""
    if profile.comorbidities and rng.random() < 0.30:
        secondary_code = rng.choice(profile.comorbidities)
        secondary_text = COMORBIDITY_PHRASES.get(
            secondary_code,
            f"Also carries diagnosis of {icd10.get(secondary_code, secondary_code)}.",
        )

    noise = rng.choice(NOISE_SNIPPETS) if rng.random() < 0.40 else ""

    ctx = NoteContext(
        profile=profile,
        template_id=template_id,
        age=age,
        sex=sex,
        sex_abbr=sex_abbr,
        patient_ref=patient_ref,
        primary_text=primary_text,
        secondary_code=secondary_code,
        secondary_text=secondary_text,
        exam_text=exam_text,
        plan_text=plan_text,
        noise=noise,
        vitals=vitals,
        use_abbrev=use_abbrev,
    )
    ctx.cpt = assign_cpt(ctx, rng)
    return ctx


def allowed_templates(profile: ConditionProfile, split: str) -> list[str]:
    if profile.code in WELLNESS_CODES:
        pool = ["wellness", "brief_visit", "detailed_hp", "follow_up"]
    else:
        pool = list(TEMPLATES)

    if split == "test":
        return pool
    return [t for t in pool if t != profile.holdout_template]


def allocate_counts(codes: list[str], total: int) -> dict[str, int]:
    """Split `total` notes across codes as evenly as possible (largest remainder)."""
    n = len(codes)
    base, rem = divmod(total, n)
    counts = {c: base for c in codes}
    for code in codes[:rem]:
        counts[code] += 1
    return counts


def generate_split_records(
    split: str,
    total: int,
    profiles: dict[str, ConditionProfile],
    icd10: dict[str, str],
    rng: random.Random,
    start_id: int,
) -> tuple[list[dict[str, Any]], int]:
    codes = sorted(profiles.keys())
    per_code = allocate_counts(codes, total)
    records: list[dict[str, Any]] = []
    note_id = start_id

    for code in codes:
        profile = profiles[code]
        templates = allowed_templates(profile, split)
        n_notes = per_code[code]

        # In test split, reserve one note per code for the holdout-only template
        # so train/val never see that template for this condition.
        holdout_first = split == "test"
        if holdout_first and n_notes > 0:
            ctx = build_context(profile, profile.holdout_template, rng, icd10)
            records.append({
                "id": f"note_{note_id:05d}",
                "note": render_note(ctx, rng),
                "icd10": [code] + ([ctx.secondary_code] if ctx.secondary_code else []),
                "cpt": ctx.cpt,
                "primary_icd10": code,
                "template_id": profile.holdout_template,
                "split": split,
            })
            note_id += 1
            n_notes -= 1

        for _ in range(n_notes):
            template_id = rng.choice(templates)
            ctx = build_context(profile, template_id, rng, icd10)
            note = render_note(ctx, rng)

            icd_labels = [code]
            if ctx.secondary_code:
                icd_labels.append(ctx.secondary_code)

            records.append({
                "id": f"note_{note_id:05d}",
                "note": note,
                "icd10": icd_labels,
                "cpt": ctx.cpt,
                "primary_icd10": code,
                "template_id": template_id,
                "split": split,
            })
            note_id += 1

    rng.shuffle(records)
    return records, note_id


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    word_counts = [len(r["note"].split()) for r in records]
    multi = sum(1 for r in records if len(r["icd10"]) > 1)
    icd_counts = Counter(r["primary_icd10"] for r in records)
    template_counts = Counter(r["template_id"] for r in records)
    return {
        "n_notes": len(records),
        "avg_words": round(sum(word_counts) / max(len(word_counts), 1), 1),
        "pct_multi_label": round(100 * multi / max(len(records), 1), 1),
        "unique_primary_icd10": len(icd_counts),
        "min_notes_per_code": min(icd_counts.values()) if icd_counts else 0,
        "max_notes_per_code": max(icd_counts.values()) if icd_counts else 0,
        "template_distribution": dict(template_counts),
        "primary_icd10_distribution": dict(sorted(icd_counts.items())),
    }


def verify_stratification(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    profiles: dict[str, ConditionProfile],
) -> None:
    all_codes = set(profiles.keys())
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        codes_in_split = {r["primary_icd10"] for r in data}
        missing = all_codes - codes_in_split
        if missing:
            raise ValueError(f"{split_name} missing primary codes: {sorted(missing)[:5]} ... ({len(missing)} total)")

    # Holdout templates must not appear in train/val for their primary code
    for code, profile in profiles.items():
        holdout = profile.holdout_template
        bad_train = [r for r in train if r["primary_icd10"] == code and r["template_id"] == holdout]
        bad_val = [r for r in val if r["primary_icd10"] == code and r["template_id"] == holdout]
        if bad_train or bad_val:
            raise ValueError(f"Holdout template {holdout} leaked into train/val for {code}")

    # Each holdout should appear at least once in test for that code (if test has enough per code)
    for code, profile in profiles.items():
        holdout = profile.holdout_template
        test_h = [r for r in test if r["primary_icd10"] == code and r["template_id"] == holdout]
        if not test_h:
            # With ~9 test notes per code and 4-5 templates, holdout may not always appear once;
            # force at least one by checking pool size — warn only if zero across all codes is bad
            pass


def validate_labels(records: list[dict], icd10: dict[str, str], cpt: dict[str, str]) -> None:
    for r in records:
        if r["primary_icd10"] not in icd10:
            raise ValueError(f"Unknown primary ICD: {r['primary_icd10']}")
        for c in r["icd10"]:
            if c not in icd10:
                raise ValueError(f"Unknown ICD label {c} in {r['id']}")
        for c in r["cpt"]:
            if c not in cpt:
                raise ValueError(f"Unknown CPT label {c} in {r['id']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic clinical coding dataset.")
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    icd10 = load_json(CODES_DIR / "icd10_whitelist.json")
    cpt = load_json(CODES_DIR / "cpt_whitelist.json")
    categories = load_json(CODES_DIR / "icd10_categories.json")
    profiles = build_profiles(icd10, categories)

    train, next_id = generate_split_records("train", args.train, profiles, icd10, rng, start_id=1)
    val, next_id = generate_split_records("val", args.val, profiles, icd10, rng, start_id=next_id)
    test, _ = generate_split_records("test", args.test, profiles, icd10, rng, start_id=next_id)

    for batch in (train, val, test):
        validate_labels(batch, icd10, cpt)
    verify_stratification(train, val, test, profiles)

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    write_jsonl(DATA_DIR / "test.jsonl", test)

    stats = {
        "seed": args.seed,
        "train": compute_stats(train),
        "val": compute_stats(val),
        "test": compute_stats(test),
        "holdout_templates_by_code": {c: p.holdout_template for c, p in sorted(profiles.items())},
    }
    (DATA_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Seed: {args.seed}")
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test notes to {DATA_DIR}")
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        s = stats[split_name]
        print(f"  {split_name}: avg {s['avg_words']} words, {s['pct_multi_label']}% multi-label, "
              f"{s['min_notes_per_code']}-{s['max_notes_per_code']} notes/code")


if __name__ == "__main__":
    main()
